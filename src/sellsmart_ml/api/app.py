import os
import jwt

from fastapi import FastAPI, Query, Header, HTTPException, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from jwt import PyJWKClient

from sellsmart_ml.storage.supabase_predictions import get_latest_prediction, save_latest_prediction
from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk
from sellsmart_ml.storage.supabase_predictions import get_all_latest_predictions
from sellsmart_ml.storage.supabase_prediction_jobs import (
    create_prediction_job,
    get_active_prediction_job,
    get_prediction_job,
    serialize_prediction_job,
)
from sellsmart_ml.jobs.prediction_worker import process_prediction_job
from typing import Optional

from sellsmart_ml.services.symbol_search import search_symbols


SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
SUPABASE_ISSUER = f"{SUPABASE_URL}/auth/v1"

jwks_client = PyJWKClient(SUPABASE_JWKS_URL)


def require_user(authorization: Optional[str] = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
            issuer=SUPABASE_ISSUER,
        )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


app = FastAPI(title="SellSmart Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://sellsmart.asia",
        "https://www.sellsmart.asia",
        "https://sellsmart-ui.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    background_tasks: BackgroundTasks,
    response: Response,
    ticker: str = Query(..., min_length=1),
    live: bool = Query(False),
    queued: bool = Query(False),
    force_refresh: bool = Query(False),
    user=Depends(require_user),
):
    ticker = ticker.upper().strip()

    # Always prefer a fresh Supabase prediction unless the caller explicitly asks
    # to regenerate. This is important for portfolio/watchlist loading: even if an
    # older frontend still sends live=true, fresh cached rows should be returned
    # immediately instead of queuing expensive FinBERT/yfinance work for every card.
    if not force_refresh:
        cached = get_latest_prediction(ticker)
        if cached is not None:
            return cached

    # Backwards compatible default: existing frontend can keep calling /predict
    # and receive a prediction object synchronously. New UI can pass queued=true
    # to keep the API responsive while a background job generates the prediction.
    if queued:
        user_id = user.get("sub") if isinstance(user, dict) else None
        job = get_active_prediction_job(ticker=ticker, user_id=user_id)

        if job is None:
            job = create_prediction_job(
                ticker=ticker,
                user_id=user_id,
                live=force_refresh,
            )
            background_tasks.add_task(process_prediction_job, job["id"])

        response.status_code = 202
        return serialize_prediction_job(job)

    prediction = predict_ticker_risk(ticker)

    # Cache live predictions so newly added user tickers become fast for
    # the next user/request. Do not fail the prediction response if cache
    # write fails for a transient Supabase issue.
    try:
        if prediction is not None:
            save_latest_prediction(prediction)
    except Exception as exc:
        print(f"[cache] Failed to save latest prediction for {ticker}: {exc}")

    return prediction


@app.post("/prediction-jobs")
def enqueue_prediction_job(
    background_tasks: BackgroundTasks,
    response: Response,
    ticker: str = Query(..., min_length=1),
    live: bool = Query(False),
    force_refresh: bool = Query(False),
    user=Depends(require_user),
):
    ticker = ticker.upper().strip()

    # Cache-first behavior must also apply to the async endpoint. The `live`
    # parameter is kept only for backwards compatibility with older UI builds.
    # Use `force_refresh=true` when you really want to bypass the cache.
    if not force_refresh:
        cached = get_latest_prediction(ticker)
        if cached is not None:
            return {
                "job_id": None,
                "ticker": ticker,
                "status": "completed",
                "progress": 100,
                "message": "Prediction loaded from fresh cache.",
                "prediction": cached,
                "error_message": None,
            }

    user_id = user.get("sub") if isinstance(user, dict) else None
    job = get_active_prediction_job(ticker=ticker, user_id=user_id)

    if job is None:
        job = create_prediction_job(
            ticker=ticker,
            user_id=user_id,
            live=force_refresh,
        )
        background_tasks.add_task(process_prediction_job, job["id"])

    response.status_code = 202
    return serialize_prediction_job(job)


@app.get("/prediction-jobs/{job_id}")
def prediction_job_status(
    job_id: str,
    user=Depends(require_user),
):
    job = get_prediction_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Prediction job not found")

    user_id = user.get("sub") if isinstance(user, dict) else None
    job_user_id = job.get("user_id")

    if job_user_id and user_id and str(job_user_id) != str(user_id):
        raise HTTPException(status_code=403, detail="Not allowed to view this prediction job")

    return serialize_prediction_job(job)


@app.get("/predictions")
def predictions(user=Depends(require_user)):
    return {
        "items": get_all_latest_predictions()
    }

@app.get("/symbols/search")
def symbols_search(
    q: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(10, ge=1, le=25),
    user=Depends(require_user),
):
    return {
        "items": search_symbols(q, limit=limit)
    }
