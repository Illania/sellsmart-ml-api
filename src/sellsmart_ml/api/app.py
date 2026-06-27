import os
import jwt

from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from jwt import PyJWKClient

from sellsmart_ml.storage.supabase_predictions import get_latest_prediction, save_latest_prediction
from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk
from sellsmart_ml.storage.supabase_predictions import get_all_latest_predictions
from typing import Optional


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
    ticker: str = Query(..., min_length=1),
    live: bool = Query(False),
    user=Depends(require_user),
):
    ticker = ticker.upper()

    if not live:
        cached = get_latest_prediction(ticker)
        if cached is not None:
            return cached

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


@app.get("/predictions")
def predictions(user=Depends(require_user)):
    return {
        "items": get_all_latest_predictions()
    }