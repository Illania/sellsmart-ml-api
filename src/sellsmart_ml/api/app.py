from fastapi import FastAPI, Query

from sellsmart_ml.storage.supabase_predictions import get_latest_prediction
from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk


app = FastAPI(title="SellSmart Risk API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    ticker: str = Query(..., min_length=1),
    live: bool = Query(False),
):
    ticker = ticker.upper()

    if not live:
        cached = get_latest_prediction(ticker)
        if cached is not None:
            return cached

    result = predict_ticker_risk(ticker)
    return result