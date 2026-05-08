from fastapi import FastAPI, Query

from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk

app = FastAPI(title="SellSmart Risk API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(ticker: str = Query(..., min_length=1)):
    result = predict_ticker_risk(ticker.upper())
    return result