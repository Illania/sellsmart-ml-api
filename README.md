# SellSmart ML API

AI-powered risk intelligence API for retail investors.

SellSmart ML API provides stock risk predictions, cached risk signals, and portfolio-ready analytics for the SellSmart platform.

The API combines machine learning, market data, technical indicators, and news sentiment analysis to help investors identify elevated downside risk before major drawdowns occur.

---

## Overview

Most investing tools focus on identifying buying opportunities.

SellSmart focuses on helping investors answer a different question:

> **"When should I reduce risk, wait, or exit a position?"**

The platform analyzes:

- Historical price behavior
- Market volatility
- Technical indicators
- Market context (SPY, QQQ, VIX)
- Financial news sentiment
- Panic-selling signals

and generates an easy-to-understand risk assessment.

---

## Features

- AI-powered stock risk scoring
- Cached prediction retrieval
- Optional live prediction mode
- News sentiment integration
- Portfolio-ready API responses
- FastAPI backend
- Supabase prediction storage
- Render deployment support

---

## Tech Stack

### Backend

- Python
- FastAPI
- Uvicorn

### Machine Learning

- XGBoost
- Scikit-learn
- Pandas
- NumPy

### Data Sources

- Yahoo Finance
- Finnhub News API
- FinBERT Sentiment Analysis

### Storage

- Supabase
- PostgreSQL

### Deployment

- Render

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

### Predict Risk

Returns the latest prediction for a ticker.

```http
GET /predict?ticker=NVDA
```

Example response:

```json
{
  "ticker": "NVDA",
  "date": "2026-06-16",
  "risk_score": 78,
  "category": "high",
  "action": "wait",
  "confidence": "medium",
  "summary": "NVDA shows elevated short-term downside risk.",
  "drivers": []
}
```

---

### Live Prediction

Runs a fresh prediction instead of returning cached results.

```http
GET /predict?ticker=NVDA&live=true
```

---

### Latest Predictions

Returns the latest available prediction for all supported tickers.

```http
GET /predictions
```

Example response:

```json
{
  "items": [
    {
      "ticker": "AAPL",
      "risk_score": 22,
      "category": "low",
      "action": "hold"
    }
  ]
}
```

---

## Supported Tickers

Current model coverage includes:

```text
AAPL
MSFT
NVDA
AMZN
GOOGL
META
TSLA
AMD
NFLX
JPM
CRM
ADBE
INTC
QCOM
PYPL
INSM
```

Additional tickers can be added through model retraining.

---

## Machine Learning Pipeline

### Prediction Horizon

The model predicts the probability of a significant downside move over the next:

```text
5 trading days
```

### Target Definition

A positive event is defined as:

```text
Future 5-day return ≤ -5%
```

### Feature Groups

#### Price Features

- Daily returns
- Rolling volatility
- Drawdown metrics
- Moving average distance
- Volume ratios

#### Technical Indicators

- RSI(14)
- Moving averages
- Trend strength

#### Market Context

- SPY
- QQQ
- VIX
- Market volatility

#### News Features

- Positive news ratio
- Negative news ratio
- Sentiment spikes
- Panic signals
- News volume anomalies

---

## Local Development

### Clone Repository

```bash
git clone https://github.com/Illania/sellsmart-ml-api.git
cd sellsmart-ml-api
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run API

```bash
uvicorn sellsmart_ml.api.app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Environment Variables

Create a `.env` file:

```env
SUPABASE_URL=
SUPABASE_KEY=
FINNHUB_API_KEY=
```

Additional configuration may be required depending on deployment environment.

---

## Deployment

The API is designed to run on Render.

Example start command:

```bash
uvicorn sellsmart_ml.api.app:app --host 0.0.0.0 --port $PORT
```

---

## Project Structure

```text
src/
└── sellsmart_ml/
    ├── api/
    ├── dataset/
    ├── features/
    ├── inference/
    ├── jobs/
    ├── models/
    ├── reports/
    ├── storage/
    ├── training/
    └── utils/
```

---

## SellSmart Platform

This repository contains the machine learning API powering the SellSmart ecosystem.

### SellSmart UI

Frontend dashboard built with:

- React
- TypeScript
- Vite

Features:

- Portfolio monitoring
- Risk alerts
- Watchlists
- Insights
- Reports

### SellSmart Vision

SellSmart aims to become an AI-powered risk intelligence platform that helps retail investors make better risk-management decisions through explainable machine learning.

---

## Disclaimer

SellSmart provides model-generated risk assessments and analytics.

The platform does **not** provide financial advice, investment recommendations, or guarantees of future performance.

All investment decisions remain the responsibility of the user.

---

## License

Copyright © SellSmart

All rights reserved.
