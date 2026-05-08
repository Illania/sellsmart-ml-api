#!/bin/zsh

cd /Users/annagulich/Desktop/sellsmart-ml || exit 1

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$FINNHUB_API_KEY" ]; then
  echo "ERROR: FINNHUB_API_KEY is not set"
  exit 1
fi

mkdir -p data/cache/live_news

TODAY=$(date +%F)
FROM=$(date -v-30d +%F)

for TICKER in AAPL MSFT NVDA AMZN GOOGL META TSLA AMD NFLX JPM CRM ADBE INTC QCOM PYPL INSM
do
  echo ""
  echo "Refreshing $TICKER..."

  OUT="data/cache/live_news/${TICKER}_raw_news.json"
  URL="https://finnhub.io/api/v1/company-news?symbol=${TICKER}&from=${FROM}&to=${TODAY}&token=${FINNHUB_API_KEY}"

  /usr/bin/curl -s "$URL" > "$OUT"

  CODE=$?

  echo "curl exit code: $CODE"
  echo "file size: $(wc -c < "$OUT") bytes"
  echo "Saved $OUT"

  sleep 1
done

echo "Done."