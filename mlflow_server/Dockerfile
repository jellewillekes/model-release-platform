FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# MLflow + postgres driver + s3 support
RUN pip install --no-cache-dir "mlflow>=2.13,<3.0" psycopg2-binary boto3

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 5000
ENTRYPOINT ["/start.sh"]
