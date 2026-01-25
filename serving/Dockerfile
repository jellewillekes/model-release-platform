FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "mlflow>=2.13,<3.0" boto3 fastapi uvicorn[standard] pandas numpy pydantic requests

COPY app.py /app/app.py
COPY smoke_test.py /app/smoke_test.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
