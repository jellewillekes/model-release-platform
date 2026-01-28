FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
  "mlflow>=2.13,<3.0" \
  boto3 fastapi "uvicorn[standard]" \
  pandas numpy pydantic requests \
  scikit-learn joblib

# Keep the package layout: /app/serving/...
RUN mkdir -p /app/serving

COPY __init__.py /app/serving/__init__.py
COPY app.py /app/serving/app.py
COPY router.py /app/serving/router.py
COPY constants.py /app/serving/constants.py
COPY smoke_test.py /app/serving/smoke_test.py

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
