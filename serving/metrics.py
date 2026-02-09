from __future__ import annotations

from prometheus_client import Counter, Histogram

# Labels are bounded, we do not label by request_id, model_name, etc.
REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total HTTP requests handled by the serving API.",
    labelnames=("endpoint", "mode", "status"),
)

PREDICT_LATENCY_SECONDS = Histogram(
    "predict_latency_seconds",
    "Latency of /predict requests in seconds.",
    labelnames=("mode", "status", "chosen"),
    # Optional buckets, Prometheus defaults work good for now.
)

SHADOW_DIFF_MAE = Histogram(
    "shadow_diff_mae",
    "Mean absolute difference between primary and shadow predictions (when shadow runs).",
    labelnames=("mode",),
)
