# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by LightGBM/XGBoost at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; the ML stack is otherwise pinned.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY pyproject.toml ./
COPY ml ./ml
COPY backend ./backend

RUN useradd --create-home --uid 1001 aegis \
    && mkdir -p /app/data \
    && chown -R aegis:aegis /app
USER aegis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
