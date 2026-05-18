# Multi-stage build to keep production image lean
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy installed python packages from builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Ensure local pip binaries are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
