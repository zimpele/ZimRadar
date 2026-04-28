FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgdal-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system -e .
COPY src/ ./src/

FROM base AS worker
RUN uv pip install --system -e ".[dev]"
COPY tests/ ./tests/
CMD ["python", "-m", "prefect", "worker", "start", "--pool", "default-agent-pool"]

FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
