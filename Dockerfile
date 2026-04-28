FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONPATH=/app
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system -e .
COPY src/ ./src/

FROM base AS worker
CMD ["python", "-m", "prefect", "worker", "start", "--pool", "default-agent-pool"]

FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
