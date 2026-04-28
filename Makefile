.PHONY: up down eval pull-model migrate test lint

DOCKER ?= /Users/luca/.orbstack/bin/docker

up:
	$(DOCKER) compose up -d
	@echo "Services started. Streamlit: http://localhost:8501  Prefect: http://localhost:4200"

down:
	$(DOCKER) compose down

pull-model:
	$(DOCKER) compose exec ollama ollama pull gemma2:9b

migrate:
	$(DOCKER) compose exec postgres psql -U zimradar -d zimradar -f /docker-entrypoint-initdb.d/001_initial.sql

eval:
	pytest tests/evals/ -v -m slow

test:
	pytest tests/ -v --ignore=tests/evals/ -x

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
