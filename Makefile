.PHONY: up down eval pull-model migrate test lint deploy register-flows

up:
	docker compose up -d
	@echo "Services started. Streamlit: http://localhost:8501  Prefect: http://localhost:4200"

down:
	docker compose down

pull-model:
	docker compose exec ollama ollama pull gemma2:9b

migrate:
	docker compose exec postgres psql -U zimradar -d zimradar -f /docker-entrypoint-initdb.d/001_initial.sql

eval:
	pytest tests/evals/ -v -m slow

test:
	pytest tests/ -v --ignore=tests/evals/ -x

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

deploy:
	ssh root@187.77.95.56 "cd /opt/zimradar && git pull && docker compose up -d --build api streamlit worker && docker compose restart api streamlit"

register-flows:
	ssh root@187.77.95.56 "cd /opt/zimradar && docker compose exec -T worker prefect --no-prompt deploy --all"
