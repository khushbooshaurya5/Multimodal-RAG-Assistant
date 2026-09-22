.PHONY: install install-dev test lint format ui api ingest eval docker

PY ?= python

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

test-models:
	$(PY) -m pytest -q -m requires_models tests/integration/test_real_models.py

lint:
	ruff check src app tests scripts && ruff format --check src app tests scripts

format:
	ruff format src app tests scripts && ruff check --fix src app tests scripts

ui:
	streamlit run app/main.py

api:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --reload

ingest:
	$(PY) scripts/ingest.py data/raw

eval:
	$(PY) scripts/evaluate.py

docker:
	docker build -t multimodal-rag-assistant .
