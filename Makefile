.PHONY: install test server pipeline docker-up docker-down clean

# Virtual environment python binary
PYTHON := .venv/bin/python
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest

install:
	pip install -r requirements.txt

test:
	$(PYTEST) tests/ -v

server:
	$(UVICORN) src.api.main:app --port 8000 --reload

pipeline:
	./run.sh pipeline

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
