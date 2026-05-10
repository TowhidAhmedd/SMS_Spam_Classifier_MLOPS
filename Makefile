.PHONY: install data train test lint run docker-up docker-down clean

install:
	pip install -r requirements-dev.txt

data:
	python src/download_data.py

train:
	python src/train.py

test:
	pytest tests/ -v --cov=src --cov=app --cov-report=term-missing

lint:
	ruff check src/ app.py tests/ --ignore E501

run:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# Docker (local full stack: API + PostgreSQL + Prometheus + Grafana)
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
