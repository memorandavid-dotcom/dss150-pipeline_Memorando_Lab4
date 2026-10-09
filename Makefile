# ============================================================
# Makefile: DSS150P Lab 4 Automation Shortcuts
# ============================================================

.PHONY: help install check-source ingest transform validate storage train evaluate run-all test docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make check-source   Verify source dataset SHA-256 checksum"
	@echo "  make ingest         Run full and incremental ingestion"
	@echo "  make transform      Run raw -> staging -> curated transformations"
	@echo "  make validate       Run data quality checks"
	@echo "  make storage        Materialize multi-format and partitioned storage"
	@echo "  make train          Train all 5 classifiers"
	@echo "  make evaluate       Evaluate models and verify reproducibility"
	@echo "  make run-all        Run full pipeline end-to-end"
	@echo "  make test           Run automated pytest test suite"
	@echo "  make docker-up      Start PostgreSQL and Airflow via Docker Compose"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make clean          Clean generated artifacts, cache, and outputs"

install:
	pip install -r requirements.txt

check-source:
	python -m src.cli check-source

ingest:
	python -m src.cli ingest --mode full
	python -m src.cli ingest --mode incremental

transform:
	python -m src.cli transform

validate:
	python -m src.cli validate

storage:
	python -m src.cli storage --skip-postgres

train:
	python -m src.cli train

evaluate:
	python -m src.cli evaluate

run-all:
	python -m src.cli run-all --skip-postgres

test:
	pytest -v

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('**/__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('**/.pytest_cache')]"
