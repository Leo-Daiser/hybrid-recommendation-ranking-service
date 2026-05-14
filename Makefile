.PHONY: build up down test init-db download-data validate-raw prepare-interactions build-features build-candidates evaluate-retrieval build-ranking-dataset train-ranker run-full-pipeline pipeline-status api db-up db-check

build:
	docker-compose build

up:
	docker-compose up -d

db-up:
	docker-compose up -d db

down:
	docker-compose down

test:
	pytest -q

init-db:
	python -m src.cli init-db

db-check:
	python -m src.cli db-check

download-data:
	python -m src.cli download-data

validate-raw:
	python -m src.cli validate-raw

prepare-interactions:
	python -m src.cli prepare-interactions

build-features:
	python -m src.cli build-features

build-candidates:
	python -m src.cli build-popularity-candidates
	python -m src.cli build-itemknn-candidates

evaluate-retrieval:
	python -m src.cli evaluate-retrieval

build-ranking-dataset:
	python -m src.cli build-ranking-dataset

train-ranker:
	python -m src.cli train-ranker

run-full-pipeline:
	python -m src.cli run-full-pipeline

pipeline-status:
	python -m src.cli pipeline-status

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
