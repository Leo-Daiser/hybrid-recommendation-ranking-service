.PHONY: build up down test init-db

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

test:
	pytest -q

init-db:
	python -m src.cli init-db
