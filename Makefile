.PHONY: up down logs test lint frontend-build

up:
	docker compose up --build

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

test:
	cd backend && pytest -q

lint:
	cd backend && ruff check app tests
	cd frontend && npm run type-check

frontend-build:
	cd frontend && npm run build
