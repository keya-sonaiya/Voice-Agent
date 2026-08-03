.PHONY: backend frontend test lint

backend:
	uvicorn app.main:app --reload

frontend:
	cd frontend && npm run dev

test:
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check app/ tests/ && black --check app/ tests/ && mypy app/
