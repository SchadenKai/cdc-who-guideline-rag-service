setup-dev:
	cd backend && uv sync --all-extras && playwright install

run-containers:
	cd deployment && docker compose up -d --build --force-recreate

run-vectors-db:
	cd deployment && docker compose -f docker-compose.milvus.yml up -d