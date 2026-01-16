run-containers:
	cd deployment && docker compose up -d

run-vectors-db:
	cd deployment && docker compose -f docker-compose.milvus.yml up -d