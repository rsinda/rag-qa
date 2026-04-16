#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Docker services (Qdrant)..."
docker compose -f docker/docker-compose.yml up -d

echo "📦 Installing Python dependencies..."
pip install -r requirement.txt

python ingestion_pipeline/ingest.py --corpus_file_path corpus.jsonl

echo "🌐 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000
