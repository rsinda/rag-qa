"""Ingestion pipeline for the RAG-QA system."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from app.vector_store import ContentVectorStore

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file_path", type=str, required=True)

args = parser.parse_args()

logger.info(f"Loading corpus from {args.corpus_file_path}")
with open(args.corpus_file_path, "r") as f:
    docs = [json.loads(line) for line in f if line.strip()]

logger.info(f"Upserting {len(docs)} documents into the vector store")
vector_store = ContentVectorStore()
vector_store.upsert_documents(docs)
logger.info(f"Upserted {len(docs)} documents into the vector store")
