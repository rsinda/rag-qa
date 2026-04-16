import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

# Sentence Transformer Embeddings
EMBEDDING_DIM: int = 384
EMBEDDING_MODEL_NAME: str = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Qdrant
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rag_qa")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

RETRIEVAL_TOP_K_PRIMARY: int = 4
RETRIEVAL_TOP_K_FALLBACK: int = 3
SELF_RAG_MAX_RETRIES: int = 2
