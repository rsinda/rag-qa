import logging
import hashlib
import uuid
from typing import List
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_embedder = SentenceTransformer(
    EMBEDDING_MODEL_NAME, model_kwargs={"torch_dtype": "float16"}
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    return _embedder.encode(texts, convert_to_numpy=True).tolist()


def content_id_to_uuid(content_id: str) -> str:
    """Deterministic UUID from content_id for upserts."""
    return str(uuid.UUID(hashlib.md5(content_id.encode()).hexdigest()))
