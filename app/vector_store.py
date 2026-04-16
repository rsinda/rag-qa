import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    KeywordIndexParams,
    KeywordIndexType,
)

from app.utils import embed_texts, content_id_to_uuid
from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_API_KEY,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


class ContentVectorStore:
    """Qdrant vector store with Country + language filtering."""

    def __init__(self) -> None:
        kwargs: Dict[str, Any] = {"host": QDRANT_HOST, "port": QDRANT_PORT}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        self._client = QdrantClient(**kwargs)
        self._embed = embed_texts
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the collection with its Schema if it doesn't exist yet.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if QDRANT_COLLECTION not in existing:
            logger.info(
                "Creating Qdrant collection '%s' (dim=%d)",
                QDRANT_COLLECTION,
                EMBEDDING_DIM,
            )
            self._client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            self._create_payload_indexes()
        else:
            logger.info("Collection '%s' already exists.", QDRANT_COLLECTION)

    def _create_payload_indexes(self) -> None:
        """
        Index 'country' as a tenant field and 'language' as a keyword field.
        """
        logger.info("Creating payload indexes on 'country' (tenant) and 'language'")
        self._client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="country",
            field_schema=KeywordIndexParams(
                type=KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
        self._client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="language",
            field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
        )

    def upsert_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        Embed and upsert a batch of content items into Qdrant.
        Uses content_id → deterministic UUID so re-ingestion is safe.
        """
        texts = [f"Title: {d['title']}\n\nBody: {d['body']}" for d in docs]
        vectors = self._embed(texts)

        points = [
            PointStruct(
                id=content_id_to_uuid(doc["content_id"]),
                vector=vector,
                payload={
                    "content_id": doc["content_id"],
                    "country": doc["country"],
                    "language": doc["language"],
                    "type": doc["type"],
                    "version": str(doc.get("version", "")),
                    "title": doc["title"],
                    "body": doc["body"],
                    "updated_at": doc["updated_at"],
                },
            )
            for doc, vector in zip(docs, vectors)
        ]

        self._client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info(
            "Upserted %d points into Qdrant collection '%s'",
            len(points),
            QDRANT_COLLECTION,
        )
        return len(points)

    def query(
        self,
        query_text: str,
        country: str,
        language: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k documents scoped to (country, language).
        """
        query_vector = self._embed([query_text])[0]

        tenant_filter = Filter(
            must=[
                FieldCondition(key="country", match=MatchValue(value=country)),
                FieldCondition(key="language", match=MatchValue(value=language)),
            ]
        )

        try:
            results = self._client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                query_filter=tenant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning(
                "Qdrant query failed (country=%s, language=%s): %s",
                country,
                language,
                exc,
            )
            return []

        hits = []
        for scored_point in results.points:
            payload = scored_point.payload or {}
            hits.append(
                {
                    **payload,
                    "match_score": round(scored_point.score, 4),
                }
            )

        logger.debug(
            "query: %d hits for country=%s language=%s", len(hits), country, language
        )
        return hits

    def count(self) -> int:
        info = self._client.get_collection(QDRANT_COLLECTION)
        return info.points_count or 0
