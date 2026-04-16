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
    RETRIEVAL_TOP_K_PRIMARY,
    RETRIEVAL_TOP_K_FALLBACK,
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
        Two-pass weighted retrieval.
            Primary   (weight=1.0): exact country + exact language
            Fallback  (weight=0.7): exact country + any other language

        Scores are normalised and merged. The country filter NEVER relaxes.

        """
        query_vector = self._embed([query_text])[0]

        # 1st Pass
        primary_filter = Filter(
            must=[
                FieldCondition(key="country", match=MatchValue(value=country)),
                FieldCondition(key="language", match=MatchValue(value=language)),
            ]
        )
        primary_hits = self._client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=primary_filter,
            limit=RETRIEVAL_TOP_K_PRIMARY,
            with_payload=True,
        )

        # 2nd Pass
        fallback_filter = Filter(
            must=[
                FieldCondition(key="country", match=MatchValue(value=country)),
            ],
            must_not=[
                FieldCondition(key="language", match=MatchValue(value=language)),
            ],
        )
        fallback_hits = self._client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=fallback_filter,
            limit=RETRIEVAL_TOP_K_FALLBACK,
            with_payload=True,
        )

        # Merge Results.

        PRIMARY_WEIGHT = 1.0
        FALLBACK_WEIGHT = 0.7

        seen_ids = set()
        merged = []
        for hit in primary_hits.points:
            payload = hit.payload
            weighted_score = round(hit.score * PRIMARY_WEIGHT, 4)
            merged.append(
                {
                    **payload,
                    "excerpt": f"Title: {payload['title']}\n\nBody: {payload['body']}",
                    "match_score": weighted_score,
                    "raw_score": hit.score,
                    "is_fallback": False,
                }
            )
            seen_ids.add(payload["content_id"])

        for hit in fallback_hits.points:
            payload = hit.payload
            if payload["content_id"] in seen_ids:
                continue
            weighted_score = round(hit.score * FALLBACK_WEIGHT, 4)
            merged.append(
                {
                    **payload,
                    "excerpt": f"Title: {payload['title']}\n\nBody: {payload['body']}",
                    "match_score": weighted_score,
                    "raw_score": hit.score,
                    "is_fallback": True,
                }
            )

        merged.sort(key=lambda x: x["match_score"], reverse=True)

        logger.debug("query: %d hits for country=%s", len(merged), country)

        return merged

    def count(self) -> int:
        info = self._client.get_collection(QDRANT_COLLECTION)
        return info.points_count or 0
