import pytest
from unittest.mock import patch, MagicMock

from app.vector_store import ContentVectorStore
from app.config import QDRANT_COLLECTION


@pytest.fixture
def mock_qdrant_client():
    with patch("app.vector_store.QdrantClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock get_collections
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_instance.get_collections.return_value = mock_collections

        yield mock_instance


@pytest.fixture
def mock_embed():
    "fixture to mock embed_texts"
    with patch("app.vector_store.embed_texts") as mock_embed_fn:
        mock_embed_fn.return_value = [[0.1, 0.2]]
        yield mock_embed_fn


def test_init_creates_collection(mock_qdrant_client):
    store = ContentVectorStore()
    mock_qdrant_client.create_collection.assert_called_once()
    assert mock_qdrant_client.create_payload_index.call_count == 2


def test_init_existing_collection(mock_qdrant_client):
    # check it does not try to create a collection if exists.
    mock_collection = MagicMock()
    mock_collection.name = QDRANT_COLLECTION
    mock_collections = MagicMock()
    mock_collections.collections = [mock_collection]
    mock_qdrant_client.get_collections.return_value = mock_collections

    # Initialize the store
    store = ContentVectorStore()

    # Should NOT attempt to create the collection again
    mock_qdrant_client.create_collection.assert_not_called()


def test_upsert_documents(mock_qdrant_client, mock_embed):
    store = ContentVectorStore()

    docs = [
        {
            "content_id": "test-id-1",
            "country": "US",
            "language": "en",
            "type": "article",
            "title": "Test Title",
            "body": "Test Body",
            "updated_at": "2023-01-01",
        }
    ]

    # Mock embedding
    store._embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    result = store.upsert_documents(docs)

    assert result == 1
    mock_qdrant_client.upsert.assert_called_once()

    # Check upsert
    call_args = mock_qdrant_client.upsert.call_args[1]
    points = call_args["points"]
    assert len(points) == 1
    assert points[0].payload["content_id"] == "test-id-1"
    assert points[0].payload["title"] == "Test Title"
    assert points[0].payload["country"] == "US"


def test_query(mock_qdrant_client):
    store = ContentVectorStore()
    store._embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    # Mock query_points for primary hit
    primary_hit = MagicMock()
    primary_hit.score = 0.9
    primary_hit.payload = {"content_id": "doc1", "title": "Doc 1", "body": "Body 1"}

    # Mock query_points for fallback hit
    fallback_hit = MagicMock()
    fallback_hit.score = 0.8
    fallback_hit.payload = {"content_id": "doc2", "title": "Doc 2", "body": "Body 2"}

    # test deduplication logic
    fallback_hit_duplicate = MagicMock()
    fallback_hit_duplicate.score = 0.7
    fallback_hit_duplicate.payload = {
        "content_id": "doc1",
        "title": "Doc 1",
        "body": "Body 1",
    }

    primary_response = MagicMock()
    primary_response.points = [primary_hit]

    fallback_response = MagicMock()
    fallback_response.points = [fallback_hit, fallback_hit_duplicate]

    mock_qdrant_client.query_points.side_effect = [primary_response, fallback_response]

    results = store.query("test query", country="US", language="en")

    # Should be 2 because doc1 shouldn't be added twice
    assert len(results) == 2

    # Validate primary result
    assert results[0]["content_id"] == "doc1"
    assert results[0]["match_score"] == 0.9  # 0.9 * 1.0 weight
    assert results[0]["is_fallback"] is False

    # Validate fallback result
    assert results[1]["content_id"] == "doc2"
    assert results[1]["match_score"] == 0.56  # 0.8 * 0.7 weight
    assert results[1]["is_fallback"] is True


def test_count(mock_qdrant_client):
    store = ContentVectorStore()

    mock_info = MagicMock()
    mock_info.points_count = 42
    mock_qdrant_client.get_collection.return_value = mock_info

    count = store.count()
    assert count == 42
    mock_qdrant_client.get_collection.assert_called_once_with(QDRANT_COLLECTION)
