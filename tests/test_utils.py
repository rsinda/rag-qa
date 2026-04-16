import pytest
import uuid
from unittest.mock import patch, MagicMock

from app.utils import embed_texts, content_id_to_uuid


def test_embed_texts():
    with patch("app.utils._embedder.encode") as mock_encode:
        mock_numpy_array = MagicMock()
        mock_numpy_array.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_encode.return_value = mock_numpy_array

        texts = ["test string 1", "test string 2"]
        result = embed_texts(texts)

        mock_encode.assert_called_once_with(texts, convert_to_numpy=True)
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_content_id_to_uuid():
    content_id = "test_content_123"
    result = content_id_to_uuid(content_id)

    # check the result is a string
    assert isinstance(result, str)

    # check the result is a valid UUID
    try:
        parsed_uuid = uuid.UUID(result)
        assert str(parsed_uuid) == result
    except ValueError:
        pytest.fail("Result is not a valid UUID string")

    # check deterministic output (same input results in same output)
    result2 = content_id_to_uuid(content_id)
    assert result == result2

    # Ensure different input results in different output
    result3 = content_id_to_uuid("different_content")
    assert result != result3
