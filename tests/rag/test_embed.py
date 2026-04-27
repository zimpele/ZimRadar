import numpy as np
from unittest.mock import patch, MagicMock
from src.rag.embed import TextEmbedder


def test_text_embedder_returns_384_dim_vector():
    with patch("src.rag.embed.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)

        embedder = TextEmbedder()
        result = embedder.embed("Test sentence about flood risk in Texas.")

    assert isinstance(result, list)
    assert len(result) == 384


def test_text_embedder_batch_returns_correct_shape():
    with patch("src.rag.embed.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)

        embedder = TextEmbedder()
        results = embedder.embed_batch(["text1", "text2", "text3"])

    assert len(results) == 3
    assert all(len(v) == 384 for v in results)
