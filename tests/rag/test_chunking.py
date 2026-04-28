from src.rag.chunking import chunk_text


def test_short_text_returns_single_chunk():
    text = "Short text under the chunk size limit."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_produces_overlapping_chunks():
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.split()) > 0


def test_chunk_overlap_is_present():
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text)
    if len(chunks) >= 2:
        end_of_first = chunks[0].split()[-10:]
        start_of_second = chunks[1].split()[:20]
        overlap = set(end_of_first) & set(start_of_second)
        assert len(overlap) > 0
