from transformers import AutoTokenizer

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


def chunk_text(text: str) -> list[str]:
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_str)
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
