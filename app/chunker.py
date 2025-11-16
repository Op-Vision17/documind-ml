# ml-service/app/chunker.py
from typing import List

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by words.
    chunk_size = number of words per chunk.
    chunk_overlap = number of overlapping words between consecutive chunks.
    """
    if not text:
        return []

    tokens = text.split()
    if len(tokens) == 0:
        return []

    chunks = []
    i = 0
    while i < len(tokens):
        end = i + chunk_size
        chunk = tokens[i:end]
        chunks.append(" ".join(chunk))
        i += chunk_size - chunk_overlap

    return chunks
