from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int


def chunk_text(text: str, *, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Chunk]:
    """
    Simple character-based chunking with overlap.

    - Keeps chunks roughly `chunk_size` characters
    - Overlaps by `chunk_overlap` characters
    - Tries to end chunks on whitespace for cleaner boundaries
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    t = (text or "").strip()
    if not t:
        return []

    chunks: List[Chunk] = []
    n = len(t)
    start = 0

    while start < n:
        raw_end = min(start + chunk_size, n)

        # Prefer to cut at whitespace to avoid chopping words.
        end = raw_end
        if raw_end < n:
            window = t[start:raw_end]
            last_space = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
            if last_space > int(chunk_size * 0.6):
                end = start + last_space

        chunk_str = t[start:end].strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, start=start, end=end))

        if end >= n:
            break

        start = max(0, end - chunk_overlap)

        # Guard: ensure forward progress even in pathological cases.
        if chunks and start <= chunks[-1].start:
            start = chunks[-1].end

    return chunks


def chunk_many(texts: Iterable[str], *, chunk_size: int = 800, chunk_overlap: int = 100) -> List[List[Chunk]]:
    return [chunk_text(t, chunk_size=chunk_size, chunk_overlap=chunk_overlap) for t in texts]


