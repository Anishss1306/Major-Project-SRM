from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from major_project_rag.config import PATHS
from major_project_rag.ingestion.chunking import chunk_text


def _env_default(name: str, fallback: str) -> str:
    val = os.getenv(name)
    return val if val else fallback


@dataclass(frozen=True)
class PineconeIndexConfig:
    parquet_path: Path = PATHS.pubmed_parquet

    # Pinecone connection/config
    api_key_env: str = "PINECONE_API_KEY"
    index_name: str = _env_default("PINECONE_INDEX_NAME", "pubmed-evidence")
    namespace: str = _env_default("PINECONE_NAMESPACE", "default")
    metric: str = _env_default("PINECONE_METRIC", "cosine")  # cosine | dotproduct | euclidean
    cloud: str = _env_default("PINECONE_CLOUD", "aws")
    region: str = _env_default("PINECONE_REGION", "us-east-1")
    host: str = _env_default("PINECONE_HOST", "")

    # Pinecone hosted embeddings
    embed_model: str = _env_default("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
    embed_input_type: str = _env_default("PINECONE_EMBED_INPUT_TYPE", "passage")  # passage|query

    # Embeddings / ingestion
    batch_size: int = 100
    min_abstract_len: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 100


def _get_api_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise EnvironmentError(
            f"Missing Pinecone API key. Set environment variable {env_name} before running indexing."
        )
    return key


def _ensure_index(pc, *, name: str, dimension: int, metric: str, cloud: str, region: str) -> None:
    # pinecone SDK returns different shapes across versions; use the stable "names()" helper when present.
    try:
        existing = pc.list_indexes().names()
    except Exception:
        existing = [i["name"] for i in pc.list_indexes()]

    if name in existing:
        # Optional: sanity-check dimension/metric if available
        try:
            desc = pc.describe_index(name)
            idx_dim = getattr(desc, "dimension", None) or desc.get("dimension")
            if idx_dim and int(idx_dim) != int(dimension):
                raise ValueError(f"Existing Pinecone index '{name}' has dimension={idx_dim}, expected {dimension}")
        except Exception:
            # If describe fails, still proceed; upsert will fail if incompatible.
            pass
        return

    from pinecone import ServerlessSpec

    print(f"Creating Pinecone index '{name}' (dim={dimension}, metric={metric}, {cloud}/{region})...")
    pc.create_index(
        name=name,
        dimension=int(dimension),
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    # Wait for readiness
    for _ in range(60):
        try:
            desc = pc.describe_index(name)
            ready = getattr(desc, "status", None)
            if isinstance(ready, dict):
                if ready.get("ready"):
                    return
            else:
                # some SDKs: desc.status['ready']
                if getattr(desc, "status", {}).get("ready"):
                    return
        except Exception:
            pass
        time.sleep(2)

    print("Warning: index creation not confirmed as ready yet; continuing anyway.")


def build_or_update_index(
    *,
    cfg: PineconeIndexConfig = PineconeIndexConfig(),
    recreate: bool = False,
    limit: Optional[int] = None,
) -> None:
    from pinecone import Pinecone

    api_key = _get_api_key(cfg.api_key_env)

    parquet_path = Path(cfg.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    df = df[df["abstract"].astype(str).str.len() > cfg.min_abstract_len]
    if limit is not None:
        df = df.head(int(limit))
    print(f"Total articles to index: {len(df)}")

    pc = Pinecone(api_key=api_key)

    # If host is provided (recommended for on-demand indices), connect directly and skip index creation.
    if cfg.host:
        if recreate:
            raise ValueError("recreate is not supported when PINECONE_HOST is set (hosted index already exists).")
        index = pc.Index(host=cfg.host)
    else:
        # Legacy path: create serverless index by name (requires dimension; not supported with hosted embeddings here).
        raise ValueError(
            "PINECONE_HOST is required in this project setup. "
            "Set PINECONE_HOST to your index host URL (e.g., https://...pinecone.io)."
        )

    records = df.to_dict("records")
    print("Starting chunking -> Pinecone embed -> upsert...")

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_meta: list[dict] = []

    def flush() -> None:
        if not batch_docs:
            return
        # Pinecone hosted embeddings
        embeds = pc.inference.embed(
            model=cfg.embed_model,
            inputs=batch_docs,
            parameters={"input_type": cfg.embed_input_type},
        )
        data = getattr(embeds, "data", None) or embeds.get("data")
        vectors = []
        for item in data:
            values = getattr(item, "values", None) or item.get("values")
            vectors.append(values)

        to_upsert = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in zip(batch_ids, vectors, batch_meta, strict=False)
        ]
        index.upsert(vectors=to_upsert, namespace=cfg.namespace)
        batch_ids.clear()
        batch_docs.clear()
        batch_meta.clear()

    for record_idx, r in enumerate(tqdm(records, desc="Chunking + indexing")):
        pmid = r.get("pmid")
        base_id = str(pmid) if pmid else f"no_id_{record_idx}"

        abstract = str(r.get("abstract", ""))
        if len(abstract) <= cfg.min_abstract_len:
            continue

        chunks = chunk_text(abstract, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        for chunk_idx, ch in enumerate(chunks):
            batch_ids.append(f"{base_id}_c{chunk_idx}")
            batch_docs.append(ch.text)
            batch_meta.append(
                {
                    "pmid": pmid,
                    "title": r.get("title"),
                    "source": r.get("source_file"),
                    "chunk_index": chunk_idx,
                    "chunk_start": ch.start,
                    "chunk_end": ch.end,
                }
            )

            if len(batch_docs) >= cfg.batch_size:
                flush()

    flush()

    print("✅ Pinecone indexing complete.")
    print(f"Host: {cfg.host}")
    print(f"Namespace: {cfg.namespace}")


def main() -> int:
    build_or_update_index(recreate=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


