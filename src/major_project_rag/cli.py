from __future__ import annotations

import argparse
from dataclasses import asdict

from major_project_rag.ingestion.pinecone_index import PineconeIndexConfig, build_or_update_index
from major_project_rag.ingestion.drugbank_vocab import parse_drugbank_vocab
from major_project_rag.ingestion.fda_parse import parse_fda_drugs
from major_project_rag.ingestion.pubmed_extract import extract_pubmed_xml_to_parquet


def _load_env() -> None:
    """
    Load environment variables from config files if python-dotenv is installed.

    Priority:
    - config.env (repo-specific)
    - .env (standard)
    - existing process environment always wins by default behavior
    """
    try:
        from dotenv import load_dotenv

        load_dotenv("config.env")
        load_dotenv(".env")
    except Exception:
        # python-dotenv is optional at runtime; users can also set vars in the shell.
        pass


def _cmd_env_check(_args: argparse.Namespace) -> int:
    import torch
    import pinecone

    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    print(f"Pinecone SDK Version: {getattr(pinecone, '__version__', 'unknown')}")
    print("PINECONE_API_KEY set?", bool(__import__('os').getenv("PINECONE_API_KEY")))
    print("✅ Environment looks ready (SDK import + torch).")
    return 0


def _cmd_pubmed_extract(_args: argparse.Namespace) -> int:
    extract_pubmed_xml_to_parquet()
    return 0


def _cmd_pinecone_index(args: argparse.Namespace) -> int:
    cfg = PineconeIndexConfig(
        batch_size=args.batch_size,
        index_name=args.index_name,
        namespace=args.namespace,
        metric=args.metric,
        cloud=args.cloud,
        region=args.region,
        host=args.host,
        embed_model=args.embed_model,
        embed_input_type=args.embed_input_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    build_or_update_index(cfg=cfg, recreate=args.recreate, limit=args.limit)
    return 0


def _cmd_drugbank_vocab(_args: argparse.Namespace) -> int:
    parse_drugbank_vocab()
    return 0


def _cmd_fda_parse(_args: argparse.Namespace) -> int:
    parse_fda_drugs()
    return 0


def _cmd_print_config(args: argparse.Namespace) -> int:
    cfg = PineconeIndexConfig()
    print(asdict(cfg))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="major_project_rag", description="Major_Project_RAG utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    env = sub.add_parser("env-check", help="Print environment diagnostics")
    env.set_defaults(func=_cmd_env_check)

    pe = sub.add_parser("pubmed-extract", help="Extract PubMed XML -> data/processed/pubmed_extracted.parquet")
    pe.set_defaults(func=_cmd_pubmed_extract)

    pi = sub.add_parser("pinecone-index", help="Chunk + embed + upsert PubMed abstracts into Pinecone")
    pi.add_argument("--recreate", action="store_true", help="Delete and recreate the Pinecone index before indexing")
    pi.add_argument("--limit", type=int, default=None, help="Limit number of records (for testing)")
    pi.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding/upserts")
    pi.add_argument("--chunk-size", type=int, default=800, help="Chunk size (characters) before embedding")
    pi.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap (characters)")
    pi.add_argument("--host", default="", help="Pinecone index host URL (recommended for on-demand indices)")
    pi.add_argument("--embed-model", default="llama-text-embed-v2", help="Pinecone embedding model name")
    pi.add_argument(
        "--embed-input-type",
        default="passage",
        help="Embedding input type for Pinecone inference (passage|query)",
    )
    pi.add_argument("--index-name", default="pubmed-evidence", help="Pinecone index name")
    pi.add_argument("--namespace", default="default", help="Pinecone namespace")
    pi.add_argument("--metric", default="cosine", help="Pinecone metric: cosine|dotproduct|euclidean")
    pi.add_argument("--cloud", default="aws", help="Pinecone serverless cloud (e.g., aws)")
    pi.add_argument("--region", default="us-east-1", help="Pinecone serverless region (e.g., us-east-1)")
    pi.set_defaults(func=_cmd_pinecone_index)

    dv = sub.add_parser("drugbank-vocab", help="Parse DrugBank vocabulary CSV -> data/processed/drugbank_vocab.parquet")
    dv.set_defaults(func=_cmd_drugbank_vocab)

    fp = sub.add_parser("fda-parse", help="Parse Drugs@FDA tables -> data/processed/fda_drugs.parquet")
    fp.set_defaults(func=_cmd_fda_parse)

    pc = sub.add_parser("print-config", help="Print default indexing config")
    pc.set_defaults(func=_cmd_print_config)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


