# Major_Project_RAG

This repository contains the **data ingestion + indexing** foundation for a Retrieval-Augmented Generation (RAG) workflow focused on **drug interaction evidence**.

## What it does (today)

- **Extract PubMed** article titles/abstracts from XML into a Parquet file.
- **Chunk, embed & index** PubMed abstracts into a hosted **Pinecone** vector index.
- **Parse DrugBank vocabulary** into a normalization table (canonical name + synonyms).
- **Parse Drugs@FDA** tables into a Parquet file (not yet used by retrieval).
- **Validate query intent** (blocks dosage/diagnosis) and **normalize drug names** from user queries.

> Note: The repo currently does **not** implement the end-user “RAG answer generator” (retrieve + LLM response). It builds the assets you’d use for that.

## New structure (recommended)

All runnable logic is packaged under `src/major_project_rag/`:

- `src/major_project_rag/__main__.py`: enables `python -m major_project_rag ...`
- `src/major_project_rag/cli.py`: CLI entrypoint + subcommands
- `src/major_project_rag/config.py`: shared filesystem paths (raw/processed)
- `src/major_project_rag/ingestion/`: ingestion + indexing steps
  - `pubmed_extract.py`: PubMed XML -> Parquet
  - `pinecone_index.py`: chunk + embed + upsert Parquet into Pinecone
  - `drugbank_vocab.py`: DrugBank vocabulary CSV -> Parquet
  - `fda_parse.py`: Drugs@FDA tab files -> Parquet
- `src/major_project_rag/rag/`: query-side helpers (no LLM generation yet)
  - `intent_filter.py`: blocks unsafe intents (dosage/diagnosis)
  - `drug_normalization.py`: maps synonyms -> canonical drug names using DrugBank vocab

## Install

```bash
pip install -r requirements.txt
```

## Run (CLI)

From the repo root:

```bash
python -m major_project_rag --help

# 1) Extract PubMed XML -> Parquet
python -m major_project_rag pubmed-extract

# 2) Chunk + embed + upsert PubMed abstracts into Pinecone
# Required: set PINECONE_API_KEY in your environment
python -m major_project_rag pinecone-index --host "YOUR_PINECONE_HOST"

# (Optional) Tune chunking (characters)
python -m major_project_rag pinecone-index --host "YOUR_PINECONE_HOST" --chunk-size 800 --chunk-overlap 100

# 3) Parse DrugBank vocabulary CSV -> Parquet
python -m major_project_rag drugbank-vocab

# 4) Parse Drugs@FDA tables -> Parquet
python -m major_project_rag fda-parse

# Checks
python -m major_project_rag env-check
python -m major_project_rag print-config
```

## File-by-file guide (what each file does)

### Core entrypoints

- **`src/major_project_rag/__main__.py`**
  - Lets you run the package as a module: `python -m major_project_rag ...`

- **`src/major_project_rag/cli.py`**
  - Defines the CLI subcommands (extract, index, parse vocab, parse FDA, env-check).
  - This is the recommended way to run all processes.

### Shared config

- **`src/major_project_rag/config.py`**
  - Centralizes filesystem paths used everywhere:
    - raw data under `data/raw/...`
    - processed outputs under `data/processed/...`

### Ingestion + indexing (the “build” side)

- **`src/major_project_rag/ingestion/pubmed_extract.py`**
  - Reads PubMed dumps from `data/raw/pubmed/*.xml`
  - Stream-parses each XML file (memory efficient)
  - Writes `data/processed/pubmed_extracted.parquet` with columns:
    - `pmid`, `title`, `abstract`, `source_file`

- **`src/major_project_rag/ingestion/pinecone_index.py`**
  - Loads `data/processed/pubmed_extracted.parquet`
  - Filters short abstracts
  - **Chunks** abstracts (default ~800 chars, 100 overlap)
  - **Embeds** chunks using Pinecone hosted embeddings (default `llama-text-embed-v2`, 1024 dims)
  - **Upserts** into your Pinecone index via its host URL (namespace `default`)
  - This is where your “embedding and upserting” happens.

- **`src/major_project_rag/ingestion/chunking.py`**
  - Implements the chunking phase used by `chroma_index.py`
  - Splits text into overlapping character windows, trying to end on whitespace

- **`src/major_project_rag/ingestion/drugbank_vocab.py`**
  - Finds a DrugBank vocab CSV in `data/raw/` (pattern `*vocab*.csv`)
  - Outputs `data/processed/drugbank_vocab.parquet` with:
    - `drug_name`, `synonyms`, `drugbank_id` (when present)

- **`src/major_project_rag/ingestion/fda_parse.py`**
  - Reads Drugs@FDA tables from `data/raw/fda/` (e.g., `Products.txt`, `Applications.txt`)
  - Normalizes types and outputs `data/processed/fda_drugs.parquet`
  - (Not yet used in retrieval, but prepared for future use.)

### Query-side helpers (the “RAG input” side)

- **`src/major_project_rag/rag/intent_filter.py`**
  - A simple regex-based safety gate:
    - blocks dosage/diagnosis-related queries

- **`src/major_project_rag/rag/drug_normalization.py`**
  - Loads `data/processed/drugbank_vocab.parquet`
  - Builds synonym → canonical mappings
  - Extracts drug terms from user queries and returns canonical names

## Data locations

By default, the CLI expects:

- PubMed XML: `data/raw/pubmed/*.xml`
- DrugBank vocab CSV: `data/raw/*vocab*.csv`
- Drugs@FDA: `data/raw/fda/Products.txt`, `data/raw/fda/Applications.txt`

Outputs:

- PubMed parquet: `data/processed/pubmed_extracted.parquet`
- DrugBank vocab parquet: `data/processed/drugbank_vocab.parquet`
- FDA parquet: `data/processed/fda_drugs.parquet`
- Pinecone: remote hosted index (default name `pubmed-evidence`)

## Pinecone setup

- Create a Pinecone account and project.
- Set your API key (recommended: use `config.env`):
  - Copy/edit `config.env` (this repo auto-loads it via `python-dotenv`)
  - Fill `PINECONE_API_KEY=...`
  - Keep it secret (it is git-ignored)

You can customize where vectors land:
- `--namespace` (default `default`)

For on-demand indexes created in the Pinecone console, you should use the **index host**:
- Set `PINECONE_HOST=...` in `config.env`, or pass `--host ...` to `pinecone-index`


