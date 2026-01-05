from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd
from lxml import etree
from tqdm import tqdm

from major_project_rag.config import PATHS


@dataclass(frozen=True)
class PubMedRecord:
    pmid: Optional[str]
    title: str
    abstract: str
    source_file: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "source_file": self.source_file,
        }


def iter_pubmed_records(xml_file: Path) -> Iterator[PubMedRecord]:
    """
    Stream-parse a single PubMed XML file and yield article records.

    This uses iterparse + element clearing for low memory usage.
    """
    context = etree.iterparse(str(xml_file), events=("end",), tag="PubmedArticle")

    for _event, elem in context:
        try:
            medline = elem.find("MedlineCitation")
            if medline is None:
                continue

            article = medline.find("Article")
            if article is None:
                continue

            title_elem = article.find("ArticleTitle")
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

            abstract_elem = article.find("Abstract")
            abstract_text = ""
            if abstract_elem is not None:
                texts = abstract_elem.findall("AbstractText")
                abstract_text = " ".join([t.text for t in texts if t is not None and t.text]).strip()

            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None

            if title and abstract_text:
                yield PubMedRecord(
                    pmid=pmid,
                    title=title,
                    abstract=abstract_text,
                    source_file=xml_file.name,
                )

        except Exception:
            # Skip malformed records but don't stop the whole parse.
            pass
        finally:
            # Free memory aggressively.
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    del context


def extract_pubmed_xml_to_parquet(
    *,
    raw_pubmed_dir: Path = PATHS.raw_pubmed,
    output_file: Path = PATHS.pubmed_parquet,
) -> Path:
    raw_pubmed_dir = Path(raw_pubmed_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    xml_files = [Path(p) for p in glob.glob(str(raw_pubmed_dir / "*.xml"))]
    print(f"🔎 Found {len(xml_files)} XML files in {raw_pubmed_dir}")
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under: {raw_pubmed_dir}")

    rows = []
    for xml_file in tqdm(xml_files, desc="Processing PubMed XML"):
        for record in iter_pubmed_records(xml_file):
            rows.append(record.to_dict())

    print(f"💾 Saving {len(rows)} extracted articles to {output_file}...")
    df = pd.DataFrame(rows)
    df.to_parquet(output_file, index=False)
    print("✅ PubMed extraction complete.")
    return output_file


def main() -> int:
    extract_pubmed_xml_to_parquet()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


