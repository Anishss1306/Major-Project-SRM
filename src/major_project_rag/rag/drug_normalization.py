from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from major_project_rag.config import PATHS


@dataclass
class DrugNormalizer:
    vocab_path: Path = PATHS.drugbank_vocab_parquet
    synonym_map: Dict[str, str] = field(default_factory=dict)  # synonym(lower) -> canonical
    canonical_drugs: Set[str] = field(default_factory=set)  # canonical(lower)

    def load_vocab(self) -> None:
        if not Path(self.vocab_path).exists():
            raise FileNotFoundError(f"DrugBank vocab parquet not found: {self.vocab_path}")

        df = pd.read_parquet(self.vocab_path)

        self.synonym_map.clear()
        self.canonical_drugs.clear()

        for name in df.get("drug_name", []):
            if name:
                clean = str(name).lower().strip()
                self.synonym_map[clean] = str(name)
                self.canonical_drugs.add(clean)

        for _, row in df.iterrows():
            canonical = row.get("drug_name")
            if not canonical:
                continue
            syn_val = row.get("synonyms", "")
            if not syn_val:
                continue

            syns = str(syn_val).replace(" | ", "|").split("|")
            for s in syns:
                clean_s = s.lower().strip()
                if clean_s and clean_s not in self.synonym_map:
                    self.synonym_map[clean_s] = str(canonical)

    def extract_and_normalize(self, query: str) -> List[str]:
        if not self.synonym_map:
            self.load_vocab()

        words = re.findall(r"\b\w+\b", query.lower())
        found = []
        for w in words:
            if w in self.synonym_map:
                found.append(self.synonym_map[w])
        return sorted(set(found))


def main() -> int:
    n = DrugNormalizer()
    q = "Can I take Advil with Tylenol?"
    print(q)
    print(n.extract_and_normalize(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


