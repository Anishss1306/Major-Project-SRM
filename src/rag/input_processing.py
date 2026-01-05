import pandas as pd
import re
import os


class InputProcessor:
    def __init__(self):
        self.vocab_path = os.path.join("data", "processed", "drugbank_vocab.parquet")
        self.synonym_map = {}
        self.canonical_drugs = set()
        self._load_vocab()

    def _load_vocab(self):
        """
        Loads the DrugBank vocabulary and builds a Synonym -> Canonical Name map.
        """
        if not os.path.exists(self.vocab_path):
            print(f"Warning: Vocab file not found at {self.vocab_path}")
            return

        print("Loading Drug Vocabulary for Normalization...")
        df = pd.read_parquet(self.vocab_path)

        # 1. Add canonical names to the map
        for name in df['drug_name']:
            if name:
                clean_name = name.lower().strip()
                self.synonym_map[clean_name] = name  # Map lower -> Title Case
                self.canonical_drugs.add(clean_name)

        # 2. Process Synonyms
        for _, row in df.iterrows():
            canonical = row['drug_name']
            if not row['synonyms']:
                continue

            # Robust split: Handle " | " or "|"
            syns = str(row['synonyms']).replace(" | ", "|").split("|")

            for s in syns:
                clean_s = s.lower().strip()
                # Canonical takes precedence
                if clean_s and clean_s not in self.synonym_map:
                    self.synonym_map[clean_s] = canonical

        print(f"Vocabulary loaded. Mapped {len(self.synonym_map)} drug terms.")

        # DEBUG: Check if common drugs exist
        print(f"DEBUG CHECK: 'advil' in map? {'advil' in self.synonym_map}")
        print(f"DEBUG CHECK: 'tylenol' in map? {'tylenol' in self.synonym_map}")

    def validate_intent(self, query: str) -> dict:
        """
        Checks if the query contains unsafe keywords.
        """
        query_lower = query.lower()

        unsafe_patterns = [
            r"how much should i take",
            r"dose",
            r"dosage",
            r"diagnose",
            r"what do i have",
            r"symptom checker"
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, query_lower):
                return {
                    "valid": False,
                    "reason": f"Safety Violation: Query contains restricted concepts ({pattern}). This system is for interaction checking only, not dosage or diagnosis."
                }

        return {"valid": True, "reason": "Safe"}

    def extract_and_normalize_drugs(self, query: str) -> list:
        """
        Scans the query for known drug names and converts them to canonical form.
        Uses Regex to handle punctuation (e.g., "Tylenol?" -> "Tylenol").
        """
        found_drugs = []

        # Regex to find words, ignoring punctuation
        # \b\w+\b matches whole words
        words = re.findall(r'\b\w+\b', query.lower())

        for word in words:
            if word in self.synonym_map:
                found_drugs.append(self.synonym_map[word])

        return list(set(found_drugs))


if __name__ == "__main__":
    processor = InputProcessor()

    test_q = "Can I take Advil with Tylenol?"
    print(f"\nQuery: {test_q}")
    result = processor.extract_and_normalize_drugs(test_q)
    print(f"Extracted Drugs: {result}")

    if not result:
        print("❌ Still failing? Check the DEBUG prints above. If False, your vocab file might need inspecting.")