from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class IntentResult:
    valid: bool
    reason: str


DEFAULT_UNSAFE_PATTERNS: List[str] = [
    r"how much should i take",
    r"dose",
    r"dosage",
    r"diagnose",
    r"what do i have",
    r"symptom checker",
]


def validate_intent(query: str, unsafe_patterns: List[str] = DEFAULT_UNSAFE_PATTERNS) -> IntentResult:
    query_lower = query.lower()
    for pattern in unsafe_patterns:
        if re.search(pattern, query_lower):
            return IntentResult(
                valid=False,
                reason=(
                    f"Safety Violation: Query contains restricted concepts ({pattern}). "
                    "This system is for interaction checking only, not dosage or diagnosis."
                ),
            )
    return IntentResult(valid=True, reason="Safe")


def main() -> int:
    examples: List[Tuple[str, str]] = [
        ("Can I take Advil with Tylenol?", "expected safe"),
        ("What dosage of ibuprofen should I take?", "expected unsafe"),
    ]
    for q, note in examples:
        res = validate_intent(q)
        print(f"{note}: {q}\n-> valid={res.valid} reason={res.reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


