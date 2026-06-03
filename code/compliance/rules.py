from typing import Dict, List

# Required PPE classes per compliance context.
# A person is compliant only if ALL required classes are correctly placed.

COMPLIANCE_RULES: Dict[str, List[str]] = {
    "construction": [
        "helmet",
        "vest",
    ],
    "construction_full": [
        "helmet",
        "vest",
        "gloves",
        "shoes",
    ],
    "medical": [
        "mask",
        "gloves",
        "medical-suit",
    ],
    "chv": [
        "helmet",
        "vest",
    ],
    "cppe5": [
        "coverall",
        "mask",
        "gloves",
    ],
    "sh17": [
        "helmet",
        "safety-vest",
        "gloves",
    ],
}

DEFAULT_RULE = "construction"
