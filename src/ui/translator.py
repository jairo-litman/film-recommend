import json
from typing import Dict, Any
from pathlib import Path


class Translator:
    def __init__(self, lang_code: str = "en"):
        json_path = Path(__file__).parent.parent / "translations" / "translations.json"
        with open(json_path, "r", encoding="utf-8") as f:
            translations = json.load(f)

        self.lang_code = lang_code
        self._t = translations.get(lang_code, translations["en"])

    def __call__(self, key: str) -> str:
        return self._t.get(key, f"[MISSING: {key}]")

    def all(self) -> Dict[str, Any]:
        return self._t
