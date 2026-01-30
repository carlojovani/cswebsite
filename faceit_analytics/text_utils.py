# faceit_analytics/text_utils.py
from __future__ import annotations
from typing import Any

def _looks_mojibake(s: str) -> bool:
    return ("Ð" in s) or ("Ñ" in s) or ("Ã" in s) or ("Р" in s and "С" in s)

def fix_mojibake(s: str) -> str:
    if not s or not isinstance(s, str):
        return s

    if _looks_mojibake(s):
        for enc in ("latin1", "cp1251"):
            try:
                repaired = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
                if not _looks_mojibake(repaired):
                    return repaired
            except Exception:
                pass
    return s

def deep_fix_text(obj: Any) -> Any:
    if isinstance(obj, str):
        return fix_mojibake(obj)
    if isinstance(obj, dict):
        return {deep_fix_text(k): deep_fix_text(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_fix_text(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(deep_fix_text(x) for x in obj)
    return obj
