from pathlib import Path
import re

p = Path("tools/check_demo_integrity.py")
s = p.read_text(encoding="utf-8")

# 1) insert helper once
if "_pick_df(" not in s:
    m = re.search(r"(?m)^(import .*\n)+", s)
    if not m:
        raise SystemExit("Can't find import block to inject _pick_df()")

    ins = """

# --- polars-safe df picker (no bool(polars.DataFrame)) ---
def _pick_df(*cands):
    for x in cands:
        if x is None:
            continue
        try:
            if hasattr(x, "is_empty") and x.is_empty():
                continue
        except Exception:
            pass
        return x
    return None

"""
    s = s[:m.end()] + ins + s[m.end():]

# 2) replace bomb df picker
s_new = re.sub(
    r"bomb\s*=\s*getattr\(dem,\s*['\"]bomb['\"]\s*,\s*None\)\s*or\s*getattr\(dem,\s*['\"]bombs['\"]\s*,\s*None\)",
    "bomb = _pick_df(getattr(dem, 'bomb', None), getattr(dem, 'bombs', None))",
    s,
)

if s_new == s:
    print("NOTE: pattern didn't match. I'll still try a simpler replacement...")
    s_new2 = s.replace(
        "bomb = getattr(dem, \"bomb\", None) or getattr(dem, \"bombs\", None)",
        "bomb = _pick_df(getattr(dem, 'bomb', None), getattr(dem, 'bombs', None))",
    )
    if s_new2 != s:
        s_new = s_new2

s = s_new

p.write_text(s, encoding="utf-8")
print("patched:", p)
