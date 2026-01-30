# -*- coding: utf-8 -*-
"""
Patch users/templates/users/profile.html to remove unsupported dict indexing like:
  some.path[bucket]
and replace with:
  some.path|get_item:bucket

Also ensures {% load dict_extras %} is present near top.

Usage:
  python tools/patch_profile_template_dict_access.py --write
"""
import argparse, re
from pathlib import Path

TEMPLATE = Path("users/templates/users/profile.html")

PATTERN = re.compile(r"(?P<expr>[A-Za-z0-9_\.]+)\[bucket\]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")

    s = TEMPLATE.read_text(encoding="utf-8", errors="replace")

    # ensure load tag
    if "{% load dict_extras %}" not in s:
        # insert right after extends if possible
        s = re.sub(r"(\{%\s*extends\s+[^%]+%\}\s*\n)", r"\1{% load dict_extras %}\n", s, count=1)

    replaced = 0
    def repl(m):
        nonlocal replaced
        replaced += 1
        return f"{m.group('expr')}|get_item:bucket"

    s2 = PATTERN.sub(repl, s)

    print("replaced [bucket] occurrences:", replaced)
    if args.write and s2 != s:
        TEMPLATE.write_text(s2, encoding="utf-8", newline="\n")
        print("written:", TEMPLATE)
    else:
        print("preview mode (no write)")

if __name__ == "__main__":
    main()
