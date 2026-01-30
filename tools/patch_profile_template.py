"""tools/patch_profile_template.py

Автоматически заменяет в users/templates/users/profile.html обращения вида:
  some_dict[bucket]
на корректное для Django templates:
  some_dict|get_item:bucket

Также добавляет `{% load dict_extras %}` если его нет.

Запуск:
  python tools/patch_profile_template.py --write

По умолчанию (без --write) печатает изменения и ничего не пишет.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TEMPLATE_PATH = Path("users/templates/users/profile.html")

# очень конкретно: ...something...[bucket]
BRACKET_RE = re.compile(r"([A-Za-z0-9_\.]+)\[bucket\]")


def patch(text: str) -> str:
    out = text
    # ensure load tag after extends
    if "{% load dict_extras %}" not in out:
        lines = out.splitlines()
        new_lines = []
        inserted = False
        for i, line in enumerate(lines):
            new_lines.append(line)
            if not inserted and line.strip().startswith("{% extends"):
                new_lines.append("{% load dict_extras %}")
                inserted = True
        out = "\n".join(new_lines)

    # replace [bucket] lookups
    out = BRACKET_RE.sub(r"\1|get_item:bucket", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template not found: {TEMPLATE_PATH}")

    text = TEMPLATE_PATH.read_text(encoding="utf-8", errors="replace")
    patched = patch(text)

    if patched == text:
        print("No changes needed")
        return

    if not args.write:
        print("=== PREVIEW (use --write to apply) ===")
        # print a small diff-like preview
        old_snip = "\n".join(text.splitlines()[0:25])
        new_snip = "\n".join(patched.splitlines()[0:25])
        print("--- old head ---\n" + old_snip)
        print("--- new head ---\n" + new_snip)
        return

    TEMPLATE_PATH.write_text(patched, encoding="utf-8")
    print(f"Patched: {TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
