# -*- coding: utf-8 -*-
"""
Fix mojibake everywhere (files + DB JSON), in one script.

Common broken pattern: UTF-8 bytes decoded as CP1251/Latin-1, producing "РђС..." / "вЂ”" etc.

Usage:
  python tools/fix_mojibake_everywhere_one.py            # preview
  python tools/fix_mojibake_everywhere_one.py --write    # write changes

It is conservative:
- For files: only writes if the fix reduces "bad marker" count.
- For DB JSON: only writes if the fix reduces bad marker count.

Works on Windows / PowerShell.
"""
import argparse, re, sys
from pathlib import Path

BAD_RE = re.compile(r"(Р[А-Яа-я]|С[А-Яа-я]|вЂ|Ð|Â)", re.U)

TEXT_EXTS = {".py", ".html", ".htm", ".txt", ".md", ".css", ".js", ".json", ".yml", ".yaml"}

def count_bad(s: str) -> int:
    return len(BAD_RE.findall(s))

def try_fix_mojibake(s: str) -> str:
    """
    Best-effort:
    1) If s looks like mojibake, try cp1251->utf8 roundtrip.
    2) Also try latin-1->utf8 (for 'Â', 'Ð' etc).
    Keep original if conversion fails.
    """
    if not BAD_RE.search(s):
        return s

    best = s
    best_bad = count_bad(s)

    # cp1251 -> utf8
    try:
        cand = s.encode("cp1251", errors="strict").decode("utf-8", errors="strict")
        if count_bad(cand) < best_bad:
            best, best_bad = cand, count_bad(cand)
    except Exception:
        pass

    # latin-1 -> utf8
    try:
        cand = s.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        if count_bad(cand) < best_bad:
            best, best_bad = cand, count_bad(cand)
    except Exception:
        pass

    return best

def deep_fix(obj):
    if isinstance(obj, str):
        return try_fix_mojibake(obj)
    if isinstance(obj, list):
        return [deep_fix(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = deep_fix(k) if isinstance(k, str) else k
            out[kk] = deep_fix(v)
        return out
    return obj

def scan_files(root: Path, write: bool) -> tuple[int, int]:
    files_with_bad = 0
    files_changed = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue

        # decode as utf-8 (fallback replace to allow scanning)
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            s = raw.decode("utf-8", errors="replace")

        bad0 = count_bad(s)
        if bad0 == 0:
            continue
        files_with_bad += 1

        fixed = try_fix_mojibake(s)
        bad1 = count_bad(fixed)

        if bad1 < bad0:
            files_changed += 1
            print(f"[FILE] {p} -> FIX {bad0} -> {bad1}")
            if write:
                p.write_text(fixed, encoding="utf-8", newline="\n")
        else:
            print(f"[FILE] {p} -> HAS BAD MARKERS but no safe change")

    return files_with_bad, files_changed

def scan_db(write: bool) -> tuple[int, int, int]:
    """
    DB part is optional; it will run only inside Django environment.
    """
    try:
        from faceit_analytics.models import AnalyticsAggregate
    except Exception as e:
        print("DB scan skipped (no Django/faceit_analytics):", e)
        return 0, 0, 0

    scanned = 0
    rows_with_bad = 0
    rows_changed = 0

    for row in AnalyticsAggregate.objects.all().iterator():
        scanned += 1
        mj = row.metrics_json or {}
        s = str(mj)
        bad0 = count_bad(s)
        if bad0 == 0:
            continue
        rows_with_bad += 1
        fixed = deep_fix(mj)
        bad1 = count_bad(str(fixed))
        if bad1 < bad0:
            rows_changed += 1
            print(f"[DB] AnalyticsAggregate id={row.id} -> FIX {bad0} -> {bad1}")
            if write:
                row.metrics_json = fixed
                row.save(update_fields=["metrics_json"])

    return scanned, rows_with_bad, rows_changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Write changes to disk/DB")
    ap.add_argument("--root", default=".", help="Project root to scan (default '.')")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print("== Scanning files ==")
    f_bad, f_changed = scan_files(root, write=args.write)
    print(f"\nFiles with bad markers: {f_bad}, files changed: {f_changed}")

    print("\n== Scanning DB ==")
    scanned, rows_bad, rows_changed = scan_db(write=args.write)
    print(f"DB scanned: {scanned}, rows with bad markers: {rows_bad}, rows changed: {rows_changed}")

    print("\n== Done ==")

if __name__ == "__main__":
    main()
