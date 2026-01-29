from __future__ import annotations
import sys
import time
import zipfile
from pathlib import Path
import requests

def download(url: str, dst: Path, *, timeout: int = 20, retries: int = 3) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                got = 0
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            pct = got * 100 // total
                            print(f"\r{dst.name}: {pct}% ({got}/{total} bytes)", end="")
                print()
            return
        except Exception as e:
            last_err = e
            print(f"[{attempt}/{retries}] failed: {e}")
            time.sleep(1.5 * attempt)
    raise SystemExit(f"Download failed after {retries} retries: {last_err}")

def unzip(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dst_dir)

if __name__ == "__main__":
    # usage: python scripts/download_awpy_artifacts.py <patch> <maps|navs|tris>
    patch = sys.argv[1]
    what = sys.argv[2]
    url = f"https://www.awpycs.com/{patch}/{what}.zip"
    home = Path.home()
    awpy_dir = home / ".awpy"  # место, куда awpy по умолчанию кладёт артефакты :contentReference[oaicite:3]{index=3}
    zip_path = awpy_dir / "downloads" / f"{what}_{patch}.zip"
    out_dir = awpy_dir / what

    print("URL:", url)
    print("ZIP:", zip_path)
    print("OUT:", out_dir)

    download(url, zip_path)
    unzip(zip_path, out_dir)
    print("Done.")
