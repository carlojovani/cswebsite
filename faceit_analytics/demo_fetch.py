import gzip
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests


def download_file(url: str, out_path: Path, timeout: int = 120):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _extract_first_dem_from_zip(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        dem_names = [n for n in zf.namelist() if n.lower().endswith(".dem")]
        if not dem_names:
            raise RuntimeError("No .dem in zip")
        target = dem_names[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(target, out_dir)
        return (out_dir / target).resolve()


def _extract_first_dem_from_tar(tar_path: Path, out_dir: Path) -> Path:
    with tarfile.open(tar_path, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.name.lower().endswith(".dem")]
        if not members:
            raise RuntimeError("No .dem in tar")
        m = members[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        tf.extract(m, out_dir)
        return (out_dir / m.name).resolve()


def _gunzip_if_needed(path: Path) -> Path:
    if path.suffix.lower() != ".gz":
        return path
    out_path = path.with_suffix("")
    with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def get_demo_dem_path(signed_url: str, work_dir: Path) -> Path:
    """
    Скачивает файл (demo / zip / gz / tar.*) и возвращает путь к .dem
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_path = work_dir / "demo_download"
    download_file(signed_url, raw_path)

    try:
        if zipfile.is_zipfile(raw_path):
            return _extract_first_dem_from_zip(raw_path, work_dir / "unzipped")
    except Exception:
        pass

    try:
        if tarfile.is_tarfile(raw_path):
            return _extract_first_dem_from_tar(raw_path, work_dir / "untarred")
    except Exception:
        pass

    try:
        with open(raw_path, "rb") as f:
            head = f.read(2)
        if head == b"\x1f\x8b":
            gz_path = raw_path.with_suffix(".gz")
            os.replace(raw_path, gz_path)
            ungz = _gunzip_if_needed(gz_path)
            if ungz.suffix.lower() == ".dem":
                return ungz
            if zipfile.is_zipfile(ungz):
                return _extract_first_dem_from_zip(ungz, work_dir / "unzipped2")
            if tarfile.is_tarfile(ungz):
                return _extract_first_dem_from_tar(ungz, work_dir / "untarred2")
    except Exception:
        pass

    dem_path = work_dir / "match.dem"
    os.replace(raw_path, dem_path)
    return dem_path


def get_local_dem_path(demo_dir: Path) -> Path:
    """
    Ищет match.dem или match.dem.zst в demo_dir и возвращает путь к .dem
    """
    demo_dir.mkdir(parents=True, exist_ok=True)

    dem = demo_dir / "match.dem"
    dem_zst = demo_dir / "match.dem.zst"

    if dem.exists():
        return dem

    if dem_zst.exists():
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Install zstandard: pip install zstandard") from exc

        out_path = dem
        dctx = zstd.ZstdDecompressor()
        with open(dem_zst, "rb") as fin, open(out_path, "wb") as fout:
            dctx.copy_stream(fin, fout)
        return out_path

    raise FileNotFoundError(f"Put match.dem or match.dem.zst into {demo_dir}")
