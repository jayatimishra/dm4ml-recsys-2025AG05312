# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Verify every file carries the roll-number watermark.

python tools/check_watermarks.py [root]      exit code 1 if any file is missing it.
Skips tool-internal stores: .git/, .dvc/cache/, .dvc/tmp/, mlartifacts/, __pycache__/.
"""
import sqlite3
import sys
from pathlib import Path

ROLL = "2025AG05312"
SKIP = (".git/", ".dvc/cache/", ".dvc/tmp/", "mlartifacts/", "__pycache__/", ".venv/", ".prefect/")
TEXT = {".py", ".sql", ".md", ".yaml", ".yml", ".txt", ".csv", ".html", ".mmd", ".json", ".lock", ".cfg",
        ".toml", ".log", ""}


def has_mark(p: Path) -> bool:
    suf = p.suffix.lower()
    if suf in TEXT or p.name in (".gitignore", ".dvcignore", "config"):
        return ROLL in p.read_text(errors="ignore")
    if suf == ".png":
        from PIL import Image
        info = Image.open(p).info
        return any(ROLL in str(v) for v in info.values())
    if suf == ".pdf":
        from pypdf import PdfReader
        r = PdfReader(str(p))
        return all(ROLL in (pg.extract_text() or "") for pg in r.pages)
    if suf == ".parquet":
        import pyarrow.parquet as pq
        md = pq.read_schema(p).metadata or {}
        return ROLL.encode() in md.get(b"watermark", b"")
    if suf == ".db":
        with sqlite3.connect(p) as c:
            try:
                return c.execute("SELECT roll_no FROM _watermark").fetchone()[0] == ROLL
            except Exception:
                return False
    if suf == ".npz":
        import numpy as np
        with np.load(p) as z:
            return "watermark" in z.files and ROLL in str(z["watermark"])
    return ROLL.encode() in p.read_bytes()  # pkl and other binaries


def main(root: Path) -> int:
    bad, n = [], 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_dir() or any(rel.startswith(s) or f"/{s}" in f"/{rel}" for s in SKIP):
            continue
        n += 1
        try:
            ok = has_mark(p)
        except Exception as e:  # unreadable file counts as a failure
            ok, rel = False, f"{rel} ({e})"
        if not ok:
            bad.append(rel)
    print(f"checked {n} files: {n - len(bad)} watermarked, {len(bad)} missing")
    for b in bad:
        print("  MISSING:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]))
