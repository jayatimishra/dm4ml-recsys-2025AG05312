# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stamp the roll-number watermark on assets not produced by the pipeline's own
writers: UI screenshots, the Mermaid architecture PNG, the MLflow DB and DVC files.
Idempotent: files already carrying the watermark are skipped."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

from src.common import ROLL_NO, ROOT, WATERMARK, stamp_sqlite  # noqa: E402


def _font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def stamp_png(path: Path) -> bool:
    im = Image.open(path)
    if WATERMARK in (im.info.get("Description", "") + im.info.get("watermark", "")):
        return False
    im = im.convert("RGBA")
    w, h = im.size
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    big = _font(max(28, w // 22))
    tile = Image.new("RGBA", (int(w * 0.6), max(60, w // 14)), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((10, 5), WATERMARK, font=big, fill=(128, 128, 128, 40))
    tile = tile.rotate(28, expand=True)
    for y in range(0, h, max(tile.size[1], 400)):
        layer.alpha_composite(tile, (max(0, (w - tile.size[0]) // 2), min(y, max(0, h - tile.size[1]))))
    small = _font(max(14, w // 110))
    tw = d.textlength(WATERMARK, font=small)
    d.rectangle([w - tw - 24, h - small.size - 18, w, h], fill=(255, 255, 255, 220))
    d.text((w - tw - 12, h - small.size - 12), WATERMARK, font=small, fill=(60, 60, 60, 255))
    out = Image.alpha_composite(im, layer).convert("RGB")
    meta = PngInfo()
    meta.add_text("Description", WATERMARK)
    meta.add_text("watermark", WATERMARK)
    out.save(path, pnginfo=meta)
    return True


def prepend_comment(path: Path, prefix: str = "# ") -> bool:
    if not path.exists():
        return False
    s = path.read_text()
    if ROLL_NO in s:
        return False
    path.write_text(f"{prefix}{WATERMARK} | AIML ZG529 Assignment\n{s}")
    return True


def main():
    pngs = [*(ROOT / "docs" / "screenshots").glob("*.png"), *(ROOT / "docs" / "execution").glob("*.png"),
            *(ROOT / "docs" / "execution_mac").glob("*.png"), ROOT / "docs" / "architecture.png"]
    for p in pngs:
        if p.exists() and stamp_png(p):
            print("stamped", p.relative_to(ROOT))
    if (ROOT / "mlflow.db").exists():
        with sqlite3.connect(ROOT / "mlflow.db") as c:
            stamp_sqlite(c)
    extra = [p for p in ROOT.rglob(".gitignore") if ".git/" not in p.as_posix() and ".dvc/cache" not in p.as_posix()]
    for p in [ROOT / ".dvc" / "config", ROOT / ".dvcignore", ROOT / "dvc.lock", *extra]:
        if prepend_comment(p):
            print("commented", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
