from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import config
from .util import utc_now, write_json

OCR_LANG = "chi_sim+eng"


def ocr_available() -> bool:
    return bool(getattr(config, "ENABLE_OCR", False) and shutil.which("tesseract"))


def tesseract_version() -> str:
    try:
        out = subprocess.check_output(["tesseract", "--version"], text=True, stderr=subprocess.STDOUT)
        return out.splitlines()[0].strip()
    except Exception:
        return "tesseract"


def _preprocess(image: Path) -> Path:
    """Boost contrast for scanned Chinese pages. Returns a temp PNG path."""
    try:
        from PIL import Image, ImageFilter, ImageOps
    except Exception:
        return image
    im = Image.open(image)
    if im.mode not in {"L", "RGB"}:
        im = im.convert("RGB")
    im = ImageOps.grayscale(im)
    # upscale small scans
    w, h = im.size
    if max(w, h) < 1600:
        im = im.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    im = ImageOps.autocontrast(im, cutoff=1)
    im = im.filter(ImageFilter.MedianFilter(size=3))
    out = image.with_suffix(".prep.png")
    im.save(out, "PNG")
    return out


def _ocr_tmp() -> Path:
    path = config.PROCESSED / "ocr_tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_tesseract(image: Path, psm: str = "4") -> dict[str, Any]:
    work = _preprocess(image)
    with tempfile.TemporaryDirectory(dir=_ocr_tmp()) as td:
        base = Path(td) / "out"
        proc = subprocess.run(
            ["tesseract", str(work), str(base), "-l", OCR_LANG, "--oem", "1", "--psm", psm],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        subprocess.run(
            ["tesseract", str(work), str(base), "-l", OCR_LANG, "--oem", "1", "--psm", psm, "tsv"],
            capture_output=True,
            timeout=120,
        )
        txt_path = Path(str(base) + ".txt")
        tsv_path = Path(str(base) + ".tsv")
        text = txt_path.read_text(encoding="utf-8", errors="replace") if txt_path.exists() else ""
        confs: list[float] = []
        if tsv_path.exists():
            for i, line in enumerate(tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()):
                if i == 0:
                    continue
                parts = line.split("\t")
                if len(parts) >= 11:
                    try:
                        c = float(parts[10])
                    except ValueError:
                        continue
                    if c >= 0:
                        confs.append(c)
        mean = sum(confs) / len(confs) if confs else None
        if work != image:
            work.unlink(missing_ok=True)
        return {
            "text": text.strip(),
            "mean_conf": round(mean, 2) if mean is not None else None,
            "n_tokens": len(confs),
            "engine": tesseract_version(),
            "psm": psm,
            "returncode": proc.returncode,
        }


def _best_tesseract(image: Path) -> dict[str, Any]:
    a = _run_tesseract(image, "4")
    # retry column/sparse pages
    compact = "".join((a.get("text") or "").split())
    if len(compact) < 40 or (a.get("mean_conf") or 0) < 55:
        b = _run_tesseract(image, "6")
        if len("".join((b.get("text") or "").split())) > len(compact):
            return b
    return a


def ocr_image_file(path: Path) -> dict[str, Any]:
    rec = _best_tesseract(path)
    rec.update({"page": 1, "locator": "ocr-image"})
    return rec


def ocr_pdf_pages(path: Path, max_pages: int = 60) -> list[dict[str, Any]]:
    import fitz

    pages = []
    doc = fitz.open(path)
    try:
        n = min(len(doc), max_pages)
        for i in range(n):
            page = doc[i]
            pix = page.get_pixmap(dpi=200, alpha=False)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=_ocr_tmp()) as tmp:
                tmp.write(pix.tobytes("png"))
                img_path = Path(tmp.name)
            try:
                rec = _best_tesseract(img_path)
            finally:
                img_path.unlink(missing_ok=True)
            rec.update({"page": i + 1, "locator": f"ocr-p{i+1}"})
            pages.append(rec)
    finally:
        doc.close()
    return pages


def archive_ocr(content_hash: str, pages: list[dict[str, Any]], source_path: str) -> Path:
    out_dir = config.PROCESSED / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_hash": content_hash,
        "source_path": source_path,
        "engine": tesseract_version(),
        "lang": OCR_LANG,
        "created_at": utc_now(),
        "pages": pages,
        "text": "\n\n".join(p.get("text") or "" for p in pages),
    }
    dest = out_dir / f"{content_hash}.json"
    write_json(dest, payload)
    (out_dir / f"{content_hash}.txt").write_text(payload["text"], encoding="utf-8")
    return dest


def pdf_needs_ocr(native_text: str, n_pages: int) -> bool:
    compact = "".join((native_text or "").split())
    if n_pages <= 0:
        return False
    return len(compact) < max(80, 100 * n_pages)


def backfill_low_quality(limit: int = 40) -> dict:
    """Re-run improved OCR on archived results that are short or low-confidence."""
    out_dir = config.PROCESSED / "ocr"
    if not out_dir.exists() or not ocr_available():
        return {"updated": 0, "skipped": 0}
    updated = skipped = 0
    hashes: list[str] = []
    for jp in sorted(out_dir.glob("*.json")):
        if updated >= limit:
            break
        try:
            data = __import__("json").loads(jp.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        text = data.get("text") or ""
        confs = [p.get("mean_conf") for p in (data.get("pages") or []) if p.get("mean_conf") is not None]
        mean = sum(confs) / len(confs) if confs else None
        compact = "".join(text.split())
        if compact and mean and mean >= 60 and len(compact) >= 80:
            skipped += 1
            continue
        src = Path(data.get("source_path") or "")
        if not src.exists():
            skipped += 1
            continue
        try:
            if src.suffix.lower() == ".pdf":
                pages = ocr_pdf_pages(src)
            else:
                pages = [ocr_image_file(src)]
            archive_ocr(jp.stem, pages, str(src))
            hashes.append(jp.stem)
            updated += 1
        except Exception:
            skipped += 1
    return {"updated": updated, "skipped": skipped, "hashes": hashes}
