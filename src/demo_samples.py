"""Download a small VITON-HD / Space example set for Studio demo pairs.

Usage:
  python -m src.demo_samples
  python -m src.demo_samples --n 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"
DEMO_DIR = ROOT / "assets" / "demo"

DATASET_CANDIDATES = (
    "SaffalPoosh/VITON-HD-test",
    "TryOnVirtual/VITON-HD-TEST",
    "forgeml/viton_hd",
)

SPACE_SOURCES = (
    ("spaces", "yisol/IDM-VTON", ("example", "examples", "ckpt/example")),
    ("spaces", "zhengchong/CatVTON", ("resource/demo/example",)),
)


def _as_image(value) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        for key in ("image", "bytes", "path", "pil"):
            if key in value:
                img = _as_image(value[key])
                if img is not None:
                    return img
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.exists():
            return Image.open(path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        from io import BytesIO

        return Image.open(BytesIO(value)).convert("RGB")
    return None


def _save_pair(idx: int, person: Image.Image, cloth: Image.Image, title: str, category: str) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{idx:02d}"
    person.save(SAMPLES_DIR / f"person_{stem}.jpg", quality=88, optimize=True)
    cloth.save(SAMPLES_DIR / f"cloth_{stem}.jpg", quality=88, optimize=True)
    (SAMPLES_DIR / f"meta_{stem}.txt").write_text(
        f"title: {title}\ncategory: {category}\nsource: VITON-HD-style demo pair\n",
        encoding="utf-8",
    )


def _from_hf_dataset(n: int) -> int:
    from datasets import load_dataset

    saved = 0
    for repo in DATASET_CANDIDATES:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
        except Exception:
            try:
                ds = load_dataset(repo, split="test", streaming=True)
            except Exception:
                continue
        for row in ds:
            person = None
            cloth = None
            for pkey in ("image", "person", "model", "human", "target"):
                if pkey in row:
                    person = _as_image(row[pkey])
                    if person is not None:
                        break
            for ckey in ("cloth", "garment", "clothes", "clothing", "condition"):
                if ckey in row:
                    cloth = _as_image(row[ckey])
                    if cloth is not None:
                        break
            # Some rows nest both images in a list
            if person is None or cloth is None:
                images = []
                for val in row.values():
                    img = _as_image(val)
                    if img is not None:
                        images.append(img)
                if person is None and images:
                    person = images[0]
                if cloth is None and len(images) > 1:
                    cloth = images[1]
            if person is None or cloth is None:
                continue
            saved += 1
            _save_pair(
                saved,
                person,
                cloth,
                title=f"VITON-HD demo {saved:02d}",
                category="upper",
            )
            if saved >= n:
                return saved
        if saved:
            return saved
    return saved


def _list_repo_images(repo_id: str, repo_type: str, prefixes: tuple[str, ...]) -> list[str]:
    from huggingface_hub import list_repo_files

    from .hf_auth import ensure_hf_login, hf_token

    ensure_hf_login()
    files = list_repo_files(repo_id, repo_type=repo_type, token=hf_token())
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = []
    for name in files:
        lower = name.lower().replace("\\", "/")
        if Path(name).suffix.lower() not in exts:
            continue
        if any(lower.startswith(p.lower().rstrip("/") + "/") or f"/{p.lower().rstrip('/')}/" in lower for p in prefixes):
            out.append(name)
        elif "example" in lower and ("person" in lower or "human" in lower or "cloth" in lower or "condition" in lower or "garm" in lower):
            out.append(name)
    return sorted(out)


def _from_space_examples(n: int) -> int:
    from huggingface_hub import hf_hub_download

    from .hf_auth import ensure_hf_login, hf_token

    ensure_hf_login()
    token = hf_token()
    saved = 0
    for repo_type, repo_id, prefixes in SPACE_SOURCES:
        try:
            files = _list_repo_images(repo_id, repo_type, prefixes)
        except Exception:
            continue
        persons = [
            f
            for f in files
            if any(k in f.lower() for k in ("person", "human", "model", "men", "women"))
            and "cloth" not in f.lower()
            and "condition" not in f.lower()
            and "garm" not in f.lower()
        ]
        clothes = [
            f
            for f in files
            if any(k in f.lower() for k in ("cloth", "condition", "garm", "upper", "dress"))
            and "person" not in f.lower()
            and "human" not in f.lower()
        ]
        if not persons or not clothes:
            continue
        count = min(n - saved, len(persons), len(clothes))
        for i in range(count):
            try:
                p_local = hf_hub_download(repo_id, persons[i], repo_type=repo_type, token=token)
                c_local = hf_hub_download(repo_id, clothes[i], repo_type=repo_type, token=token)
                person = Image.open(p_local).convert("RGB")
                cloth = Image.open(c_local).convert("RGB")
            except Exception:
                continue
            saved += 1
            category = "dress" if "overall" in clothes[i].lower() or "dress" in clothes[i].lower() else "upper"
            if "lower" in clothes[i].lower():
                category = "lower"
            _save_pair(
                saved,
                person,
                cloth,
                title=f"Demo pair {saved:02d}",
                category=category,
            )
            if saved >= n:
                return saved
        if saved:
            return saved
    return saved


def cache_first_pair_as_fallback() -> Optional[Path]:
    """Copy the first sample try-on-ready person as a demo fallback if none exists."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DEMO_DIR.glob("tryon_*.png")) + list(DEMO_DIR.glob("tryon_*.jpg"))
    if existing:
        return existing[0]
    first = next(iter(sorted(SAMPLES_DIR.glob("person_*.jpg"))), None)
    if first is None:
        return None
    dest = DEMO_DIR / "tryon_01.jpg"
    Image.open(first).convert("RGB").save(dest, quality=88)
    return dest


def download_demo_pairs(n: int = 16) -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    saved = _from_hf_dataset(n)
    if saved < n:
        extra = _from_space_examples(n - saved)
        saved += extra
    cache_first_pair_as_fallback()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download VITON-HD-style Studio demo pairs")
    parser.add_argument("--n", type=int, default=16, help="Number of person/garment pairs")
    args = parser.parse_args()
    n = download_demo_pairs(args.n)
    print(f"Saved {n} demo pairs -> {SAMPLES_DIR}")
    if n == 0:
        print(
            "No pairs downloaded. Add person_XX.jpg + cloth_XX.jpg under data/samples/ "
            "or check HF_TOKEN / network access."
        )


if __name__ == "__main__":
    main()
