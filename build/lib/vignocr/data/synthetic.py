"""Synthetic vignette fixture generator (Pillow only — no ML libs).

Draws vignette-like images deterministically (seeded RNG) so the whole pipeline
can run on CPU **today** against a fixture that mirrors the real Roboflow COCO
export byte-for-byte in structure. Alongside each split's images and
``_annotations.coco.json`` it writes:

* ``fixtures/nomenclature.csv``  — columns from ``correction.yaml`` with rows
  matching every drawn ``num_enregistrement`` code (so the matcher always hits).
* ``ground_truth.json``          — per-image expected field values, drawn band
  reimbursability, and pixel ``xywh`` boxes for every drawn class.

Invariants (asserted by ``tests`` and ``validate``):
  * ``prix + shp == ppa`` to the centime (money is :class:`decimal.Decimal`).
  * every ``num_enregistrement`` is a key present in the written nomenclature CSV.
  * the stored ``reimbursability`` matches the colour of the drawn band.
  * splits never share an image (file_name stems are globally unique).

Nothing here is hardcoded that the schema owns: class names, the reimbursability
palette, the code letter set, and the nomenclature columns all flow from configs.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vignocr.common import get_classes, get_logger, load_config, money_str

log = get_logger(__name__)

# Shared fixture contract: the generator writes this file at the dataset root,
# keyed by image file_name. Pipeline stubs and tests read it back via
# :func:`load_ground_truth`.
GROUND_TRUTH_FILENAME = "ground_truth.json"

# Roboflow-style COCO boilerplate (shape derived from the real export under data/).
_COCO_INFO_DESCRIPTION = "VignOCR synthetic fixture (generated)"
_COCO_LICENSE = {"id": 1, "url": "", "name": "Proprietary (synthetic)"}

# Band fill colours per reimbursability key. Kept distinct enough that the
# reimbursability head's colour read is unambiguous; orange is the abstain band.
_BAND_RGB: dict[str, tuple[int, int, int]] = {
    "green": (40, 170, 70),
    "red": (200, 40, 40),
    "orange": (235, 150, 30),
}

# --------------------------------------------------------------------------- #
# Deterministic value pools (small, fixture-only — NOT the real nomenclature)
# --------------------------------------------------------------------------- #

_PRODUCTS: tuple[tuple[str, str, str, str, str], ...] = (
    # (product_name, dci, dosage, forme, laboratoire)
    ("DOLIPRANE", "PARACETAMOL", "1000 MG", "COMPRIME", "SANOFI"),
    ("AUGMENTIN", "AMOXICILLINE/AC. CLAV.", "1 G", "COMPRIME", "GSK"),
    ("CLAMOXYL", "AMOXICILLINE", "500 MG", "GELULE", "GSK"),
    ("EFFERALGAN", "PARACETAMOL", "500 MG", "COMPRIME EFFERV.", "UPSA"),
    ("VENTOLINE", "SALBUTAMOL", "100 UG/DOSE", "AEROSOL", "GSK"),
    ("INEXIUM", "ESOMEPRAZOLE", "40 MG", "COMPRIME", "ASTRAZENECA"),
    ("LEVOTHYROX", "LEVOTHYROXINE", "75 UG", "COMPRIME", "MERCK"),
    ("KARDEGIC", "ACETYLSALICYLATE", "75 MG", "SACHET", "SANOFI"),
    ("GLUCOPHAGE", "METFORMINE", "850 MG", "COMPRIME", "MERCK"),
    ("TAHOR", "ATORVASTATINE", "20 MG", "COMPRIME", "PFIZER"),
    ("LASILIX", "FUROSEMIDE", "40 MG", "COMPRIME", "SANOFI"),
    ("AMLOR", "AMLODIPINE", "5 MG", "GELULE", "PFIZER"),
)

# Code letter blocks come from the parsing config (never hardcoded here).
_NUM_LOT_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


# --------------------------------------------------------------------------- #
# Value synthesis (deterministic, from the seeded RNG)
# --------------------------------------------------------------------------- #


def _letters() -> list[str]:
    """Known ``num_enregistrement`` letter blocks from parsing/fields.yaml."""
    fields = load_config("parsing/fields")
    letters = fields["fields"]["num_enregistrement"].get("letters", [])
    return list(letters) or ["D"]


def _nomenclature_columns() -> list[str]:
    """CSV columns from correction.yaml (single source of truth)."""
    cfg = load_config("nomenclature/correction")
    return list(cfg["csv"]["columns"])


def _make_enregistrement(rng: random.Random, letters: list[str]) -> str:
    """Build a code matching fields.yaml num_enregistrement.normalized_format.

    Structure: ``AA/BB/CC<LETTER>DDD/EEE`` (canonical, no internal spaces).
    """
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    c = rng.randint(0, 99)
    letter = rng.choice(letters)
    d = rng.randint(0, 999)
    e = rng.randint(0, 999)
    return f"{a:02d}/{b:02d}/{c:02d}{letter}{d:03d}/{e:03d}"


def _make_num_lot(rng: random.Random) -> str:
    """A lot code matching fields.yaml num_lot.regex (uppercase, 2..19 chars)."""
    n = rng.randint(4, 8)
    first = rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
    rest = "".join(rng.choice(_NUM_LOT_ALPHABET) for _ in range(n - 1))
    return (first + rest).upper()


def _make_dates(rng: random.Random) -> tuple[str, str]:
    """(date_fab, date_exp) as ``YYYY-MM`` with EXP strictly after FAB."""
    fab_year = rng.randint(2021, 2024)
    fab_month = rng.randint(1, 12)
    shelf_years = rng.randint(2, 4)
    exp_year = fab_year + shelf_years
    exp_month = fab_month
    return f"{fab_year:04d}-{fab_month:02d}", f"{exp_year:04d}-{exp_month:02d}"


def _make_money(rng: random.Random) -> tuple[Decimal, Decimal, Decimal]:
    """(prix, shp, ppa) with ``prix + shp == ppa`` exactly to the centime."""
    prix = (Decimal(rng.randint(2000, 250000)) / Decimal(100)).quantize(Decimal("0.01"))
    shp = (Decimal(rng.randint(0, 5000)) / Decimal(100)).quantize(Decimal("0.01"))
    ppa = (prix + shp).quantize(Decimal("0.01"))
    return prix, shp, ppa


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _font(size: int) -> ImageFont.ImageFont:
    """A deterministic bitmap font (PIL default — present everywhere, no files)."""
    try:
        # Pillow >= 10 supports a sizeable default font.
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: size-less default
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Width/height of ``text`` in ``font`` (textbbox is the modern API)."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_horizontal_field(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    *,
    pad: int = 3,
) -> tuple[int, int, int, int]:
    """Draw ``text`` at (x, y); return its padded pixel box (x, y, w, h)."""
    tw, th = _text_size(draw, text, font)
    draw.text((x, y), text, fill=(15, 15, 15), font=font)
    return (x - pad, y - pad, tw + 2 * pad, th + 2 * pad)


def _draw_vertical_field(
    base: Image.Image,
    text: str,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    *,
    pad: int = 3,
) -> tuple[int, int, int, int]:
    """Render ``text`` rotated 90° (bottom-to-top) and paste it at (x, y).

    Returns the pasted region's pixel box. Used for the vertical VIN strip
    fields (num_lot / date_fab / date_exp) which are rotated on real vignettes.
    """
    tmp_draw = ImageDraw.Draw(base)
    tw, th = _text_size(tmp_draw, text, font)
    # Horizontal text tile, then rotate; expand=True keeps the full glyph run.
    tile = Image.new("RGB", (tw + 2 * pad, th + 2 * pad), (245, 245, 245))
    ImageDraw.Draw(tile).text((pad, pad), text, fill=(15, 15, 15), font=font)
    rotated = tile.rotate(90, expand=True)
    rw, rh = rotated.size
    base.paste(rotated, (x, y))
    return (x, y, rw, rh)


def _render_image(
    rng: random.Random,
    image_size: tuple[int, int],
    fields: dict[str, str],
    color: str,
) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    """Draw one vignette; return the image and a class-name -> pixel box map.

    Layout:
      * ``entete``      — light body rectangle (the structural region).
      * ``color_band``  — a diagonal coloured band crossing the body.
      * horizontal fields (ppa/prix/shp/num_enregistrement/identity) stacked
        in the body, each with its own box.
      * ``vin``         — a left vertical strip holding rotated num_lot /
        date_fab / date_exp, each with its own box.
    """
    w, h = image_size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    boxes: dict[str, tuple[int, int, int, int]] = {}

    # --- entete: the light body rectangle (structural region) ---------------
    margin = max(8, w // 32)
    strip_w = max(40, w // 7)  # width reserved for the left vertical VIN strip
    body = (margin, margin, w - margin, h - margin)
    draw.rectangle(body, fill=(248, 248, 244), outline=(120, 120, 120), width=2)
    boxes["entete"] = (body[0], body[1], body[2] - body[0], body[3] - body[1])

    # --- color_band: a diagonal coloured band crossing the body -------------
    band_rgb = _BAND_RGB[color]
    band_thickness = max(14, h // 18)
    # Diagonal from lower-left to upper-right of the body.
    draw.line(
        [(body[0], body[3] - band_thickness), (body[2], body[1] + band_thickness)],
        fill=band_rgb,
        width=band_thickness,
    )
    # Axis-aligned bbox enclosing the diagonal band (COCO boxes are axis-aligned).
    boxes["color_band"] = (
        body[0],
        body[1] + band_thickness // 2,
        body[2] - body[0],
        (body[3] - band_thickness) - (body[1] + band_thickness // 2) + band_thickness,
    )

    # --- horizontal fields stacked in the body (right of the VIN strip) -----
    font = _font(13)
    content_x = body[0] + strip_w + 6
    y = body[1] + 10
    line_gap = 6

    horizontal_order = [
        ("product_name", fields["product_name"]),
        ("dci", fields["dci"]),
        ("dosage", fields["dosage"]),
        ("forme", fields["forme"]),
        ("laboratoire", fields["laboratoire"]),
        ("num_enregistrement", fields["num_enregistrement"]),
        ("prix", f"PRIX {fields['prix']} DA"),
        ("shp", f"SHP {fields['shp']} DA"),
        ("ppa", f"PPA = {fields['ppa']} DA"),
    ]
    for name, text in horizontal_order:
        box = _draw_horizontal_field(draw, text, content_x, y, font)
        boxes[name] = box
        _, th = _text_size(draw, text, font)
        y += th + line_gap

    # --- vin: left vertical strip with rotated num_lot/date_fab/date_exp -----
    vin_x0 = body[0] + 4
    vin_y0 = body[1] + 6
    vin_x1 = body[0] + strip_w
    vin_y1 = body[3] - 6
    draw.rectangle([vin_x0, vin_y0, vin_x1, vin_y1], outline=(90, 90, 90), width=1)
    boxes["vin"] = (vin_x0, vin_y0, vin_x1 - vin_x0, vin_y1 - vin_y0)

    vfont = _font(12)
    vx = vin_x0 + 3
    for name in ("num_lot", "date_fab", "date_exp"):
        vbox = _draw_vertical_field(img, fields[name], vx, vin_y0 + 4, vfont)
        boxes[name] = vbox
        vx += vbox[2] + 4  # advance horizontally for the next vertical column

    return img, boxes


# --------------------------------------------------------------------------- #
# COCO assembly (byte-compatible with the real Roboflow export)
# --------------------------------------------------------------------------- #


def _coco_categories() -> list[dict[str, Any]]:
    """Categories array from the schema (id/name contiguous; Roboflow shape)."""
    schema = get_classes()
    return [
        {"id": int(c["id"]), "name": str(c["name"]), "supercategory": "none"}
        for c in sorted(schema.classes, key=lambda c: c["id"])
    ]


def _empty_coco(now_iso: str) -> dict[str, Any]:
    return {
        "info": {
            "year": str(datetime.now(timezone.utc).year),
            "version": "1",
            "description": _COCO_INFO_DESCRIPTION,
            "contributor": "",
            "url": "",
            "date_created": now_iso,
        },
        "licenses": [dict(_COCO_LICENSE)],
        "categories": _coco_categories(),
        "images": [],
        "annotations": [],
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate(
    out_dir: Path | str,
    *,
    num: dict[str, int],
    seed: int,
    image_size: tuple[int, int],
) -> None:
    """Generate the synthetic dataset deterministically.

    Writes, under ``out_dir``:
      * ``<split>/<file>.jpg``                — one image per item, per split.
      * ``<split>/_annotations.coco.json``    — Roboflow-shaped COCO per split.
      * ``../nomenclature.csv``               — rows for every drawn code
        (``configs/data.yaml`` points the nomenclature CSV one level above the
        dataset root, at ``fixtures/nomenclature.csv``).
      * ``ground_truth.json``                 — per-image expected values + boxes.

    Args:
        out_dir: dataset root (e.g. ``fixtures/synthetic``).
        num: images per split, keyed by split *directory* name (train/valid/test).
        seed: master RNG seed (fully determines the output).
        image_size: ``(width, height)`` in pixels.

    Raises:
        ValueError: if ``num`` is empty or any count is negative.
    """
    out_dir = Path(out_dir)
    if not num or any(int(v) < 0 for v in num.values()):
        raise ValueError(f"`num` must be a non-empty map of non-negative counts, got {num!r}")

    schema = get_classes()
    class_names = set(schema.names)
    letters = _letters()
    columns = _nomenclature_columns()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    out_dir.mkdir(parents=True, exist_ok=True)

    ground_truth: dict[str, dict[str, Any]] = {}
    # num_enregistrement -> identity row, deduped across all splits/images.
    nomenclature_rows: dict[str, dict[str, str]] = {}
    # Global RNG seeded once -> deterministic across the full run (no leakage by
    # construction: each image gets a unique global index baked into its name).
    rng = random.Random(seed)
    global_index = 0

    for split, count in num.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        coco = _empty_coco(now_iso)
        ann_id = 1  # real export starts annotation ids at 1

        for local_i in range(int(count)):
            # ---- synthesize this vignette's field values --------------------
            product_name, dci, dosage, forme, laboratoire = _PRODUCTS[rng.randrange(len(_PRODUCTS))]
            num_enr = _make_enregistrement(rng, letters)
            num_lot = _make_num_lot(rng)
            date_fab, date_exp = _make_dates(rng)
            prix, shp, ppa = _make_money(rng)
            color = rng.choice(list(_BAND_RGB.keys()))

            fields: dict[str, str] = {
                "ppa": money_str(ppa),
                "prix": money_str(prix),
                "shp": money_str(shp),
                "num_enregistrement": num_enr,
                "num_lot": num_lot,
                "date_fab": date_fab,
                "date_exp": date_exp,
                "product_name": product_name,
                "dci": dci,
                "dosage": dosage,
                "forme": forme,
                "laboratoire": laboratoire,
            }

            # ---- render + collect boxes ------------------------------------
            img, boxes = _render_image(rng, image_size, fields, color)
            file_name = f"synthetic_{split}_{global_index:05d}.jpg"
            img.save(split_dir / file_name, format="JPEG", quality=92)

            # ---- COCO image + annotations ----------------------------------
            image_id = local_i
            coco["images"].append(
                {
                    "id": image_id,
                    "license": 1,
                    "file_name": file_name,
                    "height": int(image_size[1]),
                    "width": int(image_size[0]),
                    "date_captured": now_iso,
                }
            )
            for class_name, (bx, by, bw, bh) in boxes.items():
                if class_name not in class_names:  # defensive; drawn names are schema names
                    continue
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": schema.id_of(class_name),
                        "bbox": [float(bx), float(by), float(bw), float(bh)],
                        "area": float(bw) * float(bh),
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            # ---- ground truth + nomenclature row ---------------------------
            ground_truth[file_name] = {
                "fields": dict(fields),
                "reimbursability": color,
                "boxes": {name: [int(v) for v in box] for name, box in boxes.items()},
            }
            row = {
                "num_enregistrement": num_enr,
                "product_name": product_name,
                "dci": dci,
                "dosage": dosage,
                "forme": forme,
                "laboratoire": laboratoire,
                "tr": f"{rng.randint(0, 100)}",  # taux de remboursement (fixture value)
            }
            # Keep the first row per code stable (codes may recur across images).
            nomenclature_rows.setdefault(num_enr, {k: row.get(k, "") for k in columns})

            global_index += 1

        with open(split_dir / "_annotations.coco.json", "w", encoding="utf-8") as fh:
            json.dump(coco, fh, ensure_ascii=False, indent=2)
        log.info(
            "synthetic_split_written",
            split=split,
            images=len(coco["images"]),
            annotations=len(coco["annotations"]),
        )

    # ---- nomenclature CSV (one level above the dataset root) ----------------
    nomenclature_path = out_dir.parent / "nomenclature.csv"
    nomenclature_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nomenclature_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for code in sorted(nomenclature_rows):
            writer.writerow(nomenclature_rows[code])

    # ---- ground truth -------------------------------------------------------
    with open(out_dir / GROUND_TRUTH_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, ensure_ascii=False, indent=2)

    log.info(
        "synthetic_generated",
        out_dir=str(out_dir),
        images=len(ground_truth),
        codes=len(nomenclature_rows),
        nomenclature_csv=str(nomenclature_path),
    )


def load_ground_truth(root: Path | str) -> dict[str, dict[str, Any]]:
    """Load ``ground_truth.json`` from a synthetic dataset ``root``.

    Returns a map keyed by image ``file_name``. Raises ``FileNotFoundError`` if
    the fixture has not been generated yet.
    """
    gt_path = Path(root) / GROUND_TRUTH_FILENAME
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_FILENAME} not found under {root!r}; "
            f"run `python -m vignocr.data.synthetic` or vignocr.data.synthetic.generate(...) first."
        )
    with open(gt_path, encoding="utf-8") as fh:
        return json.load(fh)


def _generate_from_config() -> Path:
    """Regenerate the active synthetic dataset from ``configs/data.yaml``."""
    from vignocr.common import get_active_dataset, seed_everything

    ds = get_active_dataset()
    if ds.get("name") != "synthetic":
        raise SystemExit(
            f"active dataset is {ds.get('name')!r}, not 'synthetic'; "
            f"set VIGNOCR_DATA_ACTIVE=synthetic to generate the fixture."
        )
    seed = int(ds.get("seed", 1337))
    seed_everything(seed)
    root = Path(ds["root"])
    splits: dict[str, str] = ds["splits"]  # {train: train, val: valid, test: test}
    num_images: dict[str, int] = ds["num_images"]  # keyed by logical split (train/val/test)
    # Map logical split -> directory name; `num` is keyed by directory name.
    num = {splits[logical]: int(count) for logical, count in num_images.items()}
    size = tuple(int(v) for v in ds["image_size"])
    generate(root, num=num, seed=seed, image_size=(size[0], size[1]))
    return root


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    out = _generate_from_config()
    print(f"synthetic fixture written to {out}")
