"""COCO loading + per-field cropping (Pillow only — no ML libs).

The category->class mapping is built **by name** from each file's own
``categories`` array (``category_id`` -> ``name``), never from a hardcoded id, so
this stays correct whatever ids the real Roboflow export assigns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vignocr.common import BBox, get_logger

if TYPE_CHECKING:  # avoid importing PIL types at module import time for typing only
    from PIL import Image as PILImage

log = get_logger(__name__)

COCO_FILENAME = "_annotations.coco.json"


@dataclass(frozen=True)
class Crop:
    """A cropped field region: its box (in input-image pixels) plus the image."""

    name: str
    bbox: BBox
    image: PILImage.Image


@dataclass
class CocoSplit:
    """One loaded COCO split.

    Attributes:
        root: dataset root the split was loaded from.
        split: directory name of the split (e.g. ``train``).
        images: raw COCO ``images`` list.
        annotations: raw COCO ``annotations`` list.
        categories: raw COCO ``categories`` list (the file's own).
        cat_id_to_name: ``category_id`` -> class name, from this file's categories.
        cat_name_to_id: class name -> ``category_id`` (inverse).
        split_dir: absolute directory containing the images + COCO json.
    """

    root: Path
    split: str
    images: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    cat_id_to_name: dict[int, str]
    cat_name_to_id: dict[str, int]
    split_dir: Path
    _by_image: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    def annotations_for(self, image_id: int) -> list[dict[str, Any]]:
        """Annotations belonging to ``image_id`` (empty list if none)."""
        return self._by_image.get(image_id, [])

    def image_path(self, image: dict[str, Any]) -> Path:
        """Absolute path to an image record's file on disk."""
        return self.split_dir / image["file_name"]

    def class_name(self, annotation: dict[str, Any]) -> str:
        """Class name of an annotation via this file's category map (by name)."""
        return self.cat_id_to_name[int(annotation["category_id"])]


def _resolve_split_dir(root: Path, split: str) -> Path:
    """Resolve the directory holding ``split``'s images + COCO json.

    Accepts either a split *directory name* (``train``) or a logical split key
    (``val``) by consulting ``configs/data.yaml``'s ``splits`` map for the active
    dataset. A directory that exists on disk wins immediately.
    """
    direct = root / split
    if direct.is_dir():
        return direct
    # Map logical split -> directory name via config (e.g. val -> valid).
    try:
        from vignocr.common import get_active_dataset

        splits = get_active_dataset().get("splits", {})
        mapped = splits.get(split)
        if mapped and (root / mapped).is_dir():
            return root / mapped
    except Exception:  # config is best-effort here; fall through to the direct path
        log.debug("split_dir_config_lookup_failed", split=split)
    return direct


def load_split(root: Path | str, split: str) -> CocoSplit:
    """Load one COCO split from ``<root>/<split>/_annotations.coco.json``.

    The category map is derived from the file's own ``categories`` array.

    Raises:
        FileNotFoundError: if the split's COCO json is missing.
    """
    import json

    root = Path(root)
    split_dir = _resolve_split_dir(root, split)
    coco_path = split_dir / COCO_FILENAME
    if not coco_path.exists():
        raise FileNotFoundError(
            f"COCO file not found: {coco_path}. "
            f"For the synthetic fixture, run `python -m vignocr.data.synthetic` first."
        )

    with open(coco_path, encoding="utf-8") as fh:
        data = json.load(fh)

    categories = data.get("categories", [])
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in categories}
    cat_name_to_id = {str(c["name"]): int(c["id"]) for c in categories}

    annotations = data.get("annotations", [])
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        by_image[int(ann["image_id"])].append(ann)

    split_obj = CocoSplit(
        root=root,
        split=split,
        images=data.get("images", []),
        annotations=annotations,
        categories=categories,
        cat_id_to_name=cat_id_to_name,
        cat_name_to_id=cat_name_to_id,
        split_dir=split_dir,
        _by_image=dict(by_image),
    )
    log.debug(
        "coco_split_loaded",
        split=split,
        images=len(split_obj.images),
        annotations=len(annotations),
        categories=len(categories),
    )
    return split_obj


def crops_for_image(
    img_path: Path | str,
    anns: list[dict[str, Any]],
    schema: Any,
) -> dict[str, list[Crop]]:
    """Crop every annotation on one image, grouped by field (class) name.

    Args:
        img_path: path to the source image.
        anns: COCO annotations for that image (each carries ``category_id`` +
            ``bbox`` in ``[x, y, w, h]``).
        schema: a category map. Either a :class:`vignocr.common.ClassSchema`
            (mapping training ids -> names) or a plain ``{category_id: name}``
            dict (e.g. ``CocoSplit.cat_id_to_name``) — whichever the caller has.

    Returns:
        ``{field_name: [Crop, ...]}``. Annotations whose ``category_id`` is not
        resolvable by ``schema`` are skipped (logged at debug level).

    Raises:
        ImportError: if Pillow is unavailable (it is a core dep — should not
            happen on a correctly installed environment).
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a core dependency
        raise ImportError(
            "Pillow is required to crop images. Install the core package: pip install -e ."
        ) from exc

    name_of = _category_resolver(schema)

    out: dict[str, list[Crop]] = defaultdict(list)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        iw, ih = im.size
        for ann in anns:
            cid = int(ann["category_id"])
            name = name_of(cid)
            if name is None:
                log.debug("crop_unknown_category", category_id=cid, img=str(img_path))
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            # Clamp to image bounds so a slightly out-of-frame box still crops.
            x0 = max(0, int(round(x)))
            y0 = max(0, int(round(y)))
            x1 = min(iw, int(round(x + w)))
            y1 = min(ih, int(round(y + h)))
            if x1 <= x0 or y1 <= y0:
                log.debug("crop_degenerate_box", name=name, bbox=ann["bbox"], img=str(img_path))
                continue
            region = im.crop((x0, y0, x1, y1)).copy()
            out[name].append(
                Crop(
                    name=name,
                    bbox=BBox(x=x, y=y, w=w, h=h),
                    image=region,
                )
            )
    return dict(out)


def _category_resolver(schema: Any):
    """Return a ``category_id -> name | None`` callable for the given schema.

    Supports a :class:`ClassSchema` (via ``name_of``) or a plain id->name dict.
    """
    name_of = getattr(schema, "name_of", None)
    if callable(name_of):

        def _resolve(cid: int) -> str | None:
            try:
                return name_of(cid)
            except KeyError:
                return None

        return _resolve

    if isinstance(schema, dict):
        return lambda cid: schema.get(int(cid))

    raise TypeError(
        "schema must be a ClassSchema (with .name_of) or a {category_id: name} dict, "
        f"got {type(schema).__name__}"
    )
