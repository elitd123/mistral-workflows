#!/usr/bin/env python3
"""Batch-generate binary road masks for a folder of simulator screenshots.

Calls the deployed `road-mask-segmenter` workflow in batches (the platform caps
workflow input at 2MB), and writes one `<name>_mask.png` per screenshot at the
ORIGINAL image resolution (mask is upscaled back with nearest-neighbor, so it
stays a clean binary 0/255 image aligned with your training input).

Usage:
    pip install mistralai pillow
    export MISTRAL_API_KEY="your-workspace-api-key"
    python scripts/batch_mask.py --input screenshots/ --output masks/
    python scripts/batch_mask.py --input screenshots/ --output masks/ --grid 48x27 --refine 0   # cheaper bulk run
    python scripts/batch_mask.py --input screenshots/ --dry-run    # just show the batch plan
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

from PIL import Image

WORKFLOW_NAME = "road-mask-segmenter"
MAX_INPUT_BYTES = 1_800_000
MAX_MODEL_SIDE = 1540
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_TARGET = (
    "the drivable road surface (asphalt), including lane markings painted on it; "
    "exclude sidewalks, curbs, grass, gravel, buildings, vehicles, pedestrians, poles and sky"
)


def encode_image(path: Path, max_side: int = MAX_MODEL_SIDE, quality: int = 85) -> tuple[str, int, int]:
    img = Image.open(path)
    img.load()
    orig_w, orig_h = img.size
    img = img.convert("RGB")
    if max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        img = img.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}", orig_w, orig_h


def pack_batches(items: list[dict], max_input_bytes: int = MAX_INPUT_BYTES) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    for item in items:
        size = len(item["data_url"])
        if current and current_size + size > max_input_bytes:
            batches.append(current)
            current, current_size = [], 0
        current.append(item)
        current_size += size
    if current:
        batches.append(current)
    return batches


def result_masks(execution) -> list[dict]:
    result = execution.result
    structured = result["structuredContent"] if isinstance(result, dict) else result.structuredContent
    masks = structured["masks"] if isinstance(structured, dict) else structured.masks
    return [m if isinstance(m, dict) else m.model_dump() for m in masks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Folder with simulator screenshots")
    parser.add_argument("--output", type=Path, default=None, help="Output folder for masks (default: <input>/masks)")
    parser.add_argument("--grid", default="96x54", help="Mask grid COLSxROWS, e.g. 96x54 (default, smooth edges), 64x36, 48x27")
    parser.add_argument("--refine", type=int, default=1, help="Refinement passes per image: 0, 1 (default), 2")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="What should be white in the mask")
    parser.add_argument("--dry-run", action="store_true", help="Only print the batch plan, no API calls")
    args = parser.parse_args()

    output_dir = args.output or (args.input / "masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    cols, rows = (int(p) for p in args.grid.lower().split("x"))

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    todo = [p for p in files if not (output_dir / f"{p.stem}_mask.png").exists()]
    print(f"{len(files)} image(s) found, {len(todo)} to process (already-done masks are skipped)")

    items = []
    for path in todo:
        data_url, w, h = encode_image(path)
        items.append({"path": path, "data_url": data_url, "w": w, "h": h})
    batches = pack_batches(items)
    print(f"batch plan: {len(batches)} execution(s), sizes="
          f"{[sum(len(i['data_url']) for i in b) for b in batches]} bytes")

    if args.dry_run:
        for i, b in enumerate(batches):
            print(f"  batch {i + 1}: {[str(x['path'].name) for x in b]}")
        return 0

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("MISTRAL_API_KEY is not set", file=sys.stderr)
        return 2
    from mistralai.client import Mistral

    client = Mistral(api_key=api_key)
    failures = 0
    for bi, batch in enumerate(batches):
        payload = {
            "images": [it["data_url"] for it in batch],
            "grid_cols": cols,
            "grid_rows": rows,
            "refine_passes": args.refine,
            "target": args.target,
        }
        masks = None
        for attempt in range(3):
            try:
                execution = client.workflows.execute_workflow(
                    workflow_identifier=WORKFLOW_NAME, input=payload, wait_for_result=True
                )
                masks = result_masks(execution)
                break
            except Exception as exc:
                print(f"batch {bi + 1}/{len(batches)} attempt {attempt + 1} failed: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        if masks is None:
            failures += len(batch)
            print(f"batch {bi + 1}/{len(batches)} FAILED permanently: {[str(x['path'].name) for x in batch]}",
                  file=sys.stderr)
            continue
        for item, mask in zip(batch, masks):
            png = base64.b64decode(mask["png_base64"])
            img = Image.open(io.BytesIO(png)).convert("L")
            if img.size != (item["w"], item["h"]):
                img = img.resize((item["w"], item["h"]), Image.NEAREST)
            out = output_dir / f"{item['path'].stem}_mask.png"
            img.save(out)
            print(f"[{bi + 1}/{len(batches)}] {item['path'].name} -> {out} "
                  f"({img.size[0]}x{img.size[1]}, coverage {mask['coverage']:.1%})")
    print(f"done. {len(todo) - failures} mask(s) written to {output_dir}, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
