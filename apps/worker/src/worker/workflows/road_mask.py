import base64
import io
import json
import re
from datetime import timedelta

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.conversational import (
    FileField,
    FileWithMetadataValue,
    FormInput,
    SingleChoice,
    TextField,
)

SEGMENTATION_MODEL = "mistral-large-latest"
MAX_MODEL_IMAGE_SIDE = 1540
MAX_PNG_BASE64_TOTAL = 1_500_000
DEFAULT_TARGET = (
    "the drivable road surface (asphalt), including lane markings painted on it; "
    "exclude sidewalks, curbs, grass, gravel, buildings, vehicles, pedestrians, poles and sky"
)
GRID_CHOICES = [
    ("64x36", "64 x 36 (16:9, recommended)"),
    ("48x48", "48 x 48 (square)"),
    ("96x54", "96 x 54 (fine detail)"),
]
REFINE_CHOICES = [
    ("1", "1 refinement pass (recommended)"),
    ("0", "No refinement"),
    ("2", "2 refinement passes (slower)"),
]


class ImagePayload(BaseModel):
    mime: str
    data_base64: str
    width: int
    height: int


class GridMask(BaseModel):
    cols: int
    rows: int
    cells: list[list[int]]


class MaskResult(BaseModel):
    filename: str
    width: int
    height: int
    coverage: float
    grid: GridMask
    png_base64: str
    data_url: str
    file_url: str | None = None


class RoadMaskRequest(BaseModel):
    images: list[str] = Field(
        description="Simulator screenshots as HTTPS URLs or data: URLs (base64)."
    )
    target: str = Field(default=DEFAULT_TARGET, description="What should be white in the mask.")
    grid_cols: int = Field(default=64, ge=8, le=256)
    grid_rows: int = Field(default=36, ge=8, le=256)
    refine_passes: int = Field(default=1, ge=0, le=3)


class MaskUploadForm(FormInput):
    images: list[FileWithMetadataValue] = FileField(
        description="Paste or upload simulator screenshots",
        multiple=True,
        include_metadata=True,
    )
    target: str = TextField(
        description="What should be white in the mask",
        prefilled_value=DEFAULT_TARGET,
    )
    grid_size: str = SingleChoice(
        options=GRID_CHOICES,
        description="Mask grid resolution",
        prefilled_value="64x36",
    )
    refine_passes: str = SingleChoice(
        options=REFINE_CHOICES,
        description="Extra vision-model passes to clean up edges",
        prefilled_value="1",
    )


def build_segmentation_prompt(target: str, cols: int, rows: int) -> str:
    return f"""You are segmenting a driving-simulator screenshot into a binary mask.

Divide the image into a uniform grid of {cols} columns x {rows} rows (row 0 is the TOP row, column 0 is the LEFTMOST column). For each cell output 1 if the target covers the MAJORITY of that cell's area, otherwise 0.

Target to segment: {target}

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"cols": {cols}, "rows": {rows}, "cells": <{rows} arrays of {cols} integers, each 0 or 1>}}

Rules:
- cells must contain exactly {rows} rows, each with exactly {cols} values, each value only 0 or 1.
- Lane markings and road paint count as part of the road.
- Sidewalks, curbs, grass, gravel, buildings, fences, vehicles, pedestrians, poles, shadows and sky do NOT count.
- If a cell is ambiguous, output 0."""


def build_refine_prompt(target: str, cols: int, rows: int) -> str:
    return f"""The image is a driving-simulator screenshot with a semi-transparent RED overlay marking where a previous pass located the target.

Target: {target}

Review the overlay critically: a cell marked 1 (red) must show the target in the majority of its area, a cell marked 0 must not. Remove red from cells that are sidewalk, grass, gravel, buildings, vehicles, sky, shadows, etc., and add red to cells that are road but were missed.

Respond with ONLY the corrected JSON object, no prose, no markdown fences:
{{"cols": {cols}, "rows": {rows}, "cells": <{rows} arrays of {cols} integers, each 0 or 1>}}"""


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
    return str(content)


def parse_grid(text: str, cols: int, rows: int) -> GridMask:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model returned no JSON object: {text[:200]}")
    data = json.loads(match.group(0))
    cells = data.get("cells")
    if not isinstance(cells, list) or len(cells) != rows:
        raise ValueError(f"Expected {rows} rows in cells, got {len(cells) if isinstance(cells, list) else type(cells)}")
    normalized: list[list[int]] = []
    for row in cells:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"Expected {cols} columns per row, got {len(row) if isinstance(row, list) else type(row)}")
        normalized.append([1 if int(v) != 0 else 0 for v in row])
    return GridMask(cols=cols, rows=rows, cells=normalized)


def grid_to_png(grid: GridMask, width: int, height: int) -> bytes:
    from PIL import Image

    img = Image.new("L", (grid.cols, grid.rows), 0)
    px = img.load()
    for r, row in enumerate(grid.cells):
        for c, value in enumerate(row):
            px[c, r] = 255 if value else 0
    if (width, height) != (grid.cols, grid.rows):
        img = img.resize((width, height), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def grid_overlay_png(image: ImagePayload, grid: GridMask) -> bytes:
    from PIL import Image

    base = Image.open(io.BytesIO(base64.b64decode(image.data_base64))).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    px = overlay.load()
    cell_w = base.size[0] / grid.cols
    cell_h = base.size[1] / grid.rows
    for r, row in enumerate(grid.cells):
        for c, value in enumerate(row):
            if value:
                x0, y0 = round(c * cell_w), round(r * cell_h)
                x1, y1 = round((c + 1) * cell_w), round((r + 1) * cell_h)
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        px[x, y] = (255, 0, 0, 110)
    combined = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def fetch_image(ref: str) -> ImagePayload:
    from PIL import Image

    if ref.startswith("data:"):
        header, b64 = ref.split(",", 1)
        mime = header[5:].split(";")[0] or "image/png"
        b64 = re.sub(r"\s+", "", b64.replace(" ", "+"))
        b64 += "=" * (-len(b64) % 4)
        raw = base64.b64decode(b64)
    else:
        import httpx

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
            response = await http.get(ref)
            response.raise_for_status()
            raw = response.content
            mime = response.headers.get("content-type", "image/png").split(";")[0]
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        import hashlib

        raise ValueError(
            f"Undecodable image data: b64_len={len(b64)} raw_len={len(raw)} "
            f"raw_sha256={hashlib.sha256(raw).hexdigest()[:16]} head={raw[:16].hex()} ({exc})"
        ) from exc
    width, height = img.size
    img = img.convert("RGB")
    if max(width, height) > MAX_MODEL_IMAGE_SIDE:
        scale = MAX_MODEL_IMAGE_SIDE / max(width, height)
        img = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return ImagePayload(
        mime="image/jpeg",
        data_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
        width=width,
        height=height,
    )


@workflows.activity(start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=3)
async def segment_image(
    image: ImagePayload,
    target: str,
    cols: int,
    rows: int,
    client=Depends(get_mistral_client),
) -> GridMask:
    response = await client.chat.complete_async(
        model=SEGMENTATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_segmentation_prompt(target, cols, rows)},
                    {"type": "image_url", "image_url": f"data:{image.mime};base64,{image.data_base64}"},
                ],
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return parse_grid(_message_text(response.choices[0].message.content), cols, rows)


@workflows.activity(start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=3)
async def refine_mask(
    image: ImagePayload,
    grid: GridMask,
    target: str,
    client=Depends(get_mistral_client),
) -> GridMask:
    overlay_b64 = base64.b64encode(grid_overlay_png(image, grid)).decode("ascii")
    response = await client.chat.complete_async(
        model=SEGMENTATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_refine_prompt(target, grid.cols, grid.rows)},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{overlay_b64}"},
                ],
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return parse_grid(_message_text(response.choices[0].message.content), grid.cols, grid.rows)


@workflows.activity(start_to_close_timeout=timedelta(minutes=1))
async def render_mask(grid: GridMask, width: int, height: int, filename: str) -> MaskResult:
    png = grid_to_png(grid, width, height)
    b64 = base64.b64encode(png).decode("ascii")
    coverage = sum(sum(row) for row in grid.cells) / float(grid.cols * grid.rows)
    return MaskResult(
        filename=filename,
        width=width,
        height=height,
        coverage=round(coverage, 4),
        grid=grid,
        png_base64=b64,
        data_url=f"data:image/png;base64,{b64}",
    )


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def upload_mask_png(
    filename: str,
    png_base64: str,
    client=Depends(get_mistral_client),
) -> str | None:
    try:
        uploaded = await client.files.upload_async(
            file={"file_name": filename, "content": base64.b64decode(png_base64)},
        )
        signed = await client.files.get_signed_url_async(file_id=uploaded.id, expiry=24 * 7)
        return signed.url
    except Exception:
        return None


@workflows.workflow.define(
    name="road-mask-segmenter",
    workflow_display_name="Road Mask Segmenter",
    workflow_description=(
        "Paste simulator screenshots and get binary road masks back "
        "(road = white 255, everything else = black 0), segmented by Mistral Large."
    ),
)
class RoadMaskWorkflow(workflows.InteractiveWorkflow):
    @workflows.workflow.entrypoint
    async def run(self, request: RoadMaskRequest | None = None) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        if request is not None and request.images:
            refs = request.images
            names = [f"image-{i + 1}.png" for i in range(len(refs))]
            target = request.target
            cols, rows = request.grid_cols, request.grid_rows
            refine_passes = request.refine_passes
        else:
            await workflows_mistralai.send_assistant_message(
                "Paste or upload simulator screenshots below and I will return a binary mask for each "
                "(road = white, everything else = black)."
            )
            form = await self.wait_for_input(MaskUploadForm, label="Upload screenshots")
            refs = [f.url for f in form.images]
            names = [f.filename or f"image-{i + 1}.png" for i, f in enumerate(form.images)]
            target = form.target
            cols, rows = (int(part) for part in form.grid_size.split("x"))
            refine_passes = int(form.refine_passes)

        results: list[MaskResult] = []
        for i, ref in enumerate(refs):
            await workflows_mistralai.send_assistant_message(
                f"Segmenting {names[i]} ({i + 1}/{len(refs)}) with {SEGMENTATION_MODEL}..."
            )
            image = await fetch_image(ref)
            grid = await segment_image(image, target, cols, rows)
            for _ in range(refine_passes):
                grid = await refine_mask(image, grid, target)
            mask = await render_mask(grid, image.width, image.height, names[i])
            mask.file_url = await upload_mask_png(mask.filename, mask.png_base64)
            results.append(mask)

        include_png = sum(len(m.png_base64) for m in results) <= MAX_PNG_BASE64_TOTAL
        masks_payload = [
            {
                "filename": m.filename,
                "width": m.width,
                "height": m.height,
                "coverage": m.coverage,
                "grid": m.grid.model_dump(),
                "png_base64": m.png_base64 if include_png else None,
                "data_url": m.data_url if include_png else None,
                "file_url": m.file_url,
            }
            for m in results
        ]
        summary_lines = [
            f"Generated {len(results)} mask(s) with {SEGMENTATION_MODEL} "
            f"(grid {cols}x{rows}, {refine_passes} refinement pass(es)):"
        ]
        for m in results:
            line = f"- {m.filename}: {m.width}x{m.height}, road coverage {m.coverage:.1%}"
            if m.file_url:
                line += f", download: {m.file_url}"
            summary_lines.append(line)
        if not include_png:
            summary_lines.append("Output too large for inline base64; use the download file_url(s) or run fewer images per execution.")

        content: list[workflows_mistralai.TextOutput | workflows_mistralai.ResourceOutput] = [
            workflows_mistralai.TextOutput(text="\n".join(summary_lines))
        ]
        for m in results:
            if include_png:
                content.append(
                    workflows_mistralai.ResourceOutput(
                        resource=workflows_mistralai.CanvasResource(
                            canvas=workflows_mistralai.CanvasPayload(
                                type="text/markdown",
                                title=f"Mask: {m.filename}",
                                content=(
                                    f"![road mask {m.filename}]({m.data_url})\n\n"
                                    f"- Size: {m.width}x{m.height}\n"
                                    f"- Road coverage: {m.coverage:.1%}"
                                ),
                            )
                        )
                    )
                )
        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=content,
            structuredContent={"masks": masks_payload},
        )
