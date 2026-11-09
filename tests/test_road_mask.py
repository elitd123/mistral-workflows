import base64
import io
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.core.dependencies.dependency_injector import DependencyInjector
from mistralai.workflows.testing import create_test_worker_with_events
from mistralai.workflows.testing.fixtures import temporal_env_with_converter  # noqa: F401

from worker.workflows.road_mask import (
    GRID_CHOICES,
    MaskUploadForm,
    RoadMaskWorkflow,
    build_segmentation_prompt,
    fetch_image,
    grid_overlay_png,
    grid_to_png,
    parse_grid,
    refine_mask,
    render_mask,
    segment_image,
    upload_mask_png,
)
from worker.workflows.road_mask import GridMask, ImagePayload

pytest_plugins = ["mistralai.workflows.testing.fixtures"]

TASK_QUEUE = "test-task-queue"


def make_road_image_data_url(width: int = 64, height: int = 36) -> str:
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            if y >= height // 2:
                px[x, y] = (90, 90, 90)
            else:
                px[x, y] = (30, 120, 30)
    for y in range(height // 2, height):
        px[width // 2, y] = (240, 240, 240)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def bottom_half_grid(cols: int, rows: int) -> str:
    cells = [[1 if r >= rows // 2 else 0 for _ in range(cols)] for r in range(rows)]
    return json.dumps({"cols": cols, "rows": rows, "cells": cells})


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_async(self, **kwargs):
        self.calls.append(kwargs)
        content = bottom_half_grid(64, 36)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeFiles:
    async def upload_async(self, *, file):
        return SimpleNamespace(id="file-test-123")

    async def get_signed_url_async(self, *, file_id, expiry=24):
        return SimpleNamespace(url=f"https://files.example.com/{file_id}?sig=abc")


class FakeMistralClient:
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.files = FakeFiles()


@pytest.fixture
def fake_mistral_client():
    client = FakeMistralClient()
    injector = DependencyInjector.get_singleton_instance()
    snapshot = DependencyInjector.snapshot_dependencies()

    @asynccontextmanager
    async def fake_client_ctx():
        yield client

    injector._dependencies_ctx[get_mistral_client] = fake_client_ctx
    yield client
    DependencyInjector.restore_dependencies(snapshot)


def test_parse_grid_valid():
    grid = parse_grid(bottom_half_grid(4, 2), 4, 2)
    assert grid.cols == 4 and grid.rows == 2
    assert grid.cells == [[0, 0, 0, 0], [1, 1, 1, 1]]


def test_parse_grid_extracts_json_from_prose():
    text = f"Sure! Here you go:\n{bottom_half_grid(2, 2)}\nHope that helps."
    grid = parse_grid(text, 2, 2)
    assert grid.cells == [[0, 0], [1, 1]]


def test_parse_grid_rejects_wrong_shape():
    with pytest.raises(ValueError):
        parse_grid(json.dumps({"cols": 4, "rows": 4, "cells": [[0, 0], [1, 1]]}), 4, 2)
    with pytest.raises(ValueError):
        parse_grid("no json at all", 4, 2)


def test_grid_to_png_binary_and_sized():
    grid = GridMask(cols=2, rows=2, cells=[[1, 0], [0, 1]])
    png = grid_to_png(grid, 8, 4)
    img = Image.open(io.BytesIO(png))
    assert img.size == (8, 4)
    assert img.mode == "L"
    px = img.load()
    assert px[0, 0] == 255 and px[7, 0] == 0
    assert px[7, 3] == 255 and px[0, 3] == 0


def test_grid_overlay_marks_masked_cells():
    image = ImagePayload(
        mime="image/jpeg",
        data_base64=base64.b64encode(_png_bytes()).decode("ascii"),
        width=4,
        height=4,
    )
    grid = GridMask(cols=2, rows=2, cells=[[1, 0], [0, 0]])
    overlay = Image.open(io.BytesIO(grid_overlay_png(image, grid))).convert("RGB")
    px = overlay.load()
    r, g, b = px[0, 0]
    assert r - max(g, b) > 40
    r, g, b = px[3, 3]
    assert r - max(g, b) < 30


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (50, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


def test_segmentation_prompt_mentions_grid_and_target():
    prompt = build_segmentation_prompt("the road", 64, 36)
    assert "64 columns x 36 rows" in prompt
    assert "the road" in prompt
    assert "0 or 1" in prompt


def test_form_schema_has_file_field():
    schema = MaskUploadForm.model_json_schema()
    props = schema["properties"]
    assert props["images"]["type"] == "array"
    grid_consts = [c["const"] for c in props["grid_size"]["oneOf"]]
    assert grid_consts == [c[0] for c in GRID_CHOICES]


@pytest.mark.asyncio
async def test_fetch_image_from_data_url(temporal_env_with_converter):  # noqa: F811
    payload = await fetch_image(make_road_image_data_url(64, 36))
    assert payload.width == 64 and payload.height == 36
    assert payload.mime == "image/jpeg"
    img = Image.open(io.BytesIO(base64.b64decode(payload.data_base64)))
    assert img.size == (64, 36)


@pytest.mark.asyncio
async def test_workflow_end_to_end(temporal_env_with_converter, fake_mistral_client):  # noqa: F811
    async with create_test_worker_with_events(
        temporal_env_with_converter,
        [RoadMaskWorkflow],
        activities=[fetch_image, segment_image, refine_mask, render_mask, upload_mask_png],
        task_queue=TASK_QUEUE,
    ):
        started = await temporal_env_with_converter.client.execute_workflow(
            "road-mask-segmenter",
            {"images": [make_road_image_data_url()], "grid_cols": 64, "grid_rows": 36, "refine_passes": 1},
            id="test-road-mask-1",
            task_queue=TASK_QUEUE,
        )
        result = started if isinstance(started, dict) else await started.result()

    assert fake_mistral_client.chat.calls, "vision model was not called"
    first_call = fake_mistral_client.chat.calls[0]
    assert first_call["model"] == "mistral-large-latest"
    user_content = first_call["messages"][0]["content"]
    assert any(part.get("type") == "image_url" and part["image_url"].startswith("data:image/") for part in user_content)

    structured = result["structuredContent"] if isinstance(result, dict) else result.structuredContent
    masks = structured["masks"] if isinstance(structured, dict) else structured["masks"]
    assert len(masks) == 1
    mask = masks[0]
    mask = mask if isinstance(mask, dict) else mask.model_dump()
    assert mask["width"] == 64 and mask["height"] == 36
    assert mask["coverage"] == 0.5
    assert mask["file_url"] == "https://files.example.com/file-test-123?sig=abc"
    png = base64.b64decode(mask["png_base64"])
    img = Image.open(io.BytesIO(png))
    assert img.size == (64, 36)
    px = img.convert("L").load()
    assert px[32, 35] == 255 and px[32, 0] == 0
