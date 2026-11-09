# road-mask-workflow

A [Mistral Workflow](https://docs.mistral.ai/studio/workflows/overview) that turns raw
driving-simulator screenshots into binary training masks: **road = white (255),
everything else = black (0)**. Segmentation runs on `mistral-large-latest`
(Mistral Large 3), Mistral's flagship vision model, which emits a grid mask that
the workflow rasterizes into a pixel-aligned PNG.

## Use it

### In Vibe Work (paste images, get masks)

Publish the workflow as a chat assistant (it returns a
`ChatAssistantWorkflowOutput`), attach it from the `+` menu, paste or upload
screenshots in the upload form, and each mask comes back inline plus as a
downloadable file URL. See
[Conversational Workflows](https://docs.mistral.ai/studio/workflows/interacting-with-workflows/conversational_workflows).

### Via the API (batch)

```bash
curl -X POST https://api.mistral.ai/v1/workflows/road-mask-segmenter/execute \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "images": ["https://example.com/frame_001.png", "data:image/png;base64,..."],
      "target": "the drivable road surface (asphalt), including lane markings painted on it; exclude sidewalks, curbs, grass, gravel, buildings, vehicles, pedestrians, poles and sky",
      "grid_cols": 64,
      "grid_rows": 36,
      "refine_passes": 1
    },
    "wait_for_result": true
  }'
```

The result's `structuredContent.masks[]` contains, per image: `filename`,
`width`, `height`, `coverage`, the raw `grid`, the mask as `png_base64` /
`data_url`, and a `file_url` (signed URL of the uploaded PNG, valid 7 days).

- `grid_cols` x `grid_rows` controls mask precision (higher = finer edges, more
  tokens). The PNG mask is rendered at the original image resolution with
  nearest-neighbor upsampling, so it stays a clean binary mask.
- `refine_passes` re-sends the image with the current mask overlaid in red and
  asks the model to correct it; useful for cleaning up edges.
- Workflow/activity payloads are capped at 2MB by the platform; if a batch's
  inline base64 masks would exceed that, the response falls back to `file_url`
  only. Run fewer images per execution in that case.

## Develop

```bash
uv sync
cp .env.example .env  # set MISTRAL_API_KEY
PYTHONPATH=apps/worker/src uv run python -m worker.entrypoints.worker   # run the worker locally
uv run pytest                                                        # run the tests
```

## Deploy

Managed deployments build from a GitHub repository; the image is built from
`deploy/docker/Dockerfile.worker` (build context: repository root). See
[Quickstart: deploy a workflow to Mistral](https://docs.mistral.ai/studio/workflows/managing-workflows-in-production/managed-deployments/getting-started).
