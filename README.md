# motionmcp

Python SDK for **MMCP** — the [Motion Model Context Protocol][docs].

Build a vendor-neutral motion-generation HTTP server in ~30 lines: the
SDK handles wire-format validation, glTF encoding, and the standard
error envelope; you implement model loading and a single `generate()`
method.

[docs]: https://animatica.ai/mmcp

## Where this fits

MMCP is a small HTTP protocol; this is the Python SDK that makes
implementing the server side trivial.

**Officially supported servers** (built on this SDK):
- **[motionmcp-kimodo](https://github.com/animatica-ai/motionmcp-kimodo)** —
  the Kimodo SOMA model. Open-source, self-hosted.
- **MMCP Cloud** — hosted product. Adds skeleton retargeting, managed
  GPUs, batch generation, multi-tenant auth.

**Officially supported clients:**
- **[Proscenium for Blender](https://github.com/animatica-ai/proscenium-blender)** —
  directable motion generation in Blender 4.2+. Talks to any MMCP server.

Want to add another model or DCC? The protocol is yours to implement —
this SDK just makes the server side easy. See
[Servers & clients](https://animatica.ai/mmcp/docs/get-started/implementations)
for the full list and how to build your own.

```python
from motionmcp import Backbone, ModelSpec, GenerateRequest, MotionResult, serve

class MyBackbone(Backbone):
    def capabilities(self) -> ModelSpec:
        return ModelSpec(
            id="my-model",
            fps=30.0,
            canonical_skeleton=load_skeleton(),
            supports_retargeting=False,
            supported_constraints=["pose_keyframe"],
        )

    async def generate(self, req: GenerateRequest) -> MotionResult:
        rotations    = self.model.run(req)        # (N, T, J, 4) (x,y,z,w)
        translations = self.model.run_root(req)   # (N, T, 3)
        return MotionResult(
            rotations=rotations,
            root_translations=translations,
        )

if __name__ == "__main__":
    serve(MyBackbone())
```

That's the whole server. Hit `GET /capabilities` to discover the model;
`POST /generate` to run it. Returns standard glTF&nbsp;2.0 — load it with
any glTF parser.

## Install

```bash
pip install motionmcp-sdk
```

Requires Python 3.10+.

## What the SDK gives you

* **Schemas**: Pydantic models for the full wire format
  (`Skeleton`, `Segment`, `Constraint`, `Options`, `GenerateRequest`).
  Free validation, IDE autocomplete, OpenAPI docs at `/docs`.
* **Endpoints**: `GET /capabilities`, `POST /generate`, plus a generic
  exception → MMCP error envelope handler.
* **Generic checks**: unknown model, unknown joint in constraint,
  frame-out-of-range, retargeting policy, constraint count, prompt length,
  duration limit. All raise the right `error.code`.
* **glTF encoder**: numpy arrays in, glTF 2.0 JSON out, with the
  `MMCP_motion` extension correctly populated.
* **Reference impl**: `NullBackbone` returns a rest pose for any request.
  Useful for plugin development against a real server with zero ML deps.

## What you implement

* `Backbone.capabilities() -> ModelSpec` — what model you serve.
* `Backbone.generate(request) -> MotionResult` — the actual generation.
  May be `async def` or sync.
* `Backbone.setup()` / `teardown()` — optional, for model loading and
  GPU allocation.

That's it.

## Multi-model

Serve more than one model from one process:

```python
serve({
    "fast":    FastBackbone(),
    "quality": QualityBackbone(),
})
```

Or pass an iterable; ids come from each backbone's `capabilities().id`.

## NullBackbone — try it now

```bash
pip install motionmcp-sdk
python -m motionmcp.null_backbone
# → MMCP server on :8000, returning rest pose for any request
```

## What's NOT in v0

* **Async jobs** (`POST /generate` returning `202 Accepted`, `GET /generate/jobs/{id}`).
  Sync only for now; planned for v0.2.
* **Idempotency cache**. Spec-recommended but not required.
* **Binary glTF** (`model/gltf-binary`). JSON only for now.
* **fps resampling** between request fps and the model's native fps.
  Backbones currently see `request.fps(spec.fps)` and are responsible
  for honoring it.
* **Built-in retargeting**. Set `supports_retargeting=False` and require
  the canonical skeleton, or implement retargeting in your `generate()`.

These are deliberate v0 omissions to keep the surface tight. Feedback
welcome on which to prioritise next.

## Status

Alpha. The protocol is `v1.0-rc1`; the SDK API is `v0.1`. Expect minor
shape changes before `v1.0`.

## License

Apache 2.0.
