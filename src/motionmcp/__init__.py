"""motionmcp — Python SDK for the Motion Model Context Protocol (MMCP).

Build a vendor-neutral motion-generation HTTP server in ~30 lines:

    from motionmcp import Backbone, ModelSpec, GenerateRequest, MotionResult, serve

    class MyBackbone(Backbone):
        def capabilities(self) -> ModelSpec:
            return ModelSpec(
                id="my-model",
                fps=30.0,
                supports_retargeting=False,
                supported_constraints=["pose_keyframe"],
                canonical_skeleton=load_skeleton(),
            )

        async def generate(self, req: GenerateRequest) -> MotionResult:
            ...
            return MotionResult(rotations=..., root_translations=...)

    if __name__ == "__main__":
        serve(MyBackbone())
"""

from .backbone import Backbone, ModelSpec, MotionResult
from .errors import ProtocolError
from .protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_CONSTRAINTS,
    SUPPORTED_GUIDANCE_TYPES,
    DEFAULT_LIMITS,
)
from .schemas import (
    Constraint,
    EffectorTargetConstraint,
    GenerateRequest,
    Guidance,
    Joint,
    Options,
    PoseKeyframeConstraint,
    RootPathConstraint,
    Segment,
    Skeleton,
    TextSegment,
    Timing,
    UnconditionedSegment,
)
from .server import build_app, serve

__all__ = [
    "Backbone",
    "Constraint",
    "DEFAULT_LIMITS",
    "EffectorTargetConstraint",
    "GenerateRequest",
    "Guidance",
    "Joint",
    "ModelSpec",
    "MotionResult",
    "Options",
    "PROTOCOL_VERSION",
    "PoseKeyframeConstraint",
    "ProtocolError",
    "RootPathConstraint",
    "SUPPORTED_CONSTRAINTS",
    "SUPPORTED_GUIDANCE_TYPES",
    "Segment",
    "Skeleton",
    "TextSegment",
    "Timing",
    "UnconditionedSegment",
    "build_app",
    "serve",
]

__version__ = "0.1.0"
