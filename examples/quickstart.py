"""Minimal motionmcp server — your model in ~30 lines.

Run with:

    python examples/quickstart.py

Then in another shell:

    curl -s http://localhost:8080/capabilities | jq
    curl -s http://localhost:8080/generate -H 'Content-Type: application/json' \\
         -d '{"protocol_version":"1.0","model":"my-model",
              "skeleton":{"joints":[{"name":"Hips","parent":null,
                                     "rest_translation":[0,0.95,0],
                                     "rest_rotation":[0,0,0,1]}]},
              "segments":[{"type":"text","prompt":"walk","duration_frames":30}]}'
"""

import numpy as np

from motionmcp import (
    Backbone,
    GenerateRequest,
    Joint,
    ModelSpec,
    MotionResult,
    Skeleton,
    serve,
)


class MyBackbone(Backbone):
    def capabilities(self) -> ModelSpec:
        return ModelSpec(
            id="my-model",
            fps=30.0,
            canonical_skeleton=Skeleton(joints=[
                Joint(name="Hips", parent=None,
                      rest_translation=(0.0, 0.95, 0.0),
                      rest_rotation=(0.0, 0.0, 0.0, 1.0)),
            ]),
            supports_retargeting=False,
            supported_constraints=["pose_keyframe"],
        )

    async def generate(self, req: GenerateRequest) -> MotionResult:
        # Replace this with: rotations, root_translations = self.model.run(req)
        N = req.options.num_samples if req.options else 1
        T = req.total_frames
        J = len(req.skeleton.joints)

        rotations = np.zeros((N, T, J, 4), dtype=np.float32)
        rotations[..., 3] = 1.0
        root_translations = np.zeros((N, T, 3), dtype=np.float32)

        return MotionResult(
            rotations=rotations,
            root_translations=root_translations,
        )


if __name__ == "__main__":
    serve(MyBackbone())
