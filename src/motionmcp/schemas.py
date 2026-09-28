# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the MMCP wire format.

The shapes here mirror the canonical protocol spec at https://animatica.ai/mmcp/.
Keep them in sync with that source of truth.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---- Type aliases ---------------------------------------------------------

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]   # (x, y, z, w)


# ---- Skeleton -------------------------------------------------------------

class Joint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=128)
    parent: Optional[str] = None
    rest_translation: Vec3
    rest_rotation: Quaternion
    display_name: Optional[str] = Field(None, max_length=256)


class Skeleton(BaseModel):
    model_config = ConfigDict(extra="forbid")
    joints: list[Joint] = Field(..., min_length=1)
    coordinate_system: Literal["right_handed_y_up"] = "right_handed_y_up"
    units: Literal["meters"] = "meters"

    @model_validator(mode="after")
    def _validate_topology(self) -> "Skeleton":
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique within the skeleton")
        roots = [j for j in self.joints if j.parent is None]
        if len(roots) != 1:
            raise ValueError(
                f"exactly one joint must have parent=null; got {len(roots)}"
            )
        seen: set[str] = set()
        for j in self.joints:
            if j.parent is not None and j.parent not in seen:
                raise ValueError(
                    f"joint {j.name!r} references parent {j.parent!r} that is "
                    "not defined or appears later in the joints list"
                )
            seen.add(j.name)
        return self


# ---- Segments -------------------------------------------------------------

class TextSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    prompt: str = Field(..., min_length=1)
    duration_frames: int = Field(..., gt=0)
    language: Optional[str] = "en"
    # Per-segment random seed. Overrides the request-level ``Options.seed``
    # for this segment only; ``None`` falls back to ``Options.seed`` (and a
    # ``None`` there means a non-deterministic draw). Lets a multi-segment
    # request re-roll one segment independently without disturbing the seeds
    # of the others.
    seed: Optional[int] = None


class UnconditionedSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["unconditioned"]
    duration_frames: int = Field(..., gt=0)
    # See ``TextSegment.seed`` — same per-segment override semantics.
    seed: Optional[int] = None


class PoseSegment(BaseModel):
    """A single-pose generation from text.

    Produces exactly one frame of output. Unlike :class:`TextSegment`,
    which generates a span of motion, a pose segment is intrinsically
    instantaneous; it has no ``duration_frames`` field.

    Servers advertise support via ``ModelSpec.supported_segments``
    (must include ``"pose"``). Backbones without the specialized
    pose model omit ``"pose"`` from that list, and the SDK rejects
    pose segments with an ``unsupported_segment`` error before they
    reach the backbone.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["pose"]
    prompt: str = Field(..., min_length=1)
    language: Optional[str] = "en"

    # Intrinsic frame count — used by GenerateRequest.total_frames so all
    # segment types share the same surface. Not present on the wire.
    @property
    def duration_frames(self) -> int:
        return 1


class MotionReferenceSegment(BaseModel):
    """A prompt given as a motion instead of text (MMCP 1.2).

    The client sends a clip -- sampled from its own rig -- in place of a text
    prompt. The model reads it as it would read a caption (a backbone turns the
    clip into the same kind of conditioning vector its text encoder makes), so
    the segment is a ``text`` segment in every other respect: it covers
    ``duration_frames`` of the take, takes a per-segment ``seed``, mixes with
    ``text`` / ``unconditioned`` / other reference segments in any order,
    goes with constraints, ``options.loop``, ``num_samples`` and guidance
    exactly as a text segment does, and the SDK validates it with the same
    rules. What "a walk like this clip" produces is what the prompt "walk"
    would: new performances of that kind of motion, not a copy of the clip.

    ``rotations`` are local-to-parent quaternions ``(x, y, z, w)`` per frame,
    one per ``joint_names`` entry, in the same convention as
    ``pose_keyframe.joint_rotations``; ``root_positions`` are the root joint's
    world position per frame (Y-up metres), as ``pose_keyframe.root_position``.
    Quaternions are taken as sent, like ``pose_keyframe``: not required to be
    exactly unit length; a backbone normalises them as it needs. ``fps`` is
    the rate of the clip's samples, which may differ from the request's; the
    clip's length is independent of ``duration_frames``.

    ``skeleton`` (optional, same shape as ``GenerateRequest.skeleton``) is the
    reference clip's own rig: when present, ``joint_names`` / ``rotations`` /
    ``root_positions`` refer to it and the server retargets the clip from it,
    so the clip can come from any rig; when absent they refer to the request
    skeleton. Either way the output is on the request skeleton.

    Servers advertise support by listing ``"motion_reference"`` in
    ``supported_segments``; ``limits.max_reference_frames`` caps the clip's
    own length (the output length is capped like any segment's, by
    ``max_duration_seconds``).

    ``fidelity`` is deprecated and ignored: it is accepted only as ``0`` so
    clients written against the first 1.2 drafts keep working.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["motion_reference"]
    duration_frames: int = Field(..., gt=0)
    skeleton: Optional[Skeleton] = None
    joint_names: list[str] = Field(..., min_length=1)
    rotations: list[list[Quaternion]] = Field(..., min_length=2)
    root_positions: list[Vec3] = Field(..., min_length=2)
    fps: float = Field(..., gt=0)
    # Deprecated (first 1.2 drafts): accepted as 0 only, ignored.
    fidelity: float = Field(0.0, ge=0.0, le=0.0, exclude=True)
    # See ``TextSegment.seed`` -- same per-segment override semantics.
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _check_shapes(self) -> "MotionReferenceSegment":
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        t = len(self.rotations)
        if len(self.root_positions) != t:
            raise ValueError(
                f"root_positions must have one entry per frame ({t}), "
                f"got {len(self.root_positions)}"
            )
        j = len(self.joint_names)
        for i, frame in enumerate(self.rotations):
            if len(frame) != j:
                raise ValueError(
                    f"rotations[{i}] must have one quaternion per joint_names "
                    f"entry ({j}), got {len(frame)}"
                )
        return self

    @property
    def reference_frames(self) -> int:
        """Frames in the reference clip (T)."""
        return len(self.rotations)


Segment = Annotated[
    Union[TextSegment, UnconditionedSegment, PoseSegment, MotionReferenceSegment],
    Field(discriminator="type"),
]


# ---- Constraints ----------------------------------------------------------

class RootPathConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["root_path"]
    frames: list[int] = Field(..., min_length=1)
    positions_xz: list[Vec2] = Field(..., min_length=1)
    heading_radians: Optional[list[float]] = None

    @model_validator(mode="after")
    def _check_lengths(self) -> "RootPathConstraint":
        n = len(self.frames)
        if len(self.positions_xz) != n:
            raise ValueError(
                f"positions_xz must have length {n}, got {len(self.positions_xz)}"
            )
        if self.heading_radians is not None and len(self.heading_radians) != n:
            raise ValueError(
                f"heading_radians must have length {n}, got {len(self.heading_radians)}"
            )
        if any(self.frames[i + 1] <= self.frames[i] for i in range(n - 1)):
            raise ValueError("frames must be strictly ascending")
        return self


class EffectorTargetConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["effector_target"]
    joint: str
    frames: list[int] = Field(..., min_length=1)
    positions: list[Vec3] = Field(..., min_length=1)
    rotations: Optional[list[Quaternion]] = None

    @model_validator(mode="after")
    def _check_lengths(self) -> "EffectorTargetConstraint":
        n = len(self.frames)
        if len(self.positions) != n:
            raise ValueError(
                f"positions must have length {n}, got {len(self.positions)}"
            )
        if self.rotations is not None and len(self.rotations) != n:
            raise ValueError(
                f"rotations must have length {n}, got {len(self.rotations)}"
            )
        return self


class PoseKeyframeConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["pose_keyframe"]
    frame: int = Field(..., ge=0)
    joint_rotations: dict[str, Quaternion] = Field(..., min_length=1)
    root_position: Optional[Vec3] = None
    fill_mode: Literal["rest", "generate"] = "generate"


Constraint = Annotated[
    Union[RootPathConstraint, EffectorTargetConstraint, PoseKeyframeConstraint],
    Field(discriminator="type"),
]


# ---- Options --------------------------------------------------------------

class Guidance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["nocfg", "regular", "separated"]
    weight: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_weight(self) -> "Guidance":
        expected = {"nocfg": 0, "regular": 1, "separated": 2}[self.type]
        if len(self.weight) != expected:
            raise ValueError(
                f"guidance.weight must have length {expected} for "
                f"type={self.type!r}, got {len(self.weight)}"
            )
        return self


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diffusion_steps: int = Field(100, ge=1, le=500)
    num_samples: int = Field(1, ge=1, le=16)
    seed: Optional[int] = None
    post_processing: bool = True
    transition_frames: int = Field(5, ge=0, le=60)
    guidance: Optional[Guidance] = None
    # Sample the motion as a seamless cycle: the returned clip's last frame is
    # its first again (same pose, height and heading; the root moved on by one
    # cycle's travel), so repeating it has no seam. One segment only. Only send
    # it to a model that advertises ``supports_loop``.
    loop: bool = False


# ---- Timing ---------------------------------------------------------------

class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fps: float = Field(..., gt=0)


# ---- GenerateRequest ------------------------------------------------------

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: str = Field(..., pattern=r"^\d+\.\d+$")
    model: str
    skeleton: Skeleton
    segments: list[Segment] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    duration_frames: Optional[int] = Field(None, gt=0)
    timing: Optional[Timing] = None
    options: Optional[Options] = None

    @model_validator(mode="after")
    def _cross_field(self) -> "GenerateRequest":
        if not self.segments and not self.constraints:
            raise ValueError(
                "at least one of `segments` or `constraints` must be non-empty"
            )
        if not self.segments and self.duration_frames is None:
            raise ValueError(
                "`duration_frames` is required when `segments` is empty"
            )
        if self.segments and self.duration_frames is not None:
            raise ValueError(
                "`duration_frames` is forbidden when `segments` is non-empty"
            )
        return self

    @property
    def total_frames(self) -> int:
        """Total frame count for the request, regardless of segment vs constraints."""
        if self.segments:
            return sum(s.duration_frames for s in self.segments)
        assert self.duration_frames is not None
        return self.duration_frames

    @property
    def motion_reference(self) -> Optional[MotionReferenceSegment]:
        """The request's first ``motion_reference`` segment, or None."""
        for s in self.segments:
            if isinstance(s, MotionReferenceSegment):
                return s
        return None

    @property
    def motion_references(self) -> list[MotionReferenceSegment]:
        """Every ``motion_reference`` segment of the request, in order."""
        return [s for s in self.segments if isinstance(s, MotionReferenceSegment)]

    def fps(self, model_native_fps: float) -> float:
        """Effective fps for this request: ``timing.fps`` if set, else the model native."""
        return self.timing.fps if self.timing is not None else model_native_fps
