# SPDX-License-Identifier: Apache-2.0
"""Backbone abstraction — what users implement to plug their model into MMCP.

Two pieces:

* :class:`ModelSpec` — the per-model metadata served at ``GET /capabilities``.
* :class:`Backbone` — the ABC users subclass; defines :meth:`Backbone.capabilities`
  and :meth:`Backbone.generate`.

A :class:`MotionResult` is what :meth:`Backbone.generate` returns: per-sample
quaternion rotations, root translations, and optional foot-contact masks. The
SDK encodes this to glTF for you.
"""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .protocol import (
    DEFAULT_LIMITS,
    SUPPORTED_GUIDANCE_TYPES,
    SUPPORTED_SEGMENTS,
)
from .schemas import (GenerateRequest, ModelOption, PromptVocabularyEntry, Skeleton,
                      normalize_prompt)


# ---- ModelSpec — published at /capabilities --------------------------------

class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_duration_seconds: float = float(DEFAULT_LIMITS["max_duration_seconds"])
    max_num_samples: int = int(DEFAULT_LIMITS["max_num_samples"])
    max_constraints_per_request: int = int(DEFAULT_LIMITS["max_constraints_per_request"])
    max_prompt_length: int = int(DEFAULT_LIMITS["max_prompt_length"])
    max_request_bytes: int = int(DEFAULT_LIMITS["max_request_bytes"])
    max_batch_size: int = int(DEFAULT_LIMITS["max_batch_size"])
    # Most frames a ``motion_reference`` segment's clip may carry (the clip
    # itself; its ``duration_frames`` is capped like any segment's, by
    # ``max_duration_seconds``). Set it when ``supported_segments`` lists
    # ``"motion_reference"``; None = no cap.
    max_reference_frames: Optional[int] = Field(None, gt=0)
    # A ``video_reference`` segment's video (1.2): most bytes of inline
    # ``video.data`` (decoded), and most seconds of video to read when the
    # request trims it (``end_s``). Set them when ``supported_segments`` lists
    # ``"video_reference"``; the SDK enforces what it can see and the backbone
    # the rest (a URL's size, an untrimmed video's length). None = no cap.
    max_video_bytes: Optional[int] = Field(None, gt=0)
    max_video_seconds: Optional[float] = Field(None, gt=0)


class ModelSpec(BaseModel):
    """Per-model entry under ``/capabilities.models[]``.

    All fields except ``id``, ``fps``, and ``canonical_skeleton`` have spec
    defaults — you only have to fill in what differs from the conservative
    baseline.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    fps: float = Field(..., gt=0)
    canonical_skeleton: Skeleton

    supports_retargeting: bool = False
    # True when a response may carry a translation for joints other than the
    # root (``MotionResult.translations``). A rig whose joints are turned
    # rather than slid never needs it; one that DEFORMS by moving joints --
    # a face, a fin membrane, a tongue -- cannot be carried without it, and a
    # client that ignores the extra channels still gets a correct skeleton,
    # just a stiffer one. Off unless a backbone says otherwise, so a client
    # that has never heard of it is unaffected.
    supports_joint_translation: bool = False
    supports_async: bool = False
    # True when the backbone honours a per-segment ``seed`` (see
    # ``TextSegment.seed``). Clients should only attach per-segment seeds when
    # this is advertised — older servers reject unknown fields (extra=forbid).
    supports_segment_seed: bool = False
    # True when the backbone samples ``options.loop`` as a cycle. Clients
    # should only set ``loop`` when this is advertised; the SDK rejects it
    # otherwise.
    supports_loop: bool = False
    # The model can thin its own curves to a tolerance the request names, and
    # will key every frame when it does not. See ``Options.keyframe_budget``.
    supports_keyframe_reduction: bool = False
    # THE DIALS THIS MODEL HAS THAT THE PROTOCOL HAS NO WORD FOR. Declared
    # here, published in /capabilities, and sent back by name in
    # ``options.model_options``. Empty means the model takes none, and a
    # request that sends any is refused. See ``ModelOption``.
    model_options: list[ModelOption] = Field(default_factory=list)
    # HOW THIS MODEL READS A TEXT PROMPT. ``"free"`` (or unset): any text, as
    # a caption. ``"vocabulary"``: only the commands in ``prompt_vocabulary``,
    # and the server refuses any other prompt rather than ignore it. Both are
    # left out of /capabilities when unset, so a client that predates them
    # still reads this model.
    text_prompt_mode: Optional[Literal["free", "vocabulary"]] = None
    prompt_vocabulary: list[PromptVocabularyEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _vocabulary_is_usable(self) -> "ModelSpec":
        # a vocabulary-mode model with no commands refuses every text prompt
        if self.text_prompt_mode == "vocabulary":
            if not self.prompt_vocabulary:
                raise ValueError('text_prompt_mode "vocabulary" needs a prompt_vocabulary')
            if "text" not in self.supported_segments:
                raise ValueError('text_prompt_mode "vocabulary" needs "text" in supported_segments')
        seen = [normalize_prompt(e.prompt) for e in self.prompt_vocabulary]
        if len(set(seen)) != len(seen):
            raise ValueError("prompt_vocabulary lists the same prompt twice")
        return self
    # ``motion_reference`` segments are prompts like text segments (MMCP 1.2):
    # several in one request, mixed with ``text`` / ``unconditioned`` ones.
    # Implied by ``"motion_reference"`` in ``supported_segments`` -- the SDK
    # server advertises it as true for such a model, whatever is set here --
    # and kept for clients that read it.
    supports_motion_reference_mixed: bool = False
    # True when POST /generate accepts a batch body, ``{"requests": [...]}``,
    # of up to ``limits.max_batch_size`` generate requests. Advertised by the
    # SDK server for every model it serves (it can always run a batch item by
    # item); a backbone that batches on the GPU overrides ``generate_batch``.
    supports_batch: bool = False
    # True when each response sample carries its travel trajectory
    # (``MMCP_motion.samples[b].trajectory``): the path the motion moves along,
    # so a client can play it in place by removing exactly that. Advertised by
    # the SDK server for every model (it computes it, see motionmcp.trajectory).
    supports_trajectory: bool = False

    supported_constraints: list[str] = Field(default_factory=list)
    # Wire-format segment types this model accepts. Defaults to "text" and
    # "unconditioned" (every conforming server supports those). Backbones
    # with a specialized text-to-pose model add "pose", and ones that can
    # take a motion as a prompt add "motion_reference" (with
    # ``limits.max_reference_frames``), a video as a prompt
    # "video_reference" (with ``limits.max_video_bytes`` /
    # ``max_video_seconds``); the SDK rejects any segment whose type isn't
    # listed here.
    supported_segments: list[str] = Field(
        default_factory=lambda: list(SUPPORTED_SEGMENTS)
    )
    supported_guidance_types: list[str] = Field(
        default_factory=lambda: list(SUPPORTED_GUIDANCE_TYPES)
    )
    predicted_contact_joints: list[str] = Field(default_factory=list)

    native_clip_seconds: float = 10.0
    chunking: str = Field("none", pattern=r"^(none|stitched)$")
    recommended_max_duration_seconds: float = 12.0

    limits: Limits = Field(default_factory=Limits)


# ---- MotionResult — what generate() returns --------------------------------

@dataclass
class MotionResult:
    """The output of :meth:`Backbone.generate`.

    Shape conventions (B = num_samples, T = num_frames, J = num_joints):

    * ``rotations``: ``(B, T, J, 4)`` quaternions ``(x, y, z, w)``, local-to-parent.
    * ``root_translations``: ``(B, T, 3)`` world-space root positions.
    * ``translations``: optional ``(B, T, J, 3)``, a joint's offset from its
      parent, in the parent's frame -- what glTF means by a node's
      translation, and what ``rest_translation`` is when nothing has moved.
      For the root it is the world position, which is the same thing. Leave it
      None and only the root moves, which is every rig whose joints are
      turned rather than slid; set it for a rig that DEFORMS by moving joints,
      and say so with ``supports_joint_translation``.
    * ``joint_names``: optional ``(J,)`` list. Defaults to the request skeleton's
      joint order. Provide this only if your model retargeted internally and
      you need to declare a different output order — the SDK will validate
      that ``joint_names`` matches the request skeleton (or your declared
      ``canonical_to_request`` map).
    * ``foot_contacts``: optional ``{joint_name: (B, T) bool array}``. Joint
      names must exist in the request skeleton.
    * ``chunk_boundaries``: optional per-sample list of frame indices where
      the server stitched internal chunks. Empty / None means no stitching.
    * ``canonical_to_request``: optional ``{canonical_joint: request_joint}``
      map echoed in the response extension. Only populate when the server
      actually retargeted.
    * ``trajectories``: optional per-sample travel trajectory (the wire dict of
      :mod:`motionmcp.trajectory`). Leave it None and the SDK computes it; set
      it when the backbone knows better.
    """

    rotations: np.ndarray
    root_translations: np.ndarray
    translations: Optional[np.ndarray] = None
    # WHERE THE CURVES ACTUALLY BEND. {joint index: {"rotation": [frame, ...],
    # "translation": [...]}} -- the frames a channel is worth keying, for a
    # model that knows. Leave it out and every frame is keyed, which is exact
    # and is what a take being retargeted wants; fill it in and what arrives
    # in the DCC is a curve an animator can hold, keyed where it turns.
    #
    # The model chooses, because only the model knows what straightening its
    # own curve costs. An encoder sees quaternions and cannot tell which of
    # them a long chain is about to multiply into a visible error.
    keyframes: Optional[dict] = None
    joint_names: Sequence[str] | None = None
    foot_contacts: dict[str, np.ndarray] = field(default_factory=dict)
    chunk_boundaries: Sequence[Sequence[int]] | None = None
    canonical_to_request: dict[str, str] | None = None
    trajectories: Sequence[dict | None] | None = None
    # WHAT THE MODEL WANTS TO SAY ABOUT THE REQUEST IT JUST ANSWERED.
    # A list of {code, severity, message, ...}. It is not an error channel --
    # the answer is in the response either way -- it is for the things a model
    # can see and the caller cannot: that the path asked for is faster than
    # the character swims, that a drawn height was ignored, that a dial was
    # clamped by something else. Silence here is a model with nothing to add,
    # not a model that checked.
    notes: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        r = self.rotations
        if r.ndim != 4 or r.shape[-1] != 4:
            raise ValueError(
                f"rotations must be (B, T, J, 4); got {r.shape}"
            )
        B, T, J, _ = r.shape
        t = self.root_translations
        if t.shape != (B, T, 3):
            raise ValueError(
                f"root_translations must be ({B}, {T}, 3); got {t.shape}"
            )
        if self.translations is not None and self.translations.shape != (B, T, J, 3):
            raise ValueError(
                f"translations must be ({B}, {T}, {J}, 3); "
                f"got {self.translations.shape}"
            )
        for joint, mask in self.foot_contacts.items():
            if mask.shape != (B, T):
                raise ValueError(
                    f"foot_contacts[{joint!r}] must be ({B}, {T}); "
                    f"got {mask.shape}"
                )

    @property
    def num_samples(self) -> int:
        return int(self.rotations.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.rotations.shape[1])

    @property
    def num_joints(self) -> int:
        return int(self.rotations.shape[2])


# ---- Backbone ABC ----------------------------------------------------------

class Backbone(abc.ABC):
    """The contract a user implements to plug their motion model into MMCP.

    Subclasses MUST implement:

    * :meth:`capabilities` — returns a :class:`ModelSpec` describing the model.
    * :meth:`generate` — runs the model on a validated :class:`GenerateRequest`
      and returns a :class:`MotionResult`.

    :meth:`generate` may be ``async def`` or a regular function — the SDK
    handles both. Async is preferred when the underlying model has any I/O or
    benefits from yielding (queueing on a GPU worker, awaiting a remote call).

    Raise :class:`~motionmcp.errors.ProtocolError` (or one of its convenience
    constructors in :mod:`motionmcp.errors`) for protocol-level failures —
    the SDK converts them to the standard error envelope. Any other exception
    becomes a generic ``500 internal_error``.
    """

    @abc.abstractmethod
    def capabilities(self) -> ModelSpec:
        """Return the per-model entry that goes under ``/capabilities.models[]``."""

    @abc.abstractmethod
    def generate(self, request: GenerateRequest) -> MotionResult | Awaitable[MotionResult]:
        """Run motion generation for one validated request.

        May be implemented as either ``def`` or ``async def``.
        """

    def generate_batch(
        self, requests: list[GenerateRequest],
    ) -> list[MotionResult | BaseException] | Awaitable[list[MotionResult | BaseException]]:
        """Optional: run several validated requests at once.

        Return one entry per request, in order: a :class:`MotionResult`, or
        the exception that item raised (a :class:`ProtocolError` for a clean
        envelope). Override it when the model can batch — one GPU pass over
        several characters instead of several passes. Not overriding it is
        fine: the SDK then runs :meth:`generate` on each item in turn.

        May be implemented as either ``def`` or ``async def``.
        """
        raise NotImplementedError

    # Optional hook: called once at server startup. Use it to load model
    # weights, allocate GPU buffers, warm caches.
    def setup(self) -> None:  # pragma: no cover - default no-op
        return

    # Optional hook: called once at server shutdown.
    def teardown(self) -> None:  # pragma: no cover - default no-op
        return


def is_async_generate(backbone: Backbone) -> bool:
    """Return True if the backbone's ``generate`` is an async coroutine function."""
    return inspect.iscoroutinefunction(backbone.generate)
