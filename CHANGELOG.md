# Changelog

## Unreleased

MMCP protocol 1.2.

### Added

- `motion_reference` segment (`MotionReferenceSegment`): send a clip
  (`joint_names`, per-frame local `rotations` T×J×4, `root_positions` T×3,
  `fps`) and get `options.num_samples` new performances of the same kind of
  motion, `duration_frames` long (default: the reference length). `fidelity`
  in [0, 0.2] (default 0.0) picks how close they stay to the clip.
  `options.loop` with a reference is `400 invalid_request`; T ≥ 2 and consistent lengths are
  schema-checked (422); its joints must be on the request skeleton
  (`unknown_joint`).
- A reference is a prompt like a text segment's: on a model that advertises
  `supports_motion_reference_mixed` (new `ModelSpec` flag, default false) a
  request may carry several, in any order, mixed with `text` /
  `unconditioned` segments, each covering its own `duration_frames` and
  stitched with the same transitions. With more than one segment each
  reference needs `duration_frames` and `fidelity` 0 (`400 invalid_request`,
  `details.segment` names it); a `pose` segment never goes with one. Without
  the flag a reference stands alone (a second reference or another segment
  is `400 invalid_request`), so existing backbones are unchanged.
- `MotionReferenceSegment.seed`: per-segment seed, as `TextSegment.seed`.
- `GenerateRequest.motion_references` (every reference, in order).
- `MotionReferenceSegment.skeleton` (optional, the request `Skeleton`
  schema): the clip's own rig, so a reference can come from any rig. When
  set, `joint_names` / `rotations` / `root_positions` refer to it and
  `unknown_joint` is checked against it; a non-canonical one needs
  `supports_retargeting` (`retargeting_unsupported`).
- `duration_frames` may differ from the reference length; with
  `fidelity > 0` it must equal it (`400 invalid_request`).
- `limits.max_reference_frames` (optional): when set, a reference or a
  `duration_frames` longer than it is `400 invalid_options`. Left out of
  `/capabilities` when unset.
- Models opt in by listing `"motion_reference"` in `supported_segments`; the
  existing gate still answers `unsupported_segment` for models that don't.
- `MMCP_motion.samples[b].reference = {"fidelity": f}` on every sample of a
  reference response (added by the SDK); the client parser returns it as
  `motion["reference"]` (None otherwise).
- `GenerateRequest.motion_reference`, `MotionReferenceSegment.output_frames` /
  `.reference_frames`, `schemas.segment_frames()`; `total_frames` counts a
  reference's output length.
- Error code `invalid_request` (400).

### Changed

- `PROTOCOL_VERSION` is `"1.2"` (was `"1.0"`; 1.1 was never stamped here).
  The SDK still sets no `options.guidance` default: a backbone picks its own
  per segment type.
