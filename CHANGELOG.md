# Changelog

## Unreleased

MMCP protocol 1.2.

### Added

- `motion_reference` segment (`MotionReferenceSegment`): a prompt given as a
  motion instead of text. Send a clip (`joint_names`, per-frame local
  `rotations` T×J×4, `root_positions` T×3, `fps`) where a text segment
  carries its prompt; the model reads it as it reads a caption and makes new
  motion of that kind. Otherwise the segment **is a text segment** and the
  SDK validates it with the text rules: `duration_frames` (required),
  per-segment `seed`, any number of them mixed in any order with `text` /
  `unconditioned` / `pose` segments, `options.loop` (on a `supports_loop`
  model, one segment), constraints over the whole take, `num_samples`,
  guidance, batch bodies. The response is a text response's (no extra
  metadata).
- Its payload's own checks: T ≥ 2 and consistent lengths (422); joints on
  the request skeleton, or on the segment's own `skeleton` when it carries
  one (`unknown_joint`); a non-canonical segment `skeleton` needs
  `supports_retargeting` (`retargeting_unsupported`).
- `MotionReferenceSegment.skeleton` (optional, the request `Skeleton`
  schema): the clip's own rig, so a reference can come from any rig.
- `limits.max_reference_frames` (optional): when set, a clip longer than it
  is `400 invalid_options` (`details.segment` names it). It caps the clip,
  not the output (that is `max_duration_seconds`, as for text). Left out of
  `/capabilities` when unset.
- Models opt in by listing `"motion_reference"` in `supported_segments`; the
  existing gate still answers `unsupported_segment` for models that don't.
  `supports_motion_reference_mixed` (`ModelSpec`) is advertised `true` for
  every such model -- a reference always mixes like text -- and kept for
  clients of the first 1.2 drafts.
- `MotionReferenceSegment.fidelity` is deprecated and ignored: accepted as
  `0` only (so first-draft clients that send `"fidelity": 0.0` keep
  working), not passed on to the backbone.
- `GenerateRequest.motion_reference` / `.motion_references`,
  `MotionReferenceSegment.reference_frames`.
- Error code `invalid_request` (400), for backbones.

### Changed

- `PROTOCOL_VERSION` is `"1.2"` (was `"1.0"`; 1.1 was never stamped here).
  The SDK still sets no `options.guidance` default: a backbone applies its
  text default, to a reference as to text.
