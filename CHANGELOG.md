# Changelog

## 0.9.0

MMCP protocol 1.2: motion and video references as prompts.

### Added

- `video_reference` segment (`VideoReferenceSegment`): a prompt given as a
  video instead of text. `video` is exactly one of `{"url": "https://..."}`
  (the server fetches it) or `{"data": <base64>, "media_type": "video/mp4" |
  "video/quicktime" | "video/webm"}`; optional `start_s` / `end_s` trim
  (`0 <= start_s < end_s`) and `fps` (a hint); the server follows the most
  prominent person in the video. Otherwise it **is a text segment**,
  validated with the text rules exactly as `motion_reference` is. The SDK
  checks the payload (one source; an https URL with a host and no user
  info, whitespace or bad port; standard padded base64, its size worked
  out from its length; finite numbers; the trim; 422) and never decodes
  the video: a backbone gets it parsed (`VideoSource.decoded()`,
  `.num_bytes`).
- `limits.max_video_bytes` (decoded inline data over it is
  `413 payload_too_large`) and `limits.max_video_seconds` (a trimmed video
  over it is `400 invalid_options`), both optional, `details.segment`
  naming the segment, left out of `/capabilities` when unset. A URL's size
  and an untrimmed video's length are the server's to enforce.
- Models opt in by listing `"video_reference"` in `supported_segments`;
  the existing gate answers `unsupported_segment` for models that don't.
  No server implements it yet.
- `GenerateRequest.video_reference` / `.video_references`, `VideoSource`.
- `motionmcp.client.video_source(path | url=, max_bytes=)`: a video file
  (base64, media type from the extension, size checked before reading) or
  an https URL (the server's URL rule) as a `video` object. Standard
  library only.
- `motionmcp.fetch_video_url(url, max_bytes=, timeout=)`: fetch a
  `video.url` without SSRF -- https only, every resolved address and every
  redirect hop must be public, the connection pinned to the checked
  address, a streamed byte cap and a deadline. Standard library only.
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
  is `400 invalid_options` (`details.segment` names it). It counts frames
  as sent: a server MUST accept a clip within it at any `fps` (resampling
  or trimming to its own window). It caps the clip, not the output (that is
  `max_duration_seconds`, as for text). Left out of `/capabilities` when
  unset.
- A motion reference must include its skeleton's root joint
  (`400 invalid_request`), and its numbers must be finite (422).
- Models opt in by listing `"motion_reference"` in `supported_segments`; the
  existing gate still answers `unsupported_segment` for models that don't.
  `supports_motion_reference_mixed` is advertised `true` exactly when a
  model lists `"motion_reference"` (a reference always mixes like text).
- `GenerateRequest.motion_reference` / `.motion_references`,
  `MotionReferenceSegment.reference_frames`.
- Error code `invalid_request` (400), for backbones.
- Client: `poll_job(server_url, location, ...)` resumes polling an
  accepted job, e.g. after refreshing a token on a 401 mid-poll, instead of
  starting a second job. (#8)

### Changed

- `PROTOCOL_VERSION` is `"1.2"` (was `"1.0"`; 1.1 was never stamped here).
  The SDK still sets no `options.guidance` default: a backbone applies its
  text default, to a reference as to text.
- Trajectory: the root moves along its shape (still / line / arc / Bezier,
  chosen by fit) with the body's centre of mass, not at a constant speed, so
  a stride's speeding up and slowing down no longer shows as the body
  floating forward and back when played in place. Models are just
  still/line/arc/bezier; params gain `speed_range` and drop the distance
  keys. (#7)
- Client: every `MmcpError` from `generate()` carries `details["phase"]`
  (`"generate"` or `"poll"`); poll errors also carry `details["location"]`.
  Codes and messages are unchanged. (#8)

- *Clarified:* clients SHOULD send identity `rest_rotation` on every
  joint (1.0 meanings unchanged: `rest_translation` / `rest_rotation` are
  parent-local, `fill_mode: "rest"` pins to `rest_rotation`); a server that
  can't honour a non-identity one MUST refuse it with `400 invalid_skeleton`.
- The SDK server refuses a body over `limits.max_request_bytes` before
  parsing it (`413 payload_too_large`: from `Content-Length`, or while
  streaming one sent without), and each model's own cap after.
- A segment type the model doesn't list -- or one the SDK doesn't know --
  is `unsupported_segment`, checked before the segment's payload is parsed.

### Fixed

- A request failing one of the schema's own validators (e.g. `root_path`
  lengths) is a `422 schema_validation` envelope with the message, not a
  500 from an unserialisable error context; the envelope no longer echoes
  the offending input (which could be megabytes of `video.data`).
- A skeleton with a bad topology (the request's or a segment's) is
  `400 invalid_skeleton`, as documented, not `422 schema_validation`.
