# SPDX-License-Identifier: Apache-2.0
"""Protocol-level constants. Versioned with the spec."""

from __future__ import annotations


PROTOCOL_VERSION = "1.0"
PROTOCOL_MAJOR = 1

ROTATION_FORMAT = "quaternion_xyzw"
COORDINATE_SYSTEM = "right_handed_y_up"
UNITS = "meters"

RESPONSE_FORMATS = ["gltf_2.0_json"]

SUPPORTED_CONSTRAINTS = ["root_path", "effector_target", "pose_keyframe"]
SUPPORTED_GUIDANCE_TYPES = ["nocfg", "regular", "separated"]

DEFAULT_LIMITS = {
    "max_duration_seconds":         30.0,
    "max_num_samples":              16,
    "max_constraints_per_request":  64,
    "max_prompt_length":            1000,
    "max_request_bytes":            1_048_576,
}
