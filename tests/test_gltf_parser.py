# SPDX-License-Identifier: Apache-2.0
"""Golden-file coverage for the glTF spine of ``motion_data``.

The audit's coverage map found the entire data path dark: every number a
generation puts on a rig flows through ``parse_gltf``, and no SDK test
exercised it — trust rode on the plugin suites alone. These tests build a
minimal but *complete* glTF document the way the MMCP server does (embedded
base64 buffers, per-sample animations, the ``MMCP_motion`` contacts
extension) and pin the parsed output value by value.

Three joints, two frames: small enough to verify by hand, rich enough to
cover rotation + translation channels, rest-geometry accumulation, contact
mapping by NAME, and the multi-sample path.
"""

from __future__ import annotations

import numpy as np
import pytest
from gltf_doc import build_gltf

from motionmcp.client.gltf_parser import parse_gltf, parse_gltf_samples

# ---------------------------------------------------------------------------
# The golden values
# ---------------------------------------------------------------------------

def test_shapes_names_and_fps():
    m = parse_gltf(build_gltf())
    assert m["joint_names"] == ["Hips", "Spine", "Head"]
    assert m["num_frames"] == 2 and m["num_joints"] == 3
    assert m["fps"] == pytest.approx(30.0)
    assert m["local_rot_mats"].shape == (2, 3, 3, 3)
    assert m["posed_joints"].shape == (2, 3, 3)


def test_rotation_and_translation_values():
    m = parse_gltf(build_gltf())
    assert np.allclose(m["local_rot_mats"][0], np.eye(3), atol=1e-6)
    # 90° about +Y maps +X → -Z: column picture of the standard Ry(90°).
    ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    assert np.allclose(m["local_rot_mats"][1, 0], ry90, atol=1e-6)
    assert np.allclose(m["local_rot_mats"][1, 1], np.eye(3), atol=1e-6)
    assert np.allclose(m["posed_joints"][:, 0],
                       [[1, 2, 3], [4, 5, 6]], atol=1e-6)
    # Nodes without a translation track stay at the zero the parser documents.
    assert np.allclose(m["posed_joints"][:, 1:], 0.0)


def test_rest_geometry_is_accumulated_not_local():
    m = parse_gltf(build_gltf())
    assert m["hierarchy"] == [("Hips", None), ("Spine", "Hips"),
                              ("Head", "Spine")]
    assert m["rest_positions"]["Hips"] == pytest.approx((0.0, 1.0, 0.0))
    assert m["rest_positions"]["Spine"] == pytest.approx((0.0, 1.5, 0.0))
    # The give-away value: 1.8 is only reachable by walking the chain.
    assert m["rest_positions"]["Head"] == pytest.approx((0.0, 1.8, 0.0))


def test_contacts_are_mapped_by_name_and_degrade_silently():
    doc = build_gltf(contacts=[{"foot_contacts": {
        "Head": [True, False],
        "NotAJoint": [True, True],          # unknown name → ignored
        "Spine": [True],                    # wrong length → ignored
    }}])
    m = parse_gltf(doc)
    assert m["foot_contacts"].dtype == bool
    assert m["foot_contacts"][:, 2].tolist() == [True, False]
    assert not m["foot_contacts"][:, :2].any()

    # No extension at all: all-False, never a raise (the 2026-05-21 captures).
    bare = parse_gltf(build_gltf())
    assert not bare["foot_contacts"].any()


def test_multi_sample_pairs_animations_with_extension_entries():
    doc = build_gltf(num_samples=2,
                     contacts=[{"foot_contacts": {"Head": [True, True]}}, {}])
    samples = parse_gltf_samples(doc)
    assert len(samples) == 2
    assert samples[0]["foot_contacts"][:, 2].all()
    assert not samples[1]["foot_contacts"].any()
    # A mismatched extension length degrades to no contacts for EVERY sample.
    doc = build_gltf(num_samples=2,
                     contacts=[{"foot_contacts": {"Head": [True, True]}}])
    assert not any(s["foot_contacts"].any() for s in parse_gltf_samples(doc))


def test_refusals():
    with pytest.raises(ValueError, match="no animations"):
        parse_gltf({"nodes": [], "buffers": []})
    with pytest.raises(ValueError, match="no rotation channels"):
        parse_gltf(build_gltf(with_rotations=False))
