# SPDX-License-Identifier: Apache-2.0
"""``glb_to_gltf``: a GLB unpacks into the same glTF JSON document.

The golden values come from parsing the JSON document directly; the GLB
round trip must give the same arrays, so ``gltf_parser`` never has to know
the answer arrived as GLB.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from gltf_doc import build_glb, build_gltf

from motionmcp.client.glb import glb_to_gltf
from motionmcp.client.gltf_parser import parse_gltf


class TestGlbToGltf:
    def test_round_trip_parses_to_the_same_golden_values(self):
        doc = build_gltf()
        golden = parse_gltf(doc)

        converted = glb_to_gltf(build_glb(doc))
        result = parse_gltf(converted)

        json.dumps(converted)  # no bytes in the document
        assert result["joint_names"] == golden["joint_names"]
        for key in ("local_rot_mats", "posed_joints"):
            assert np.allclose(result[key], golden[key]), key
        assert result["rest_positions"].keys() == golden["rest_positions"].keys()
        for name, pos in golden["rest_positions"].items():
            assert np.allclose(result["rest_positions"][name], pos), name
        assert converted["buffers"][0]["byteLength"] == doc["buffers"][0]["byteLength"]

    def test_bad_magic_is_value_error(self):
        data = bytearray(build_glb(build_gltf()))
        data[0:4] = b"gLTF"

        with pytest.raises(ValueError):
            glb_to_gltf(bytes(data))

    def test_bad_version_is_value_error(self):
        data = bytearray(build_glb(build_gltf()))
        struct.pack_into("<I", data, 4, 1)

        with pytest.raises(ValueError):
            glb_to_gltf(bytes(data))

    def test_length_mismatch_is_value_error(self):
        data = build_glb(build_gltf())

        with pytest.raises(ValueError):
            glb_to_gltf(data[:-4])

    def test_glb_without_bin_chunk_returns_the_document_unchanged(self):
        doc = build_gltf()

        assert glb_to_gltf(build_glb(doc, with_bin=False)) == doc
