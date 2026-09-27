# SPDX-License-Identifier: Apache-2.0
"""The travel trajectory: synthetic bodies, and the server's response."""

from __future__ import annotations

import math

import numpy as np
import pytest

from motionmcp.trajectory import resolve_roles, trajectory_from_points

FPS = 30.0


def _walker(n=90, speed=1.2, sway=0.04, turn=0.0, loop=True, ramp=False):
    """A stick walker (+Z up): the pelvis sways and surges about a line (or an
    arc, ``turn`` rad/s), feet planted in turn, one step every 0.5 s."""
    t = np.arange(n) / FPS
    step = 0.5
    if turn:
        r = speed / turn
        s = speed * t
        path = np.stack([r * (1 - np.cos(s / r)), -r * np.sin(s / r)], 1)
        heading = s / r
    else:
        # ramp: from standing to `speed` over the take (a start, a run-up)
        dist = speed * t * t / (2 * t[-1]) if ramp else speed * t
        path = np.stack([np.zeros(n), -dist], 1)
        heading = np.zeros(n)
    fwd = np.stack([np.sin(heading), -np.cos(heading)], 1)
    left = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
    phase = 2 * np.pi * t / (2 * step)
    hips_xy = path + left * (sway * np.sin(phase))[:, None] + fwd * (0.03 * np.sin(2 * phase))[:, None]
    hips = np.c_[hips_xy, 0.9 + 0.02 * np.cos(2 * phase)]
    J = {"hips": (hips, hips + [0, 0, 0.1]), "spine": (hips + [0, 0, 0.1], hips + [0, 0, 0.3]),
         "chest": (hips + [0, 0, 0.3], hips + [0, 0, 0.5]), "neck": (hips + [0, 0, 0.5], hips + [0, 0, 0.6]),
         "head": (hips + [0, 0, 0.6], hips + [0, 0, 0.8])}
    for side, sgn, off in (("l", 1, 0.0), ("r", -1, step)):
        # a foot is planted for 0.6 s of every 1 s: at its plant spot, down; else swinging, up
        foot = np.zeros((n, 3))
        for i in range(n):
            k = math.floor((t[i] + off) / (2 * step))
            u = ((t[i] + off) % (2 * step)) / (2 * step)
            plant_t = k * 2 * step - off + 0.3
            j = int(np.clip(round(plant_t * FPS), 0, n - 1))
            spot = path[j] + left[j] * sgn * 0.1
            if u < 0.6:
                foot[i] = [spot[0], spot[1], 0.07]
            else:
                nxt = path[min(n - 1, j + int(2 * step * FPS))] + left[j] * sgn * 0.1
                w = (u - 0.6) / 0.4
                foot[i] = [*(spot + (nxt - spot) * w), 0.07 + 0.12 * math.sin(math.pi * w)]
        toe = foot + np.c_[fwd * 0.15, np.full(n, -0.04)]
        J[f"{side}_foot"] = (foot, toe)
        J[f"{side}_leg"] = (hips + np.c_[left * sgn * 0.1, np.zeros(n)], (hips + foot) / 2)
        J[f"{side}_shin"] = ((hips + foot) / 2, foot)
    return J, path


def test_straight_walk_is_a_line_at_its_speed():
    J, _ = _walker(speed=1.2)
    t = trajectory_from_points(J, FPS, loop=True, root_xy=J["hips"][0][-1, :2] - J["hips"][0][0, :2])
    assert t["model"] == "line"
    assert abs(t["params"]["speed"] - 1.2) < 0.05
    # the sway and surge are not in it: the trajectory is straight
    assert np.abs(t["xy"][:, 0]).max() < 0.01
    assert np.allclose(t["yaw"], 0)


def test_an_acceleration_is_timed_not_averaged():
    """A run-up from standing: a constant-speed line would run ahead of the body
    early and behind it late -- in place, the body slides back, then forward."""
    J, path = _walker(n=150, speed=4.0, ramp=True)
    t = trajectory_from_points(J, FPS, loop=False)
    assert "distance curve" in t["model"]
    # along the direction of travel the trajectory keeps pace with the path
    err = np.abs(t["xy"][:, 1] - path[:, 1])
    assert err.max() < 0.08, err.max()


def test_loop_closes_on_the_roots_travel():
    J, _ = _walker(speed=1.0)
    travel = J["hips"][0][-1, :2] - J["hips"][0][0, :2]
    t = trajectory_from_points(J, FPS, loop=True, root_xy=travel)
    assert np.allclose(t["xy"][-1] - t["xy"][0], travel, atol=1e-9)


def test_curving_walk_is_an_arc_and_turns():
    J, _ = _walker(n=120, speed=1.1, turn=0.5)
    t = trajectory_from_points(J, FPS, loop=True)
    assert t["model"] == "arc"
    assert abs(math.radians(t["params"]["turn_deg_s"]) - 0.5) < 0.05
    assert t["yaw"][-1] > 1.5                          # it turned ~2 rad over the take


def test_standing_is_still():
    J, _ = _walker(speed=0.0, sway=0.02)
    t = trajectory_from_points(J, FPS, loop=False)
    assert t["model"] == "still"


def _chain(*pairs):
    return {n: p for n, p in pairs}


def test_roles_read_every_leg_convention():
    soma = _chain(("Hips", None), ("Spine1", "Hips"), ("Chest", "Spine1"), ("Neck1", "Chest"), ("Head", "Neck1"),
                  ("LeftLeg", "Hips"), ("LeftShin", "LeftLeg"), ("LeftFoot", "LeftShin"), ("LeftToeBase", "LeftFoot"))
    core27 = _chain(("Hips", None), ("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"), ("Spine3", "Spine2"),
                    ("Neck", "Spine3"), ("Head", "Neck"),
                    ("LeftUpLeg", "Hips"), ("LeftLeg", "LeftUpLeg"), ("LeftFoot", "LeftLeg"), ("LeftToeBase", "LeftFoot"))
    unreal = _chain(("pelvis", None), ("spine_01", "pelvis"), ("spine_02", "spine_01"), ("neck_01", "spine_02"),
                    ("head", "neck_01"), ("thigh_l", "pelvis"), ("calf_l", "thigh_l"), ("foot_l", "calf_l"),
                    ("ball_l", "foot_l"))
    rigify = _chain(("DEF-pelvis", None), ("DEF-thigh.L", "DEF-pelvis"), ("DEF-shin.L", "DEF-thigh.L"),
                    ("DEF-foot.L", "DEF-shin.L"), ("DEF-toe.L", "DEF-foot.L"))
    for rig, thigh, shin, toe in ((soma, "LeftLeg", "LeftShin", "LeftToeBase"),
                                  (core27, "LeftUpLeg", "LeftLeg", "LeftToeBase"),
                                  (unreal, "thigh_l", "calf_l", "ball_l"),
                                  (rigify, "DEF-thigh.L", "DEF-shin.L", "DEF-toe.L")):
        names = list(rig)
        roles = resolve_roles(names, rig)
        assert names[roles["l_leg"]] == thigh and names[roles["l_shin"]] == shin and names[roles["l_toe"]] == toe
        assert roles["hips"] == 0
    roles = resolve_roles(list(core27), core27)
    assert list(core27)[roles["neck"]] == "Neck" and list(core27)[roles["chest"]] == "Spine3"


def test_roles_through_the_retargeting_map():
    rig = _chain(("b0", None), ("b1", "b0"), ("b2", "b1"), ("b3", "b2"))
    roles = resolve_roles(list(rig), rig, {"Hips": "b0", "LeftLeg": "b1", "LeftShin": "b2", "LeftFoot": "b3"})
    assert roles["hips"] == 0 and roles["l_leg"] == 1 and roles["l_shin"] == 2 and roles["l_foot"] == 3


def _server(backbone=None):
    """A test client on the SDK server; skipped where only the client is installed."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from motionmcp import build_app
    from motionmcp.null_backbone import NullBackbone

    return TestClient(build_app(backbone or NullBackbone()))


def test_server_advertises_and_returns_a_trajectory():
    client = _server()
    model = client.get("/capabilities").json()["models"][0]
    assert model["supports_trajectory"] is True
    body = {
        "protocol_version": "1.0",
        "model": "null",
        "skeleton": model["canonical_skeleton"],
        "segments": [{"type": "text", "prompt": "stand", "duration_frames": 20}],
    }
    r = client.post("/generate", json=body)
    assert r.status_code == 200, r.text
    sample = r.json()["extensions"]["MMCP_motion"]["samples"][0]
    traj = sample["trajectory"]
    assert traj["model"] == "still"                    # the null model stands still
    assert len(traj["position"]) == len(traj["yaw"]) == sample["num_frames"]


def test_client_parser_hands_on_the_trajectory():
    from motionmcp.client.gltf_parser import parse_gltf_samples

    client = _server()
    model = client.get("/capabilities").json()["models"][0]
    doc = client.post("/generate", json={
        "protocol_version": "1.0", "model": "null", "skeleton": model["canonical_skeleton"],
        "segments": [{"type": "text", "prompt": "stand", "duration_frames": 12}],
    }).json()
    (motion,) = parse_gltf_samples(doc)
    assert motion["trajectory"]["model"] == "still"


def _looping_walker():
    """Rest pose carried forward 1 m/s along +Z, swaying: a loop's travel."""
    from motionmcp.null_backbone import NullBackbone

    class LoopingWalker(NullBackbone):
        def capabilities(self):
            return super().capabilities().model_copy(update={"supports_loop": True})

        async def generate(self, request):
            result = await super().generate(request)
            T = result.root_translations.shape[1]
            t = np.arange(T) / FPS
            root = result.root_translations.copy()
            root[:, :, 0] += 0.04 * np.sin(2 * np.pi * t)            # sway
            root[:, :, 2] += 1.0 * t                                    # travel
            result.root_translations = root
            return result

    return LoopingWalker()


def test_a_loop_closes_on_its_roots_travel_through_the_server():
    pytest.importorskip("fastapi")
    client = _server(_looping_walker())
    model = client.get("/capabilities").json()["models"][0]
    doc = client.post("/generate", json={
        "protocol_version": "1.0", "model": "null", "skeleton": model["canonical_skeleton"],
        "segments": [{"type": "text", "prompt": "walk", "duration_frames": 31}],
        "options": {"loop": True},
    }).json()
    traj = doc["extensions"]["MMCP_motion"]["samples"][0]["trajectory"]
    assert traj["loop"] is True and traj["model"] in ("line", "arc", "still")
    p = np.array(traj["position"])
    assert np.allclose(p[-1] - p[0], [0.0, 1.0], atol=1e-3)         # exactly the cycle's travel
    assert np.abs(p[:, 0] - p[0, 0]).max() < 0.01                   # and not its sway


def test_a_stop_and_turn_back_is_still_checked_along_the_path():
    from motionmcp.trajectory import _split_error

    path = np.array([[0, 0], [1, 0], [2, 0], [2, 0], [2, 0], [1, 0], [0, 0]], float)
    xy = path.copy()
    xy[3] += [0.3, 0.0]                        # 30 cm off, at the stationary frame
    along, _ = _split_error(xy, path)
    assert abs(along[3]) > 0.29
