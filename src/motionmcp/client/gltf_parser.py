# SPDX-License-Identifier: Apache-2.0
"""glTF 2.0 → motion_data parser for the MMCP server response.

``client.generate()`` returns a parsed glTF 2.0 JSON document (Python dict).
``parse_gltf`` extracts the skeleton animation tracks and returns the standard
``motion_data`` dict the caller applies to its own scene.

glTF animation conventions used by the MMCP server:
  - Each ``animations[]`` entry is one sample (variant): with ``num_samples > 1``
    the server returns N entries, all sharing the same node/channel structure.
    ``parse_gltf`` returns the first; ``parse_gltf_samples`` returns all N.
  - ``channels[*].target.path == "rotation"``   → XYZW quaternions, local space.
  - ``channels[*].target.path == "translation"`` → XYZ meters. On the root it
    is the world position; on any other joint (MMCP 1.4, from a model that
    advertises ``supports_joint_translation``) it is the joint's offset from
    its parent, in the parent's frame.
  - Every sampler is ``LINEAR`` and has its own ``input`` accessor. Since MMCP
    1.4 a channel may be keyed on some frames only; it is resampled onto the
    sample's whole timeline (slerp for rotations), which is what a glTF reader
    plays back, so every track comes out ``num_frames`` long.
  - Buffer data is embedded as base64 data URIs.
  - Node names are the SOMA-77 joint names.
  - The ``MMCP_motion`` extension pairs ``samples[i]`` with ``animations[i]``;
    ``samples[i].foot_contacts`` is a ``dict[str, list[bool]]`` keyed by joint
    name (verified live; absent from some older captures — never required).
  - ``nodes[].translation`` is the LOCAL rest offset in meters; world-space
    rest positions are accumulated down ``nodes[].children``, applying
    ``nodes[].rotation`` to child offsets when non-identity.

No external dependencies beyond numpy.
"""

import base64

import numpy as np

__all__ = ["parse_gltf", "parse_gltf_samples", "xyzw_to_rotmat"]


# glTF component-type enum → numpy dtype
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

# glTF accessor type → number of components per element
_TYPE_N = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_gltf(gltf_doc):
    """Parse a glTF 2.0 document and return the first sample's ``motion_data`` dict.

    *gltf_doc* is the Python dict returned by ``client.generate()``.
    When the server returns multiple samples (``num_samples > 1``) this returns
    only the first; use :func:`parse_gltf_samples` to get every sample.

    Returns:
        dict with keys:
            ``local_rot_mats`` – np.ndarray [T, J, 3, 3]
            ``posed_joints``   – np.ndarray [T, J, 3]  world positions, meters
                                 (the root's; zero for the other joints)
            ``local_translations`` – np.ndarray [T, J, 3]  each joint's offset
                                 from its parent, in the parent's frame: the
                                 channel where the server sent one, the rest
                                 offset where it did not; the root's is its
                                 world position
            ``fps``            – float
            ``num_frames``     – int
            ``num_joints``     – int
            ``joint_names``    – list[str]  SOMA-77 names in joint-index order
            ``foot_contacts``  – np.ndarray [T, J] bool, aligned to
                                 ``joint_names``; all-``False`` columns for
                                 joints the server did not report
            ``hierarchy``      – list[(joint, parent_or_None)]  joint *names*,
                                 in ``joint_names`` order
            ``rest_positions`` – dict[name → (x, y, z)]  world-space rest,
                                 meters (accumulated, not the local offsets)
    """
    return parse_gltf_samples(gltf_doc)[0]


def parse_gltf_samples(gltf_doc):
    """Parse a glTF 2.0 document into one ``motion_data`` dict per sample.

    The MMCP server encodes ``num_samples`` motion variants as N entries in the
    ``animations[]`` array, all sharing the same node/channel structure and
    buffers. Returns one ``motion_data`` dict (same shape as :func:`parse_gltf`)
    per entry, in ``animations[]`` order.

    Foot contacts and the travel trajectory (``motion_data["trajectory"]``,
    None when the server sent none) come from the ``MMCP_motion`` extension,
    whose ``samples[i]`` pairs with ``animations[i]``. Degrades to no contacts (never raises) when
    the block is absent or the lengths disagree — servers have shipped
    samples without a ``foot_contacts`` key, and a parser that refused them
    would lose the motion over metadata.

    Returns:
        list[dict] – one motion_data dict per ``animations[]`` entry.

    Raises:
        ValueError – if the document has no animations, or an animation has no
            rotation channels.
    """
    if not gltf_doc.get("animations"):
        raise ValueError("glTF document contains no animations")

    raw_buffers = _decode_buffers(gltf_doc.get("buffers", []))
    nodes       = gltf_doc.get("nodes", [])
    anims       = gltf_doc["animations"]

    ext = gltf_doc.get("extensions", {}).get("MMCP_motion", {})
    ext = ext if isinstance(ext, dict) else {}
    ext_fps = ext.get("fps")
    ext_samples = ext.get("samples", [])
    if not isinstance(ext_samples, list) or len(ext_samples) != len(anims):
        ext_samples = [{}] * len(anims)   # absent / mismatched → no contacts

    def read_acc(idx):
        return _read_accessor(gltf_doc, raw_buffers, idx)

    out = []
    for anim, sample in zip(anims, ext_samples):
        sample = sample if isinstance(sample, dict) else {}
        motion = _parse_animation(
            anim, nodes, read_acc,
            sample.get("foot_contacts"),
            clock=(ext_fps, sample.get("num_frames")),
        )
        # The travel trajectory, as the server sent it (glTF frame): None from
        # servers without supports_trajectory.
        motion["trajectory"] = sample.get("trajectory")
        out.append(motion)
    return out


def _parse_animation(anim, nodes, read_acc, contacts=None, clock=(None, None)):
    """Parse one ``animations[]`` entry into a ``motion_data`` dict.

    *contacts* is the sample's raw ``foot_contacts`` wire dict
    (``dict[str, list[bool]]`` keyed by joint name) or ``None``.
    *clock* is ``(fps, num_frames)`` from the ``MMCP_motion`` extension, either
    possibly ``None``; the timeline every track is resampled onto.
    """
    rot_tracks   = {}  # node_idx -> (times [T], xyzw [T, 4])
    trans_tracks = {}  # node_idx -> (times [T], xyz  [T, 3])

    for ch in anim["channels"]:
        s  = anim["samplers"][ch["sampler"]]
        ni = ch["target"]["node"]
        times  = read_acc(s["input"]).ravel().astype(np.float32)
        values = read_acc(s["output"])
        path   = ch["target"]["path"]
        if path == "rotation":
            rot_tracks[ni] = (times, values.astype(np.float32))
        elif path == "translation":
            trans_tracks[ni] = (times, values.astype(np.float32))

    if not rot_tracks:
        raise ValueError("glTF animation has no rotation channels")

    sorted_nodes = sorted(rot_tracks)
    num_joints   = len(sorted_nodes)
    grid, fps    = _timeline(clock, [t for t, _ in (*rot_tracks.values(),
                                                    *trans_tracks.values())])
    num_frames   = int(grid.shape[0])
    rot_tracks   = {ni: _slerp_onto(t, q, grid) for ni, (t, q) in rot_tracks.items()}
    trans_tracks = {ni: _lerp_onto(t, v, grid) for ni, (t, v) in trans_tracks.items()}

    joint_names = [
        nodes[ni]["name"] if ni < len(nodes) else str(ni)
        for ni in sorted_nodes
    ]

    local_rot_mats     = np.zeros((num_frames, num_joints, 3, 3), dtype=np.float32)
    posed_joints       = np.zeros((num_frames, num_joints, 3),    dtype=np.float32)
    local_translations = np.zeros((num_frames, num_joints, 3),    dtype=np.float32)

    # The root is the joint no node names as a child. Only its translation is a
    # world position; any other joint's is an offset from its parent, which
    # `posed_joints` (world positions) must not be handed.
    children = {c for n in nodes for c in n.get("children", ())}
    for j, ni in enumerate(sorted_nodes):
        local_rot_mats[:, j] = xyzw_to_rotmat(rot_tracks[ni])
        rest = nodes[ni].get("translation", (0.0, 0.0, 0.0)) if ni < len(nodes) else (0, 0, 0)
        local_translations[:, j] = trans_tracks.get(ni, np.asarray(rest, dtype=np.float32))
        if ni in trans_tracks and ni not in children:
            posed_joints[:, j] = trans_tracks[ni]

    # Foot contacts: mapped BY NAME from the wire dict into joint_names order,
    # never by position — the full-joint bool axis avoids inventing a second
    # column-ordering convention. Unknown names / length mismatches degrade to
    # an all-False column, never raise.
    foot_contacts = np.zeros((num_frames, num_joints), dtype=bool)
    if isinstance(contacts, dict):
        col = {name: j for j, name in enumerate(joint_names)}
        for name, flags in contacts.items():
            j = col.get(name)
            if j is None or not isinstance(flags, (list, tuple)) \
                    or len(flags) != num_frames:
                continue
            foot_contacts[:, j] = np.asarray(flags, dtype=bool)

    hierarchy, rest_positions = _node_rest_geometry(nodes, sorted_nodes, joint_names)

    return {
        "local_rot_mats": local_rot_mats,
        "posed_joints":   posed_joints,
        "local_translations": local_translations,
        "fps":            fps,
        "num_frames":     num_frames,
        "num_joints":     num_joints,
        "joint_names":    joint_names,
        "foot_contacts":  foot_contacts,
        "hierarchy":      hierarchy,
        "rest_positions": rest_positions,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _timeline(clock, track_times):
    """The frame times every track is put on, and the fps they imply.

    The extension states the clock outright, and is trusted when it does. A
    document without it (a pre-1.4 capture, a hand-built one) is keyed on every
    frame, so its longest input is the clock; fps is inferred from its steps,
    30 when that is ambiguous.
    """
    fps, num_frames = clock
    if isinstance(fps, (int, float)) and fps > 0 \
            and isinstance(num_frames, int) and num_frames > 0:
        return np.arange(num_frames, dtype=np.float32) / np.float32(fps), float(fps)
    grid = max(track_times, key=len)
    if len(grid) > 1:
        dt = float(grid[-1] - grid[0]) / (len(grid) - 1)
        return grid, (1.0 / dt if dt > 1e-9 else 30.0)
    return grid, 30.0


def _same_times(times, grid):
    return times.shape == grid.shape and np.allclose(times, grid, atol=1e-6)


def _lerp_onto(times, values, grid):
    """A LINEAR track at *grid*: held before its first key and after its last."""
    values = np.asarray(values, dtype=np.float32)
    if _same_times(times, grid):
        return values
    return np.stack([np.interp(grid, times, values[:, k])
                     for k in range(values.shape[1])], axis=1).astype(np.float32)


def _slerp_onto(times, quats, grid):
    """A LINEAR rotation track at *grid*: glTF interpolates rotations by slerp,
    along the shorter arc, and holds them outside the keyed range."""
    quats = np.asarray(quats, dtype=np.float32)
    if _same_times(times, grid):
        return quats
    if len(times) == 1:
        return np.repeat(quats, len(grid), axis=0)
    g = np.clip(grid, times[0], times[-1])
    i = np.clip(np.searchsorted(times, g, side="right") - 1, 0, len(times) - 2)
    span = times[i + 1] - times[i]
    u = ((g - times[i]) / np.where(span > 0, span, 1.0)).astype(np.float64)[:, None]
    q0 = quats[i].astype(np.float64)
    q1 = quats[i + 1].astype(np.float64)
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sin = np.sin(theta)
    near = sin < 1e-6                    # (nearly) the same rotation: lerp is exact
    safe = np.where(near, 1.0, sin)
    w0 = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / safe)
    w1 = np.where(near, u, np.sin(u * theta) / safe)
    q = w0 * q0 + w1 * q1
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def _node_rest_geometry(nodes, sorted_nodes, joint_names):
    """Joint hierarchy + world-space rest positions from the glTF ``nodes[]``.

    ``nodes[].translation`` is the *local* rest offset (meters), so world-space
    rest positions are accumulated down the ``children`` graph, applying each
    node's rest ``rotation`` to the child offset when it is non-identity (all
    nine in-repo captures carry identity rest rotations, but the accumulation
    must not assume it). ``children`` is in *node*-index space; joint index
    ``j`` maps to node ``sorted_nodes[j]``, and hierarchy parents are joint
    *names* — a parent node that is not itself animated is walked through to
    the nearest animated ancestor, the root mapping to ``None``.

    Returns:
        ``(hierarchy, rest_positions)`` in the ``motion_data`` shape
        :func:`parse_gltf` documents — ``list[tuple[str, str | None]]`` in
        ``joint_names`` order,
        and ``dict[str, tuple[float, float, float]]`` world-space meters.
    """
    parent_of = {}
    for pi, node in enumerate(nodes):
        for ci in node.get("children", []):
            if isinstance(ci, int) and 0 <= ci < len(nodes):
                parent_of[ci] = pi

    world_pos = {}   # node_idx -> np.ndarray [3]   world rest position
    world_rot = {}   # node_idx -> np.ndarray [3,3] world rest rotation

    def accumulate(ni):
        """Memoised root-to-leaf accumulation for node *ni* and its ancestors."""
        chain = []
        while ni is not None and ni not in world_pos:
            chain.append(ni)
            ni = parent_of.get(ni)
            if ni in chain:          # cycle guard: sever, treat as root
                ni = None
        for ci in reversed(chain):
            node = nodes[ci]
            t = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
            q = np.asarray([node.get("rotation", (0.0, 0.0, 0.0, 1.0))], dtype=np.float64)
            R = xyzw_to_rotmat(q)[0].astype(np.float64)
            pi = parent_of.get(ci)
            if pi is None or pi not in world_pos:
                world_pos[ci] = t
                world_rot[ci] = R
            else:
                world_pos[ci] = world_pos[pi] + world_rot[pi] @ t
                world_rot[ci] = world_rot[pi] @ R

    node_joint = {ni: joint_names[j] for j, ni in enumerate(sorted_nodes)}

    hierarchy      = []
    rest_positions = {}
    for j, ni in enumerate(sorted_nodes):
        name = joint_names[j]
        if not (0 <= ni < len(nodes)):
            hierarchy.append((name, None))
            rest_positions[name] = (0.0, 0.0, 0.0)
            continue
        accumulate(ni)
        pi   = parent_of.get(ni)
        seen = set()
        while pi is not None and pi not in node_joint and pi not in seen:
            seen.add(pi)
            pi = parent_of.get(pi)
        hierarchy.append((name, node_joint.get(pi)))
        rest_positions[name] = tuple(float(v) for v in world_pos[ni])
    return hierarchy, rest_positions


def _decode_buffers(buffers):
    """Return a list of ``bytes`` objects, one per glTF buffer."""
    raw = []
    for buf in buffers:
        uri = buf.get("uri", "")
        if uri.startswith("data:"):
            _, data_part = uri.split(",", 1)
            raw.append(base64.b64decode(data_part))
        else:
            raw.append(b"")  # external URIs not supported in this context
    return raw


def _read_accessor(gltf_doc, raw_buffers, idx):
    """Decode accessor *idx* and return a numpy array."""
    acc  = gltf_doc["accessors"][idx]
    bv   = gltf_doc["bufferViews"][acc["bufferView"]]
    buf  = raw_buffers[bv["buffer"]]

    byte_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count       = acc["count"]
    dtype       = _COMPONENT_DTYPE[acc["componentType"]]
    n_comp      = _TYPE_N[acc["type"]]

    item_bytes = np.dtype(dtype).itemsize * n_comp
    raw        = buf[byte_offset: byte_offset + count * item_bytes]
    arr        = np.frombuffer(raw, dtype=dtype)
    if n_comp > 1:
        arr = arr.reshape(count, n_comp)
    return arr


def xyzw_to_rotmat(xyzw):
    """Convert [N, 4] XYZW quaternions to [N, 3, 3] rotation matrices."""
    x = xyzw[:, 0].astype(np.float64)
    y = xyzw[:, 1].astype(np.float64)
    z = xyzw[:, 2].astype(np.float64)
    w = xyzw[:, 3].astype(np.float64)

    R = np.empty((len(xyzw), 3, 3), dtype=np.float32)
    R[:, 0, 0] = (1 - 2*(y*y + z*z)).astype(np.float32)
    R[:, 0, 1] = (2*(x*y - z*w)).astype(np.float32)
    R[:, 0, 2] = (2*(x*z + y*w)).astype(np.float32)
    R[:, 1, 0] = (2*(x*y + z*w)).astype(np.float32)
    R[:, 1, 1] = (1 - 2*(x*x + z*z)).astype(np.float32)
    R[:, 1, 2] = (2*(y*z - x*w)).astype(np.float32)
    R[:, 2, 0] = (2*(x*z - y*w)).astype(np.float32)
    R[:, 2, 1] = (2*(y*z + x*w)).astype(np.float32)
    R[:, 2, 2] = (1 - 2*(x*x + y*y)).astype(np.float32)
    return R
