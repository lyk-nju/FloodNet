"""Checkpoint compatibility shims for the flag-gated 4D→7D traj migration (T_B_08).

`expand_traj_input_4d_to_7d` lets an old 4D-era checkpoint load into a 7D-encoder
model. It is a no-op unless the target model is 7D (`target_in_dim == 7`), so the
4D path is unaffected.

Trajectory encoder shape changes (in_dim 4 → 7):
  local_traj_encoder.net.0.weight : Conv1d in-channels  4→7  (RAW traj feature)
  local_traj_encoder.net.2.weight : Conv1d out-channels 4→7  (internal feature)
  local_traj_encoder.net.2.bias   : 4→7                       (internal feature)
  traj_encoder.mlp.0.weight       : Linear in-features  4→7  (internal feature)

Two different expansion rules, by what the axis MEANS:

* RAW-feature input axis (net.0 in-channels): the channels ARE the trajectory
  feature layout. Old 4D `[x, z, legacy_h0, legacy_h1]` → new 7D
  `[x, y, z, cos, sin, fwd, yaw]`. Only x/z are carried (old 0→new 0, old 1→new 2);
  the legacy heading channels are DROPPED (NOT mapped onto the new physical-yaw
  heading — different semantics, see T_A_01 legacy-heading analysis), and
  y/cos/sin/fwd/yaw zero-init.

* INTERNAL feature axis (net.2 out-channels and mlp.0 in-features — these are the
  local encoder's *learned* compressed channels, not raw x/z): keep the old 4
  channels in place and append 3 zero channels. This preserves the old
  computation EXACTLY, so a 7D input `[x, 0, z, 0, 0, 0, 0]` reproduces the old
  4D forward on `[x, z, 0, 0]` (Done-criterion #3), while the new channels start
  at zero and become learnable during fine-tune.
"""

from __future__ import annotations

import torch

# RAW-feature semantic map: old 4D [x, z, lh0, lh1] -> new 7D [x, y, z, ...]
_RAW_SRC = [0, 1]   # old channels x, z
_RAW_DST = [0, 2]   # new positions x, z (old 2,3 dropped; new 1,3,4,5,6 zero)


def expand_traj_input_4d_to_7d(state_dict, target_in_dim: int) -> int:
    """In-place expand 4D traj-encoder weights to 7D in `state_dict`.

    No-op unless `target_in_dim == 7`. Returns the number of tensors expanded.
    Matches keys by suffix so it works with or without a module prefix.
    """
    if target_in_dim != 7:
        return 0
    n = 0
    for key in list(state_dict.keys()):
        w = state_dict[key]
        if not torch.is_tensor(w):
            continue

        # RAW input axis (Conv1d in-channels) → semantic x/z map.
        if key.endswith("local_traj_encoder.net.0.weight") and w.dim() == 3 and w.shape[1] == 4:
            nw = w.new_zeros(w.shape[0], 7, w.shape[2])
            nw[:, _RAW_DST, :] = w[:, _RAW_SRC, :]
            state_dict[key] = nw
            n += 1

        # INTERNAL output axis (Conv1d out-channels) → in-place + zero pad.
        elif key.endswith("local_traj_encoder.net.2.weight") and w.dim() == 3 and w.shape[0] == 4:
            nw = w.new_zeros(7, w.shape[1], w.shape[2])
            nw[:4] = w
            state_dict[key] = nw
            n += 1
        elif key.endswith("local_traj_encoder.net.2.bias") and w.dim() == 1 and w.shape[0] == 4:
            nb = w.new_zeros(7)
            nb[:4] = w
            state_dict[key] = nb
            n += 1

        # INTERNAL input axis (Linear in-features = local encoder output) → in-place + zero pad.
        elif key.endswith("traj_encoder.mlp.0.weight") and w.dim() == 2 and w.shape[1] == 4:
            nw = w.new_zeros(w.shape[0], 7)
            nw[:, :4] = w
            state_dict[key] = nw
            n += 1

    return n


__all__ = ["expand_traj_input_4d_to_7d"]
