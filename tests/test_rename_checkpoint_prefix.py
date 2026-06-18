from __future__ import annotations

import torch

from tools.rename_checkpoint_prefix import rename_checkpoint_keys


def test_rename_checkpoint_prefix_writes_updated_state_dict(tmp_path):
    ckpt = tmp_path / "input.ckpt"
    out = tmp_path / "output.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.weight": torch.ones(1),
                "other.bias": torch.zeros(1),
            },
            "epoch": 3,
        },
        ckpt,
    )

    renamed = rename_checkpoint_keys(
        ckpt,
        out,
        old_prefix="model.",
        new_prefix="net.",
    )

    payload = torch.load(out, map_location="cpu")
    assert renamed == 1
    assert payload["epoch"] == 3
    assert "net.encoder.weight" in payload["state_dict"]
    assert "model.encoder.weight" not in payload["state_dict"]
    assert "other.bias" in payload["state_dict"]


def test_rename_checkpoint_prefix_dry_run_does_not_write(tmp_path):
    ckpt = tmp_path / "input.ckpt"
    out = tmp_path / "output.ckpt"
    torch.save({"state_dict": {"module.weight": torch.ones(1)}}, ckpt)

    renamed = rename_checkpoint_keys(
        ckpt,
        out,
        old_prefix="module.",
        dry_run=True,
    )

    assert renamed == 1
    assert not out.exists()
