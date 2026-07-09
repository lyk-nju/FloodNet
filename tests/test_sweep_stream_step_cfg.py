import pytest

from tools import sweep_stream_step_cfg


def test_apply_caption_index_selects_requested_caption():
    sample_batch = {
        "text": ["old caption"],
        "text_all": [["caption one", "caption two", "caption three"]],
    }

    selected = sweep_stream_step_cfg._apply_caption_index(sample_batch, 1)

    assert selected is sample_batch
    assert sample_batch["text"] == ["caption two"]
    assert sample_batch["_caption_index"] == 1
    assert sample_batch["_caption_text"] == "caption two"


def test_apply_caption_index_rejects_out_of_range_caption():
    sample_batch = {"text_all": [["caption one"]]}

    with pytest.raises(ValueError, match="caption_index"):
        sweep_stream_step_cfg._apply_caption_index(sample_batch, 3)


def test_parse_cfg_values_accepts_explicit_comma_list():
    values = sweep_stream_step_cfg._parse_cfg_values("1.2,2.2,3.0")

    assert values == [1.2, 2.2, 3.0]


def test_root_feedback_xz_blend_alpha_has_hard_replace_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_stream_step_cfg.py",
            "--ckpt",
            "model.ckpt",
            "--meta_path",
            "test.txt",
            "--out_dir",
            "out",
        ],
    )

    args = sweep_stream_step_cfg._parse_args()

    assert args.root_feedback_xz_blend_alpha == 1.0

