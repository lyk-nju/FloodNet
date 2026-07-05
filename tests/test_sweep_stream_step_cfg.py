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


def test_write_csv_includes_best_of_k_runtime_metrics(tmp_path):
    csv_path = tmp_path / "results.csv"
    row = {
        "cfg_text": 1.25,
        "cfg_traj": 3.0,
        "ade_mean": 0.1,
        "ade_std": 0.0,
        "stream_best_of_k_switch_count_mean": 2.0,
        "stream_best_of_k_step_count_mean": 46.0,
        "stream_best_of_k_total_elapsed_sec_mean": 4.5,
        "best_of_k_debug": True,
    }

    sweep_stream_step_cfg._write_csv(csv_path, [row])

    text = csv_path.read_text()
    assert "stream_best_of_k_switch_count_mean" in text
    assert "stream_best_of_k_step_count_mean" in text
    assert "stream_best_of_k_total_elapsed_sec_mean" in text
    assert "best_of_k_debug" in text
