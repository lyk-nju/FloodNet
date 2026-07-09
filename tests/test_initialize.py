from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from utils.initialize import save_config_and_codes


def test_save_config_and_codes_prunes_outputs_before_scanning(tmp_path, monkeypatch):
    (tmp_path / "train.py").write_text("print('train')\n")
    outputs_script = tmp_path / "outputs" / "old_run" / "sanity_check" / "old.py"
    outputs_script.parent.mkdir(parents=True)
    outputs_script.write_text("print('old')\n")

    def fail_rglob(self, pattern):
        raise FileNotFoundError("outputs traversal should be pruned before scan")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "rglob", fail_rglob)
    cfg = SimpleNamespace(exp_name="ldf", config=OmegaConf.create({"exp_name": "ldf"}))

    save_config_and_codes(cfg, tmp_path / "outputs" / "new_run")

    sanity_dir = tmp_path / "outputs" / "new_run" / "sanity_check"
    assert (sanity_dir / "ldf.yaml").exists()
    assert (sanity_dir / "train.py").exists()
    assert not (sanity_dir / "outputs" / "old_run" / "sanity_check" / "old.py").exists()


def test_save_config_and_codes_prunes_eval_output_snapshots(tmp_path, monkeypatch):
    (tmp_path / "train_refiner.py").write_text("print('train')\n")
    old_snapshot = (
        tmp_path
        / "eval"
        / "output_eval"
        / "old_run"
        / "sanity_check"
        / "nested.py"
    )
    old_snapshot.parent.mkdir(parents=True)
    old_snapshot.write_text("print('nested')\n")

    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(
        exp_name="root_refiner",
        config=OmegaConf.create({"exp_name": "root_refiner"}),
    )

    save_config_and_codes(cfg, tmp_path / "outputs" / "new_run")

    sanity_dir = tmp_path / "outputs" / "new_run" / "sanity_check"
    assert (sanity_dir / "root_refiner.yaml").exists()
    assert (sanity_dir / "train_refiner.py").exists()
    assert not (
        sanity_dir
        / "eval"
        / "output_eval"
        / "old_run"
        / "sanity_check"
        / "nested.py"
    ).exists()
