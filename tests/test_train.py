## python3 -m pytest tests/test_train.py
## end-to-end smoke test of scripts/train.py --fold full on a few fake days

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import build_dataloaders, trim_edges

from test_evaluate import make_fake_setup

ROOT = Path(__file__).resolve().parents[1]


def test_train_full_uses_all_non_test_years_and_saves_last_epoch(tmp_path):
    ## a day in every year, so a wrong year selection would show in norm_stats / the log
    dates = ["20231201", "20240301", "20250301", "20260301"]
    cfg_path, _ = make_fake_setup(tmp_path, dates, tmp_path / "unused")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["train"].update(epochs=2, batch_size=2, patch_size=None)
    cfg_path.write_text(yaml.safe_dump(cfg))

    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
                    "--fold", "full"], check=True)

    run = tmp_path / "run" / "temp" / "full"
    assert (run / "best.pt").exists()
    log = (run / "train.log").read_text()
    assert "train [2023, 2024, 2025]  val []" in log
    assert "epoch 1: train" in log and " val " not in log.split("epoch 0:")[1]

    ## stats from the three training days only, not the 2026 test day
    stats = np.load(run / "norm_stats.npz")
    root = Path(cfg["data"]["processed_dir"]) / "160m"
    sea = np.load(Path(cfg["data"]["processed_dir"]) / "static" / "land_mask.npy").astype(bool)
    train = np.concatenate([np.load(root / f"{d}_temp.npy")[sea] for d in dates[:3]])
    np.testing.assert_allclose(stats["temp_mean"], train.mean(), rtol=1e-6)


def _train(cfg_path, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path), *args],
                   check=True)


def _fake_run(tmp_path, **train):
    dates = ["20231201", "20231202", "20240301", "20250301"]
    cfg_path, _ = make_fake_setup(tmp_path, dates, tmp_path / "unused")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["train"].update(epochs=1, batch_size=2, patch_size=16, **train)
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_same_seed_gives_identical_weights_and_other_seed_differs(tmp_path):
    ## random crops (patch_size) + shuffling + model init all have to be pinned by the seed
    cfg_path = _fake_run(tmp_path, seed=3)
    out = tmp_path / "out"

    _train(cfg_path, "--fold", "full", "--output_dir", out / "a")
    _train(cfg_path, "--fold", "full", "--output_dir", out / "b")
    _train(cfg_path, "--fold", "full", "--output_dir", out / "c", "--seed", "4", "--train_edge_pixels", "3")

    a, b, c = (torch.load(out / r / "temp" / "full" / "best.pt") for r in "abc")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)

    ## the resolved config is saved with the run, incl. the --seed override
    saved = yaml.safe_load((out / "c" / "temp" / "full" / "config.yaml").read_text())
    assert saved["train"]["seed"] == 4 and saved["train"]["edge_pixels"] == 3
    assert saved["split"]["train"]["years"] == [2023, 2024, 2025]
    assert "seed 4  epochs 1" in (out / "c" / "temp" / "full" / "train.log").read_text()


def test_val_edge_pixels_is_independent_of_edge_pixels(tmp_path):
    cfg_path = _fake_run(tmp_path, edge_pixels=5, val_edge_pixels=2)
    raw = yaml.safe_load(cfg_path.read_text())
    apply_fold(raw, "val2024")
    train_loader, val_loader = build_dataloaders(dict_to_namespace(raw))

    sea = train_loader.dataset.sea_mask
    torch.testing.assert_close(train_loader.dataset.loss_mask[0], trim_edges(sea, 5).float())
    torch.testing.assert_close(val_loader.dataset.loss_mask[0], trim_edges(sea, 2).float())


def test_val_edge_pixels_defaults_to_all_sea_cells(tmp_path):
    cfg_path = _fake_run(tmp_path, edge_pixels=5)
    raw = yaml.safe_load(cfg_path.read_text())
    del raw["train"]["val_edge_pixels"]
    apply_fold(raw, "val2024")
    _, val_loader = build_dataloaders(dict_to_namespace(raw))

    torch.testing.assert_close(val_loader.dataset.loss_mask[0], val_loader.dataset.sea_mask.float())
