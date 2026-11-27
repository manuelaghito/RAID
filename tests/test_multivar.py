## python3 -m pytest tests/test_multivar.py
## several target variables in one model (e.g. u and v): one output channel per target, each a
## correction to the input channel of the same variable; evaluate.py scores each in eval_<split>/<var>/

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from raid.config import dict_to_namespace
from raid.data.file_utils import daily_filename
from raid.losses import masked_mse
from raid.models import build_model
from raid.models.cnn import SimpleSRCNN
from raid.models.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (40, 64)


@pytest.mark.parametrize("model", [SimpleSRCNN(in_channels=5, hidden_dim=8, n_blocks=1, out_channels=2),
                                   UNet(in_channels=5, base_channels=8, depth=2, out_channels=2)])
def test_two_outputs_start_as_their_own_input_channels(model):
    x = torch.randn(2, 5, 32, 48)
    out = model(x)
    assert out.shape == (2, 2, 32, 48)
    torch.testing.assert_close(out, x[:, :2])


def test_masked_mse_averages_over_channels_and_sea_cells():
    pred = torch.zeros(1, 2, 2, 2)
    target = torch.stack([torch.full((2, 2), 1.0), torch.full((2, 2), 3.0)])[None]
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])          # 3 sea cells
    ## channel errors 1 and 9 on every sea cell -> mean 5
    assert masked_mse(pred, target, mask).item() == pytest.approx(5.0)


def _cfg(inputs, targets):
    cfg = yaml.safe_load((ROOT / "configs" / "step1b_unet.yaml").read_text())
    cfg["data"].update(input_vars=inputs, target_vars=targets)
    return dict_to_namespace(cfg)


def test_build_model_has_one_output_per_target():
    model = build_model(_cfg(["u", "v"], ["u", "v"]))
    assert model.output_conv.out_channels == 2


def test_build_model_rejects_inputs_not_starting_with_targets():
    with pytest.raises(ValueError, match="must start with target_vars"):
        build_model(_cfg(["v", "u"], ["u", "v"]))


def _fake_uv(tmp_path, dates):
    root = tmp_path / "processed"
    for sub in ("800m_on_160m_grid", "160m", "static"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    land = np.ones(SHAPE, np.uint8)
    land[:5, :10] = 0
    ## v is ~10x larger than u, so swapping them would show in the metrics
    scale = {"u": 0.1, "v": 1.0}
    for d in dates:
        for v, sc in scale.items():
            t = (sc * rng.normal(size=SHAPE)).astype(np.float32)
            t[land == 0] = np.nan
            np.save(root / "160m" / daily_filename(v, d), t)
            np.save(root / "800m_on_160m_grid" / daily_filename(v, d),
                    t + 0.3 * sc * rng.normal(size=SHAPE).astype(np.float32))
    np.save(root / "static" / "land_mask.npy", land)
    for name in ("bathymetry", "distance_to_coast"):
        np.save(root / "static" / f"{name}.npy", rng.uniform(1, 100, SHAPE).astype(np.float32))

    cfg = yaml.safe_load((ROOT / "configs" / "step1b_unet_uv.yaml").read_text())
    cfg["data"]["processed_dir"] = str(root)
    cfg["train"]["num_workers"] = 0
    cfg["model"].update(base_channels=8, depth=2)
    run = tmp_path / "run"
    cfg["output_dir"] = str(run)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path, root, run, land, scale


def test_train_two_targets_runs_and_saves_config(tmp_path):
    cfg_path, _, run, _, _ = _fake_uv(tmp_path, ["20231201", "20231202", "20240301", "20250301"])
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["train"].update(epochs=2, batch_size=2, patch_size=16)
    cfg_path.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
                    "--fold", "val2024"], check=True)
    out = run / "u_v" / "val2024"
    assert (out / "best.pt").exists()
    stats = np.load(out / "norm_stats.npz")
    assert {"u_mean", "u_std", "v_mean", "v_std"} <= set(stats.files)
    assert "epoch 1: train" in (out / "train.log").read_text()


def test_evaluate_two_targets_writes_one_folder_per_variable(tmp_path):
    ## untrained model = baseline for both variables, and u/v must not get mixed up
    dates = ["20260101", "20260102", "20260103"]
    cfg_path, root, run, land, scale = _fake_uv(tmp_path, dates)
    run.mkdir()
    cfg = yaml.safe_load(cfg_path.read_text())
    np.savez(run / "norm_stats.npz", u_mean=0.0, u_std=0.1, v_mean=0.0, v_std=1.0)
    torch.save(build_model(dict_to_namespace(cfg)).state_dict(), run / "best.pt")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(cfg_path)], check=True)

    out = run / "eval_test"
    assert "[u]" in (out / "evaluate.log").read_text() and "[v]" in (out / "evaluate.log").read_text()
    for v, sc in scale.items():
        daily = np.load(out / v / "daily_metrics.npy")
        assert daily.shape == (len(dates), 4)
        np.testing.assert_allclose(daily[:, 0], daily[:, 1], rtol=1e-5)         # model == baseline
        assert 0.2 * sc < daily[0, 1] < 0.4 * sc                                  # this variable's error size
        pred = np.load(out / v / f"pred_{v}.npy")
        base = np.load(root / "800m_on_160m_grid" / daily_filename(v, dates[0]))
        np.testing.assert_allclose(pred[0][land == 1], base[land == 1], rtol=1e-4, atol=1e-5)
        assert (out / v / "monthly_metrics.csv").exists() and (out / v / f"rmse_map_model_{v}.npy").exists()
