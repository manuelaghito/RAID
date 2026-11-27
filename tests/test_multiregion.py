## multi-region training (data.processed_dir a list): pooled stats over all regions' training years,
## validation on the first region only, test years left out everywhere, evaluate.py --processed_dir
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import DownscalingDataset, build_dataloaders, compute_static_stats
from raid.data.file_utils import daily_filename
from raid.models import build_model

ROOT = Path(__file__).resolve().parents[1]
DATES = ["20231201", "20240301", "20240302", "20250301", "20260101", "20260102"]


def _region(root, shape, offset, depth_scale, seed, test_value=None):
    ## one region: u/v around `offset`, its own grid shape and bathymetry range; test_value (if given)
    ## fills the 2026 files, so any use of test years in training would show in the stats
    for sub in ("800m_on_160m_grid", "160m", "static"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(seed)
    land = np.ones(shape, np.uint8)
    land[:3, :4] = 0
    for d in DATES:
        for v in ("u", "v"):
            if test_value is not None and d.startswith("2026"):
                t = np.full(shape, test_value, np.float32)
            else:
                t = (offset + 0.1 * rng.normal(size=shape)).astype(np.float32)
            t[land == 0] = np.nan
            np.save(root / "160m" / daily_filename(v, d), t)
            np.save(root / "800m_on_160m_grid" / daily_filename(v, d),
                    (t + 0.03 * rng.normal(size=shape)).astype(np.float32))
    np.save(root / "static" / "land_mask.npy", land)
    np.save(root / "static" / "bathymetry.npy", rng.uniform(10, depth_scale, shape).astype(np.float32))
    np.save(root / "static" / "distance_to_coast.npy", rng.uniform(0, 20, shape).astype(np.float32))
    return land


def _setup(tmp_path, n_regions=2):
    a, b = tmp_path / "processed", tmp_path / "processed_B"
    _region(a, (24, 32), 0.0, 100, 0)
    _region(b, (40, 28), 1.0, 1000, 1, test_value=50.0)
    cfg = yaml.safe_load((ROOT / "configs" / "step1b_unet_uv_regions.yaml").read_text())
    cfg["data"]["processed_dir"] = [str(a), str(b)][:n_regions]
    cfg["train"].update(num_workers=0, epochs=1, batch_size=2, patch_size=16, crops_per_day=3)
    cfg["model"].update(base_channels=8, depth=2)
    cfg["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, a, b


def _pooled(roots, years, var="u", sub="160m"):
    vals = []
    for r in roots:
        sea = np.load(r / "static" / "land_mask.npy").astype(bool)
        for d in DATES:
            if int(d[:4]) in years:
                vals.append(np.load(r / sub / daily_filename(var, d))[sea].astype(np.float64))
    a = np.concatenate(vals)
    return a.mean(), a.std()


def test_pooled_stats_test_years_left_out_and_val_on_first_region(tmp_path):
    path, cfg, a, b = _setup(tmp_path)
    apply_fold(cfg, "val2024")
    c = dict_to_namespace(cfg)
    train_loader, val_loader = build_dataloaders(c)

    ## training: both regions, train years only (2023, 2025), crops_per_day crops per day in each
    parts = train_loader.dataset.datasets
    assert [p.root for p in parts] == [a, b]
    assert [p.dates for p in parts] == [["20231201", "20250301"]] * 2
    assert len(train_loader.dataset) == 2 * 2 * 3
    ## validation: first region, val year only
    assert val_loader.dataset.root == a and val_loader.dataset.dates == ["20240301", "20240302"]

    stats = np.load(Path(c.output_dir) / "norm_stats.npz")
    m, s = _pooled([a, b], {2023, 2025})
    np.testing.assert_allclose([stats["u_mean"], stats["u_std"]], [m, s], rtol=1e-6)
    assert stats["u_mean"] < 1.0           # the 2026 value 50 of region B never entered
    ## static fields standardised with the pooled stats in every region, saved for evaluation
    st = compute_static_stats([a, b], ["bathymetry", "distance_to_coast"])
    np.testing.assert_allclose(stats["static_bathymetry_mean"], st["bathymetry"][0])
    for p in parts + [val_loader.dataset]:
        sea = p.sea_mask.numpy()
        h = np.load(p.root / "static" / "bathymetry.npy")
        np.testing.assert_allclose(p.static[0].numpy()[sea], ((h - st["bathymetry"][0]) / st["bathymetry"][1])[sea],
                                   rtol=1e-5, atol=1e-5)

    ## crops from both grid shapes batch together
    x, y, mask = next(iter(train_loader))
    assert x.shape == (2, 5, 16, 16) and y.shape == (2, 2, 16, 16)


def test_single_region_unchanged(tmp_path):
    ## one region: own stats, static fields over its own sea cells, no static_* keys saved (as before)
    path, cfg, a, _ = _setup(tmp_path, n_regions=1)
    cfg["data"]["processed_dir"] = str(a)
    apply_fold(cfg, "val2024")
    c = dict_to_namespace(cfg)
    train_loader, _ = build_dataloaders(c)
    stats = np.load(Path(c.output_dir) / "norm_stats.npz")
    assert not any(k.startswith("static_") for k in stats.files)
    ds = train_loader.dataset
    sea = ds.sea_mask.numpy()
    h = np.load(a / "static" / "bathymetry.npy").astype(np.float32)
    np.testing.assert_array_equal(ds.static[0].numpy()[sea], ((h - h[sea].mean()) / h[sea].std())[sea])


def test_train_and_evaluate_on_each_region(tmp_path):
    path, cfg, a, b = _setup(tmp_path)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(path), "--fold", "val2024"],
                   check=True)
    run = tmp_path / "run" / "u_v" / "val2024"
    assert "regions (train)" in (run / "train.log").read_text()
    ## untrained weights so the model equals the baseline; scoring must then match the baseline everywhere
    torch.save(build_model(dict_to_namespace(cfg)).state_dict(), run / "best.pt")
    ev = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(path), "--fold", "val2024"]
    subprocess.run(ev, check=True)
    subprocess.run(ev + ["--processed_dir", str(b)], check=True)

    for out, root in [(run / "eval_test", a), (run / "eval_test_processed_B", b)]:
        assert f"data {root}" in (out / "evaluate.log").read_text()
        for v in ("u", "v"):
            daily = np.load(out / v / "daily_metrics.npy")
            assert daily.shape == (2, 4)                                     # 2026 days of that region
            np.testing.assert_allclose(daily[:, 0], daily[:, 1], rtol=1e-4)
            pred = np.load(out / v / f"pred_{v}.npy")
            assert pred.shape[1:] == np.load(root / "static" / "land_mask.npy").shape


def test_single_region_run_scored_on_other_region_uses_training_static_stats(tmp_path):
    ## an A04-only model on A03 must see A03's bathymetry standardised as in its training (A04's stats)
    path, cfg, a, b = _setup(tmp_path, n_regions=1)
    cfg["data"]["processed_dir"] = str(a)
    path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "norm_stats.npz", u_mean=0.0, u_std=0.1, v_mean=0.0, v_std=0.1)
    torch.save(build_model(dict_to_namespace(cfg)).state_dict(), run / "best.pt")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(path),
                    "--processed_dir", str(b)], check=True)
    assert (run / "eval_test_processed_B" / "u" / "daily_metrics.npy").exists()

    ## and the dataset evaluate.py builds: A04's static stats applied to region B's fields
    st = compute_static_stats([a], ["bathymetry", "distance_to_coast"])
    cfg["data"]["processed_dir"] = str(b)
    ds = DownscalingDataset(dict_to_namespace(cfg), years=[2026], stats={v: (0.0, 0.1) for v in ("u", "v")},
                            static_stats=st)
    sea = ds.sea_mask.numpy()
    h = np.load(b / "static" / "bathymetry.npy")
    np.testing.assert_allclose(ds.static[0].numpy()[sea], ((h - st["bathymetry"][0]) / st["bathymetry"][1])[sea],
                               rtol=1e-5, atol=1e-5)
