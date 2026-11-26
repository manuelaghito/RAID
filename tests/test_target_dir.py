## data.target_dir (train on e.g. 160m_adj), evaluate.py --target_dir, and scripts/make_target_adj.py
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import build_dataloaders
from raid.data.file_utils import daily_filename
from raid.models import build_model

ROOT = Path(__file__).resolve().parents[1]
DATES = ["20231201", "20240301", "20240302", "20250301", "20260101", "20260102"]
SHAPE = (40, 48)


def _data(root, rng):
    for sub in ("800m_on_160m_grid", "160m", "160m_adj", "static"):
        (root / sub).mkdir(parents=True)
    land = np.ones(SHAPE, np.uint8)
    land[:4, :6] = 0
    for d in DATES:
        for v in ("u", "v"):
            b = ndimage.gaussian_filter(rng.normal(size=SHAPE), 6).astype(np.float32)
            t = (b + 0.1 * rng.normal(size=SHAPE)).astype(np.float32)
            adj = (t + 5.0).astype(np.float32)               # recognisably different from the real target
            for sub, a in (("800m_on_160m_grid", b), ("160m", t), ("160m_adj", adj)):
                a = a.copy()
                a[land == 0] = np.nan
                np.save(root / sub / daily_filename(v, d), a)
    np.save(root / "static" / "land_mask.npy", land)
    np.save(root / "static" / "bathymetry.npy", rng.uniform(10, 100, SHAPE).astype(np.float32))
    np.save(root / "static" / "distance_to_coast.npy", rng.uniform(0, 20, SHAPE).astype(np.float32))
    return land.astype(bool)


def _cfg(tmp_path):
    root = tmp_path / "processed"
    sea = _data(root, np.random.default_rng(0))
    cfg = yaml.safe_load((ROOT / "configs" / "step1b_unet_uv_adj.yaml").read_text())
    cfg["data"]["processed_dir"] = str(root)
    cfg["train"].update(num_workers=0, epochs=1, batch_size=2, patch_size=16, crops_per_day=2)
    cfg["model"].update(base_channels=8, depth=2)
    cfg["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, root, sea


def test_training_reads_target_dir(tmp_path):
    path, cfg, root, sea = _cfg(tmp_path)
    assert cfg["data"]["target_dir"] == "160m_adj"
    apply_fold(cfg, "val2024")
    c = dict_to_namespace(cfg)
    train_loader, val_loader = build_dataloaders(c)
    ## targets (and the stats of the target variables) come from 160m_adj: 5 above the real 160m target
    stats = np.load(Path(c.output_dir) / "norm_stats.npz")
    adj = np.concatenate([np.load(root / "160m_adj" / daily_filename("u", d))[sea] for d in ("20231201", "20250301")])
    np.testing.assert_allclose([stats["u_mean"], stats["u_std"]], [adj.mean(), adj.std()], rtol=1e-5)
    assert val_loader.dataset.dir_160m == root / "160m_adj"


def test_evaluate_against_real_target(tmp_path):
    path, cfg, root, sea = _cfg(tmp_path)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(path), "--fold", "val2024"],
                   check=True)
    run = tmp_path / "run" / "u_v" / "val2024"
    torch.save(build_model(dict_to_namespace(cfg)).state_dict(), run / "best.pt")   # model = baseline
    ev = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--config", str(path), "--fold", "val2024"]
    subprocess.run(ev, check=True)
    subprocess.run(ev + ["--target_dir", "160m"], check=True)
    for out, sub in ((run / "eval_test", "160m_adj"), (run / "eval_test_target160m", "160m")):
        daily = np.load(out / "u" / "daily_metrics.npy")
        t = [np.load(root / sub / daily_filename("u", d))[sea] for d in ("20260101", "20260102")]
        b = [np.load(root / "800m_on_160m_grid" / daily_filename("u", d))[sea] for d in ("20260101", "20260102")]
        np.testing.assert_allclose(daily[:, 1], [np.sqrt(np.mean((bb - tt) ** 2)) for bb, tt in zip(b, t)], rtol=1e-4)


def test_make_target_adj_removes_daily_mismatch_keeps_mean(tmp_path):
    ## t = b + a large blob at a different place every day + small-scale noise: target_adj must follow b at the
    ## large scale (up to Lbar), Lbar must come from the fit years only
    root = tmp_path / "p"
    for sub in ("800m_on_160m_grid", "160m", "static"):
        (root / sub).mkdir(parents=True)
    H, W = 60, 80
    sea = np.ones((H, W), bool)
    sea[:5, :10] = False
    np.save(root / "static" / "land_mask.npy", sea.astype(np.uint8))
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:H, :W]
    for k, d in enumerate(DATES):
        b = ndimage.gaussian_filter(rng.normal(size=(H, W)), 8)
        cy, cx = rng.uniform(15, 45), rng.uniform(15, 65)
        blob = (100.0 if d.startswith("2026") else 1.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 200)
        t = b + blob + 0.05 * rng.normal(size=(H, W))
        for sub, a in (("800m_on_160m_grid", b), ("160m", t)):
            np.save(root / sub / daily_filename("u", d), np.where(sea, a, np.nan).astype(np.float32))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_target_adj.py"), "--processed_dir", str(root),
                    "--vars", "u", "--sigma", "5", "--workers", "2"], check=True)
    lbar = np.load(root / "160m_adj" / "Lbar_u.npy")
    assert np.nanmax(np.abs(lbar)) < 2           # the 2026 blobs (x100) never entered Lbar

    w = ndimage.gaussian_filter(sea.astype(float), 5, mode="constant")
    S = lambda f: ndimage.gaussian_filter(np.where(sea, np.nan_to_num(f), 0.0), 5, mode="constant") / w
    for d in DATES:
        adj = np.load(root / "160m_adj" / daily_filename("u", d))
        b = np.load(root / "800m_on_160m_grid" / daily_filename("u", d))
        t = np.load(root / "160m" / daily_filename("u", d))
        assert np.isnan(adj[~sea]).all() and np.isfinite(adj[sea]).all()
        ## large scale of target_adj - b: exactly Lbar up to the smoothing of the small-scale part
        resid = S(adj - b) - np.nan_to_num(lbar)
        assert np.abs(resid[sea]).max() < 0.25 * np.abs(S(t - b)[sea]).max()
    assert (root / "160m_adj" / "provenance.txt").exists()
