## evaluate a trained run on the test split: model vs bicubic baseline (input channel 0), in physical units.
## inputs are normalised with the run's own training stats (norm_stats.npz), as in training.
## python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2025

## CUDA_VISIBLE_DEVICES=1
## usage: 
#python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2025
#python3 scripts/evaluate.py --config configs/step1a_cnn.yaml --run_dir experiments/step1a_cnn/val2024
#Results go to <run_dir>/eval_test; monthly_metrics.csv has RMSE, bias (pred - target) and skill per month
## --fold sets the run dir (<output_dir>/<var>/<fold>) and the fold's val years, for scoring CV runs:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold val2024 --split val
## --fold full: the final model trained on split.full (test split only, it has no val years):
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold full
## --fold ensemble averages the predictions of all fold models (test split only),
## results go to <output_dir>/<var>/ensemble/eval_test:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold ensemble
## --output_dir overrides output_dir from the config (same as train.py), e.g. for runs kept elsewhere:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold ensemble --output_dir /some/where/unet_try2
## --edge_pixels N scores only the interior: N cells along every grid edge are left out of all metrics and maps
## (as train.edge_pixels does for the loss), so edge 0/5/10 runs can be compared on the same cells.
## results go to eval_<split>_trim<N> next to the normal eval_<split>:
#python3 scripts/evaluate.py --config configs/step1b_unet.yaml --fold ensemble --edge_pixels 10

#CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate.py --config configs/step1b_unet.yaml 
#      --run_dir /data/prosjekt/15915_RAID/train/step1B/temp/val2025  
#      --fold val2025 --output_dir experiments/step1b_unet/temp/

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from raid.config import apply_fold, dict_to_namespace, var_dir
from raid.data.dataset import DownscalingDataset, denormalize, trim_edges
from raid.data.file_utils import daily_filename
from raid.log import tee_output
from raid.metrics import UNITS, masked_bias, masked_rmse
from raid.models import build_model



def load_stats(path, names):
    s = np.load(path)
    return {v: (float(s[f"{v}_mean"]), float(s[f"{v}_std"])) for v in names}


def load_members(cfg, run_dirs, years, device):
    ## one (model, dataset, stats) per run: each model sees inputs normalised with its own training stats
    names = dict.fromkeys(cfg.data.input_vars + cfg.data.target_vars)
    members = []
    for run in run_dirs:
        stats = load_stats(run / "norm_stats.npz", names)
        ds = DownscalingDataset(cfg, years=years, stats=stats)
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        model.eval()
        members.append((model, ds, stats))
    dates = members[0][1].dates
    for run, (_, ds, _) in zip(run_dirs, members):
        assert ds.dates == dates, f"{run}: different days than {run_dirs[0]}"
    return members


def monthly_metrics(dates, daily):
    ## per calendar month (YYYY-MM): pooled RMSE = sqrt(mean daily MSE), bias = mean daily bias
    months = np.array([f"{d[:4]}-{d[4:6]}" for d in dates])
    rows = []
    for m in dict.fromkeys(months):                                  # in date order
        d = daily[months == m]
        rmse_m, rmse_b = np.sqrt((d[:, :2] ** 2).mean(0))
        rows.append({"month": m, "n_days": len(d),
                     "rmse_model": float(rmse_m), "rmse_baseline": float(rmse_b),
                     "bias_model": float(d[:, 2].mean()), "bias_baseline": float(d[:, 3].mean()),
                     "skill": float(1 - rmse_m / rmse_b)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="override output_dir from config, as in train.py: runs are read from "
                             "<output_dir>[/<var>/<fold>], results written next to them")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="dir with best.pt + norm_stats.npz (default: output_dir from config)")
    parser.add_argument("--split", type=str, default="test", help="split name from the config")
    parser.add_argument("--fold", type=str, default=None,
                        help="CV fold name from split.folds (run dir <output_dir>/<var>/<fold>, val = the fold's val "
                             "years), or 'ensemble' to average all fold models (test split only)")
    parser.add_argument("--edge_pixels", type=int, default=0,
                        help="leave N cells along every grid edge out of the metrics (default 0: all sea cells); "
                             "results go to eval_<split>_trim<N>")
    args = parser.parse_args()
    suffix = f"_trim{args.edge_pixels}" if args.edge_pixels > 0 else ""

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir
    ensemble = args.fold == "ensemble"
    if ensemble:
        ## each fold model was trained on the other folds' val years, so only the test split is unseen by all
        if args.split != "test":
            parser.error("--fold ensemble only works with --split test")
        if args.run_dir is not None:
            parser.error("--fold ensemble reads <output_dir>/<var>/<fold> for every fold; --run_dir is not used")
        run_dirs = [Path(var_dir(raw)) / name for name in raw["split"]["folds"]]
        out = Path(var_dir(raw)) / "ensemble" / f"eval_{args.split}{suffix}"
    else:
        if args.fold is not None:
            apply_fold(raw, args.fold)
        elif args.split in ("train", "val"):
            parser.error(f"--split {args.split} needs --fold: train/val years come from split.folds")
        if not raw["split"].get(args.split, {}).get("years"):
            parser.error(f"no {args.split} years for fold {args.fold}")
        run_dirs = [Path(args.run_dir or raw["output_dir"])]
        out = run_dirs[0] / f"eval_{args.split}{suffix}"
    cfg = dict_to_namespace(raw)
    out.mkdir(parents=True, exist_ok=True)
    tee_output(out / "evaluate.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    members = load_members(cfg, run_dirs, getattr(cfg.split, args.split).years, device)
    loaders = [DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)
               for _, ds, _ in members]
    ds = members[0][1]     # dates, sea mask and file dirs are the same for every member

    ## one model output per target variable; output k is the correction to input channel k (same variable),
    ## so the bicubic baseline for target k is input_vars[k]. With several targets each variable gets its
    ## own subfolder <out>/<var>/ holding the same files a single-variable evaluation writes to <out>/
    targets = list(cfg.data.target_vars)
    dests = {v: (out / v if len(targets) > 1 else out) for v in targets}
    ## copy, not a view: starting the DataLoader workers (forkserver, the default from python 3.14) moves the
    ## dataset's tensors into shared memory, which would leave a .numpy() view pointing at freed memory
    sea = ds.sea_mask.numpy().copy()
    T, (H, W) = len(ds), sea.shape
    ## cells that are scored: all sea cells, or the interior with --edge_pixels
    scored = trim_edges(sea, args.edge_pixels)
    acc = {}
    for v, d in dests.items():
        d.mkdir(parents=True, exist_ok=True)
        acc[v] = dict(
            ## (T, H, W) physical units, land = NaN; written day by day so the whole array never sits in RAM
            pred=np.lib.format.open_memmap(d / f"pred_{v}.npy", mode="w+", dtype=np.float32, shape=(T, H, W)),
            ## per day: rmse_model, rmse_baseline, bias_model, bias_baseline (bias = pred - target)
            daily=np.zeros((T, 4), np.float32),
            sq_model=np.zeros((H, W)), sq_base=np.zeros((H, W)), bias_sum=np.zeros((H, W)))

    with torch.no_grad():
        for i, batches in enumerate(zip(*loaders)):
            outs = [model(x.to(device))[0].cpu().numpy() for (model, _, _), (x, _, _) in zip(members, batches)]
            date = ds.dates[i]
            for k, v in enumerate(targets):
                a = acc[v]
                ## ensemble = mean of the members' predictions in physical units (a single run is a 1-member ensemble)
                p = np.mean([denormalize(o[k], *st[v]) for o, (_, _, st) in zip(outs, members)], axis=0)
                ## target and bicubic baseline straight from the physical-unit files, not via the normalised x/y:
                ## the dataset fills NaN with 0 after normalising, which would turn them into the fold's mean
                t = np.load(ds.dir_160m / daily_filename(v, date))
                b = np.load(ds.dir_800m / daily_filename(cfg.data.input_vars[k], date))
                n_t, n_b = np.isnan(t[sea]).sum(), np.isnan(b[sea]).sum()
                if n_t or n_b:
                    raise ValueError(f"{date} {v}: NaN on sea cells, target {n_t}, baseline input {n_b}")
                a["pred"][i] = np.where(sea, p, np.nan)
                a["daily"][i] = (masked_rmse(p, t, scored), masked_rmse(b, t, scored),
                                 masked_bias(p, t, scored), masked_bias(b, t, scored))
                a["sq_model"] += (p - t) ** 2
                a["sq_base"] += (b - t) ** 2
                a["bias_sum"] += p - t

    land = lambda arr: np.where(scored, arr, np.nan).astype(np.float32)
    print(f"{args.split}{f' (fold {args.fold})' if args.fold else ''}: {T} days ({ds.dates[0]} -> {ds.dates[-1]}), "
          f"run{'s' if ensemble else ''} {', '.join(map(str, run_dirs))}")
    if args.edge_pixels > 0:
        print(f"  interior only: {args.edge_pixels} edge cells left out, {scored.sum()} of {sea.sum()} sea cells scored")
    for v, d in dests.items():
        a = acc[v]
        a["pred"].flush()
        daily = a["daily"]
        np.save(d / "dates.npy", np.array(ds.dates))                         # (T,) 'YYYYMMDD'
        np.save(d / "daily_metrics.npy", daily)                              # (T, 4), columns above
        np.save(d / f"rmse_map_model_{v}.npy", land(np.sqrt(a["sq_model"] / T)))
        np.save(d / f"rmse_map_baseline_{v}.npy", land(np.sqrt(a["sq_base"] / T)))
        np.save(d / f"bias_map_model_{v}.npy", land(a["bias_sum"] / T))

        ## same sea mask every day, so the overall RMSE is the root of the mean daily MSE
        rmse_m, rmse_b = np.sqrt((daily[:, :2] ** 2).mean(0))
        unit = UNITS.get(v, "")
        if len(targets) > 1:
            print(f"\n  [{v}]")
        print(f"  model    RMSE {rmse_m:.4f} {unit}  bias {daily[:, 2].mean():+.4f}")
        print(f"  baseline RMSE {rmse_b:.4f} {unit}  bias {daily[:, 3].mean():+.4f}")
        print(f"  skill vs baseline: {1 - rmse_m / rmse_b:+.1%}")

        monthly = monthly_metrics(ds.dates, daily)
        with open(d / "monthly_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(monthly[0]))
            w.writeheader()
            w.writerows({k: f"{x:.6f}" if isinstance(x, float) else x for k, x in r.items()} for r in monthly)
        print(f"\n  {'month':<8} {'days':>4}  {'RMSE model':>10} {'baseline':>9}  {'bias model':>10} {'baseline':>9}  {'skill':>6}")
        for r in monthly:
            print(f"  {r['month']:<8} {r['n_days']:>4}  {r['rmse_model']:>10.4f} {r['rmse_baseline']:>9.4f}  "
                  f"{r['bias_model']:>+10.4f} {r['bias_baseline']:>+9.4f}  {r['skill']:>+6.1%}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
