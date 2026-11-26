## does a model reproduce the small-scale features and gradients of the 160m target, and are they tied to the
## bathymetry? RMSE rewards smooth fields, so this compares target (t), bicubic baseline (b) and model (p) on:
##   gradient magnitude |grad f| (per km, central differences, only where the cell and its 4 neighbours are sea)
##   small-scale part   f' = f - S(f), S = Gaussian smoothing over sea cells only (sigma 10 cells = 1.6 km)
##   variance ratio     var(f'_p) / var(f'_t)        < 1: too smooth, > 1: too much (or wrong) detail
##   pattern corr       corr(f'_p, f'_t)             is the detail in the right place?
##   bathymetry link    per bathymetry-slope bin: mean |grad f| and rms f' for t, b, p, and the per-day spatial
##                      correlation of log|grad f| with log|grad h| (h = depth)
## the slope bins are quantiles of |grad h| over valid sea cells, so each bin holds ~the same number of cells.
## Days = the --pred_dir days (an eval_<split> dir from evaluate.py with pred_<var>.npy + dates.npy). No GPU.
## usage:
#python3 scripts/smallscale_stats.py --config configs/step1b_unet.yaml \
#    --pred_dir /data/prosjekt/15915_RAID/train/step1B/temp/20261119_edgestudy/edge0_seed0/temp/ensemble/eval_test \
#    --out /data/prosjekt/15915_RAID/validate/smallscale/temp_20261119
## outputs in --out: summary.txt, by_slope.csv, daily.csv, maps.npz, maps.png, by_slope.png

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage

from raid.data.file_utils import daily_filename
from raid.metrics import UNITS

CELL_KM = 0.16
N_SLOPE_BINS = 8
FIELDS = ("t", "b", "p")
NAMES = {"t": "target 160m", "b": "bicubic 800m", "p": "model"}

G = {}


def grad_mag(f, valid):
    ## |grad f| per km with central differences; NaN where a neighbour is land
    gy = np.zeros_like(f)
    gx = np.zeros_like(f)
    gy[1:-1] = (f[2:] - f[:-2]) / (2 * CELL_KM)
    gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * CELL_KM)
    return np.where(valid, np.hypot(gx, gy), np.nan)


def smooth(f):
    num = ndimage.gaussian_filter(np.where(G["sea"], f, 0.0), G["sigma"], mode="constant")
    return num / G["wsum"]


def init(sea, valid, sigma, dirs, var, pred_path, slope_idx, logslope):
    G.update(sea=sea, valid=valid, sigma=sigma, dirs=dirs, var=var, slope_idx=slope_idx, logslope=logslope)
    G["wsum"] = np.maximum(ndimage.gaussian_filter(sea.astype(np.float64), sigma, mode="constant"), 1e-12)
    G["pred"] = np.load(pred_path, mmap_mode="r")


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def process(chunk):
    sea, valid, var = G["sea"], G["valid"], G["var"]
    dir_800m, dir_160m = G["dirs"]
    nb = N_SLOPE_BINS
    H, W = sea.shape
    maps = {f"{k}_{f}": np.zeros((H, W)) for k in ("grad", "ss2") for f in FIELDS}
    sums = {f"{k}_{f}": np.zeros(nb) for k in ("grad", "ss2") for f in FIELDS}
    sums["n"] = np.bincount(G["slope_idx"], minlength=nb).astype(np.float64) * len(chunk)
    rows = []
    for j, date in chunk:
        fld = {"t": np.load(dir_160m / daily_filename(var, date)).astype(np.float64),
               "b": np.load(dir_800m / daily_filename(var, date)).astype(np.float64),
               "p": np.asarray(G["pred"][j], dtype=np.float64)}
        row = {"date": date}
        ss = {}
        for f, x in fld.items():
            x = np.where(sea, x, 0.0)
            g = grad_mag(x, valid)
            s = np.where(sea, x - smooth(x), 0.0)
            ss[f] = s[valid]
            gv = g[valid]
            row[f"grad_mean_{f}"] = gv.mean()
            row[f"grad_p90_{f}"] = np.percentile(gv, 90)
            row[f"var_ss_{f}"] = (ss[f] ** 2).mean()
            row[f"corr_slope_{f}"] = corr(np.log(gv + 1e-6), G["logslope"])
            row[f"corr_ssabs_slope_{f}"] = corr(np.abs(ss[f]), G["logslope"])
            maps[f"grad_{f}"] += np.where(valid, g, 0)
            maps[f"ss2_{f}"] += np.where(valid, s ** 2, 0)
            sums[f"grad_{f}"] += np.bincount(G["slope_idx"], weights=gv, minlength=nb)
            sums[f"ss2_{f}"] += np.bincount(G["slope_idx"], weights=ss[f] ** 2, minlength=nb)
        for f in ("b", "p"):
            row[f"corr_ss_{f}t"] = corr(ss[f], ss["t"])
            row[f"mse_ss_{f}t"] = ((ss[f] - ss["t"]) ** 2).mean()
        rows.append(row)
    return rows, maps, sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--var", default=None, help="variable (default: first of data.target_vars)")
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=10, help="small-scale cutoff in 160m cells (10 = 1.6 km)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--processed_dir", default=None, help="override data.processed_dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    var = args.var or cfg["data"]["target_vars"][0]
    unit = UNITS.get(var, "")
    root = Path(args.processed_dir or cfg["data"]["processed_dir"])
    sea = np.load(root / "static" / "land_mask.npy").astype(bool)
    h = np.load(root / "static" / "bathymetry.npy").astype(np.float64)
    ## a cell is valid if it and its 4 neighbours are sea (and not on the grid border)
    valid = sea.copy()
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    valid[1:-1, 1:-1] &= sea[:-2, 1:-1] & sea[2:, 1:-1] & sea[1:-1, :-2] & sea[1:-1, 2:]
    slope = grad_mag(h, valid)[valid]                          # m per km
    edges = np.quantile(slope, np.linspace(0, 1, N_SLOPE_BINS + 1))
    slope_idx = np.clip(np.searchsorted(edges, slope, side="right") - 1, 0, N_SLOPE_BINS - 1)
    logslope = np.log(slope + 1e-3)

    pred_path = args.pred_dir / f"pred_{var}.npy"
    dates = [str(d) for d in np.load(args.pred_dir / "dates.npy", allow_pickle=True)]
    dirs = (root / "800m_on_160m_grid", root / "160m")
    items = list(enumerate(dates))
    chunks = [items[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]
    print(f"{var}: {len(dates)} days {dates[0]} -> {dates[-1]}, {valid.sum()} valid cells, "
          f"sigma {args.sigma:g} cells ({args.sigma * CELL_KM:.2f} km), pred {pred_path}")
    with Pool(len(chunks), initializer=init,
              initargs=(sea, valid, args.sigma, dirs, var, pred_path, slope_idx, logslope)) as pool:
        results = pool.map(process, chunks)

    rows = sorted((r for res in results for r in res[0]), key=lambda r: r["date"])
    T = len(rows)
    maps = {k: sum(res[1][k] for res in results) / T for k in results[0][1]}
    sums = {k: sum(res[2][k] for res in results) for k in results[0][2]}
    args.out.mkdir(parents=True, exist_ok=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    lines = [f"{var} [{unit}]: {T} days {dates[0]} -> {dates[-1]}, pred {pred_path}",
             f"small scale = f - S(f), sigma {args.sigma:g} cells ({args.sigma * CELL_KM:.2f} km); "
             f"gradients in {unit}/km", ""]
    lines.append(f"{'':28s}" + "".join(f"{NAMES[f]:>16s}" for f in FIELDS))
    for k, label in [("grad_mean", "mean |grad f|"), ("grad_p90", "p90 |grad f|"), ("var_ss", "var small-scale")]:
        lines.append(f"{label:28s}" + "".join(f"{m(f'{k}_{f}'):16.4g}" for f in FIELDS))
    for k, label in [("grad_mean", "mean |grad f| / target"), ("grad_p90", "p90 |grad f| / target"),
                     ("var_ss", "var small-scale / target")]:
        lines.append(f"{label:28s}" + "".join(f"{m(f'{k}_{f}') / m(f'{k}_t'):16.3f}" for f in FIELDS))
    lines.append(f"{'corr(log|grad f|, log|grad h|)':28s}" + "".join(f"{m(f'corr_slope_{f}'):16.3f}" for f in FIELDS))
    lines.append(f"{'corr(|f_small|, log|grad h|)':28s}" + "".join(f"{m(f'corr_ssabs_slope_{f}'):16.3f}" for f in FIELDS))
    lines += ["", "small-scale part vs target (pattern):",
              f"  corr(b', t') {m('corr_ss_bt'):.3f}   corr(p', t') {m('corr_ss_pt'):.3f}",
              f"  MSE(b' - t') {m('mse_ss_bt'):.4g}   MSE(p' - t') {m('mse_ss_pt'):.4g}   "
              f"small-scale skill {1 - np.sqrt(m('mse_ss_pt') / m('mse_ss_bt')):+.1%}"]

    slope_rows = []
    for i in range(N_SLOPE_BINS):
        r = {"slope_lo": edges[i], "slope_hi": edges[i + 1]}
        for f in FIELDS:
            r[f"grad_{f}"] = sums[f"grad_{f}"][i] / sums["n"][i]
            r[f"rms_ss_{f}"] = np.sqrt(sums[f"ss2_{f}"][i] / sums["n"][i])
        slope_rows.append(r)
    lines += ["", "by bathymetry slope |grad h| (m/km), equal-count bins:",
              f"  {'slope':>14s}" + "".join(f"{'grad ' + f:>10s}" for f in FIELDS)
              + "".join(f"{'rms ' + f + chr(39):>10s}" for f in FIELDS) + f"{'grad p/t':>10s}{'rms p/t':>10s}"]
    for r in slope_rows:
        lines.append(f"  {r['slope_lo']:6.1f}-{r['slope_hi']:7.1f}" + "".join(f"{r[f'grad_{f}']:10.4g}" for f in FIELDS)
                     + "".join(f"{r[f'rms_ss_{f}']:10.4g}" for f in FIELDS)
                     + f"{r['grad_p'] / r['grad_t']:10.3f}{r['rms_ss_p'] / r['rms_ss_t']:10.3f}")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for path, data in [(args.out / "daily.csv", rows), (args.out / "by_slope.csv", slope_rows)]:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in data)

    out_maps = {f"mean_grad_{f}": np.where(valid, maps[f"grad_{f}"], np.nan) for f in FIELDS}
    out_maps |= {f"rms_ss_{f}": np.where(valid, np.sqrt(maps[f"ss2_{f}"]), np.nan) for f in FIELDS}
    out_maps["slope"] = np.where(valid, grad_mag(h, valid), np.nan)
    np.savez_compressed(args.out / "maps.npz", **out_maps)

    fig, axs = plt.subplots(2, 4, figsize=(22, 9))
    gmax = np.nanpercentile(out_maps["mean_grad_t"], 98)
    smax = np.nanpercentile(out_maps["rms_ss_t"], 98)
    for c, f in enumerate(FIELDS):
        for r, (k, vmax, lab) in enumerate([("mean_grad", gmax, f"mean |grad| [{unit}/km]"),
                                            ("rms_ss", smax, f"rms small-scale [{unit}]")]):
            ax = axs[r, c]
            ax.set_facecolor("0.85")
            im = ax.imshow(out_maps[f"{k}_{f}"], origin="lower", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{NAMES[f]}: {lab}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[0, 3]
    ax.set_facecolor("0.85")
    im = ax.imshow(np.log10(out_maps["slope"] + 0.1), origin="lower", interpolation="nearest", cmap="cividis")
    ax.set_title("log10 bathymetry slope [m/km]"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[1, 3]
    ratio = out_maps["rms_ss_p"] / np.maximum(out_maps["rms_ss_t"], 1e-12)
    im = ax.imshow(ratio, origin="lower", vmin=0, vmax=2, cmap="RdBu_r", interpolation="nearest")
    ax.set_facecolor("0.85")
    ax.set_title("rms small-scale model / target"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{var}: {T} days {dates[0]}-{dates[-1]}")
    fig.tight_layout()
    fig.savefig(args.out / "maps.png", dpi=80)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    mid = np.sqrt(np.maximum(edges[:-1], 1e-2) * edges[1:])
    for f in FIELDS:
        axs[0].plot(mid, [r[f"grad_{f}"] for r in slope_rows], "o-", label=NAMES[f])
        axs[1].plot(mid, [r[f"rms_ss_{f}"] for r in slope_rows], "o-", label=NAMES[f])
    for ax, lab in zip(axs, [f"mean |grad f| [{unit}/km]", f"rms small-scale [{unit}]"]):
        ax.set_xscale("log"); ax.set_xlabel("bathymetry slope |grad h| [m/km]"); ax.set_ylabel(lab); ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{var}: by bathymetry slope (equal-count bins)")
    fig.tight_layout()
    fig.savefig(args.out / "by_slope.png", dpi=90)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
