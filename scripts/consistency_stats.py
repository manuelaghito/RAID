## how consistent are the 160m target and the 800m input, and where does the baseline (and model) error live?
## NorFjords (160m) and NorKyst (800m) are separate simulations: eddies and fronts drift apart, so part of the
## 160m-vs-800m difference is chaotic (not predictable from the 800m input) and part is sub-grid detail the
## 800m grid cannot resolve (what downscaling should learn). The two are separated by scale, per day:
##   S(f)  = Gaussian smoothing of f over sea cells only (land never leaks in), sigma = --sigma cells
##   L     = S(t) - S(b)          large-scale mismatch: the models disagree at a scale both resolve (chaotic)
##   e     = b - t                baseline error (b = bicubic 800m on the 160m grid, t = 160m target)
##   s     = e + L                small-scale part of the baseline error (sub-800m detail), e = -L + s
## MSE(e) = MSE(L) + MSE(s) - 2 mean(L s); the cross term is reported, it is small when the scales separate.
## Cell-days are binned by |L| ("consistency"): the share of cell-days and of baseline MSE per bin shows how
## much of the error is in inconsistent cells. With --pred_dir the same split is done for a model's errors
## (test days only), i.e. where its gain over the baseline comes from.
## Uses only the target-vs-input comparison, so it is a diagnosis: |L| needs the 160m field and could never be
## a model input. No GPU; one process per chunk of days.
## usage:
#python3 scripts/consistency_stats.py --config configs/step1b_unet.yaml \
#    --pred_dir /data/prosjekt/15915_RAID/train/step1B/temp/20261023/ensemble/eval_test \
#    --out /data/prosjekt/15915_RAID/validate/consistency/temp_20261119
## outputs in --out: summary.txt, by_bin.csv, by_scale.csv, by_month.csv, by_distance.csv, by_edge.csv,
## persistence.csv, maps.npz (per-cell RMS of L, s, e and fraction of inconsistent days), daily.csv, *.png

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

CELL_KM = 0.16
DIST_BINS = [0, 1, 2, 5, 10, 20, 50, np.inf]          # km to coast
EDGE_BINS = [0, 5, 10, 20, 50, 100, 200, np.inf]       # cells to the nearest open-sea grid edge
LAGS = [1, 2, 3, 5, 10, 20, 30]                        # days, persistence of the mismatch pattern

## filled in each worker by init()
G = {}


def smooth(f, sigma):
    ## normalised convolution: mean over sea cells within the kernel only
    num = ndimage.gaussian_filter(np.where(G["sea"], f, 0.0), sigma, mode="constant")
    return num / G["wsum"][sigma]


def init(sea, sigmas, dirs, var, pred_path, pred_dates, bins, edge_cells, dist_km):
    G.update(sea=sea, sigmas=sigmas, dirs=dirs, var=var, bins=bins, edge_cells=edge_cells, dist_km=dist_km)
    G["wsum"] = {s: np.maximum(ndimage.gaussian_filter(sea.astype(np.float64), s, mode="constant"), 1e-12)
                 for s in sigmas}
    G["pred"] = np.load(pred_path, mmap_mode="r") if pred_path else None
    G["pred_idx"] = {d: i for i, d in enumerate(pred_dates)} if pred_dates is not None else {}


def binned_sums(key, values, sq_terms):
    ## per bin of `key`: count and the sums of each squared-error field
    idx = np.digitize(key, values) - 1
    n = len(values) - 1
    out = {"count": np.bincount(idx, minlength=n)[:n].astype(np.float64)}
    for name, v in sq_terms.items():
        out[name] = np.bincount(idx, weights=v, minlength=n)[:n]
    return out


def process(chunk):
    ## one chunk of (index, date): per-day scalars, summed maps, binned sums; also returns L maps for persistence
    sea, var, sig0 = G["sea"], G["var"], G["sigmas"][0]
    dir_800m, dir_160m = G["dirs"]
    H, W = sea.shape
    maps = {k: np.zeros((H, W)) for k in ("L2", "s2", "e2", "incons", "pL2", "ps2", "pe2", "n_pred")}
    acc = {}
    days, Lmaps = [], []

    def add(group, sums):
        g = acc.setdefault(group, {k: np.zeros_like(v) for k, v in sums.items()})
        for k, v in sums.items():
            g[k] += v

    for i, date in chunk:
        t = np.load(dir_160m / daily_filename(var, date)).astype(np.float64)
        b = np.load(dir_800m / daily_filename(var, date)).astype(np.float64)
        e = b - t
        row = {"date": date}
        for sigma in G["sigmas"]:
            L = smooth(t, sigma) - smooth(b, sigma)
            s = e + L
            row[f"mse_e"] = (e[sea] ** 2).mean()
            row[f"mse_L_s{sigma}"] = (L[sea] ** 2).mean()
            row[f"mse_s_s{sigma}"] = (s[sea] ** 2).mean()
            row[f"cross_s{sigma}"] = -2 * (L[sea] * s[sea]).mean()
            if sigma == sig0:
                L0, s0 = L, s
        absL = np.abs(L0[sea])
        thr = G["bins"][len(G["bins"]) // 2]
        row["frac_incons"] = (absL >= thr).mean()
        row["absL_p50"], row["absL_p90"] = np.percentile(absL, [50, 90])
        maps["L2"] += np.where(sea, L0 ** 2, 0)
        maps["s2"] += np.where(sea, s0 ** 2, 0)
        maps["e2"] += np.where(sea, e ** 2, 0)
        maps["incons"] += np.where(sea, np.abs(L0) >= thr, 0)

        terms = {"e2": e[sea] ** 2, "L2": L0[sea] ** 2, "s2": s0[sea] ** 2}
        j = G["pred_idx"].get(date)
        if j is not None:
            p = np.asarray(G["pred"][j], dtype=np.float64)
            ep = p - t
            Lp = smooth(t, sig0) - smooth(np.where(sea, p, 0.0), sig0)
            sp = ep + Lp
            terms.update(pe2=ep[sea] ** 2, pL2=Lp[sea] ** 2, ps2=sp[sea] ** 2, be2=e[sea] ** 2)
            row["mse_p"] = (ep[sea] ** 2).mean()
            row[f"mse_pL_s{sig0}"] = (Lp[sea] ** 2).mean()
            row[f"mse_ps_s{sig0}"] = (sp[sea] ** 2).mean()
            maps["pe2"] += np.where(sea, ep ** 2, 0)
            maps["pL2"] += np.where(sea, Lp ** 2, 0)
            maps["ps2"] += np.where(sea, sp ** 2, 0)
            maps["n_pred"] += sea
            ## model-vs-baseline per consistency bin, test days only
            pterms = {k: terms[k] for k in ("be2", "pe2", "pL2", "ps2")}
            pterms["bL2"], pterms["bs2"] = terms["L2"], terms["s2"]
            add("pred_by_bin", binned_sums(absL, G["bins"], pterms))
            add("pred_by_dist", binned_sums(G["dist_km"][sea], DIST_BINS, pterms))
            add("pred_by_edge", binned_sums(G["edge_cells"][sea], EDGE_BINS, pterms))
        base = {k: terms[k] for k in ("e2", "L2", "s2")}
        add("by_bin", binned_sums(absL, G["bins"], base))
        add("by_dist", binned_sums(G["dist_km"][sea], DIST_BINS, base))
        add("by_edge", binned_sums(G["edge_cells"][sea], EDGE_BINS, base))
        days.append(row)
        Lmaps.append((i, L0[sea].astype(np.float32), e[sea].astype(np.float32)))
    return days, maps, acc, Lmaps


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        ## union of keys in first-seen order: e.g. only test days carry the model columns
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def bin_label(edges, i, unit=""):
    lo, hi = edges[i], edges[i + 1]
    return f"{lo:g}+{unit}" if np.isinf(hi) else f"{lo:g}-{hi:g}{unit}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config with data.processed_dir and the variable")
    parser.add_argument("--var", default=None, help="variable (default: first of data.target_vars)")
    parser.add_argument("--years", type=int, nargs="*", default=None, help="default: all days on disk")
    parser.add_argument("--sigma", type=float, nargs="+", default=[10, 5, 20],
                        help="smoothing scales in 160m cells, the first one is the main one (10 = 1.6 km)")
    parser.add_argument("--bins", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0, np.inf],
                        help="|L| bin edges in the variable's units; the middle edge is the 'inconsistent' threshold")
    parser.add_argument("--pred_dir", type=Path, default=None,
                        help="an eval_<split> dir from evaluate.py (pred_<var>.npy + dates.npy) to split model errors")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--processed_dir", default=None, help="override data.processed_dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    var = args.var or cfg["data"]["target_vars"][0]
    root = Path(args.processed_dir or cfg["data"]["processed_dir"])
    dir_800m, dir_160m = root / "800m_on_160m_grid", root / "160m"
    sea = np.load(root / "static" / "land_mask.npy").astype(bool)
    dist_km = np.load(root / "static" / "distance_to_coast.npy").astype(np.float64)
    bins = list(args.bins)
    if bins[-1] != np.inf:
        bins.append(np.inf)

    ## distance (cells) to the nearest open-sea grid edge = the 160m model's open boundary
    edge = np.zeros_like(sea)
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        edge[sl] = sea[sl]
    edge_cells = ndimage.distance_transform_edt(~edge)

    dates = sorted(p.name[:8] for p in dir_160m.glob(daily_filename(var, "[0-9]" * 8))
                   if (dir_800m / p.name).exists())
    if args.years:
        dates = [d for d in dates if int(d[:4]) in args.years]
    pred_path = pred_dates = None
    if args.pred_dir:
        pred_path = args.pred_dir / f"pred_{var}.npy"
        pred_dates = [str(d) for d in np.load(args.pred_dir / "dates.npy")]
    print(f"{var}: {len(dates)} days {dates[0]} -> {dates[-1]}, sigma {args.sigma} cells, bins {bins}, "
          f"pred {args.pred_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    indexed = list(enumerate(dates))
    chunks = [indexed[k:k + 8] for k in range(0, len(indexed), 8)]
    Lall = np.zeros((len(dates), sea.sum()), np.float32)
    Eall = np.zeros_like(Lall)          # baseline error e = b - t on sea cells
    days, maps, acc = [], None, {}
    with Pool(args.workers, initializer=init,
              initargs=(sea, args.sigma, (dir_800m, dir_160m), var, pred_path, pred_dates, bins, edge_cells,
                        dist_km)) as pool:
        for k, (d, m, a, Ls) in enumerate(pool.imap_unordered(process, chunks)):
            days += d
            maps = m if maps is None else {key: maps[key] + m[key] for key in maps}
            for group, sums in a.items():
                g = acc.setdefault(group, {key: np.zeros_like(v) for key, v in sums.items()})
                for key, v in sums.items():
                    g[key] += v
            for i, Lsea, Esea in Ls:
                Lall[i] = Lsea
                Eall[i] = Esea
            if (k + 1) % 20 == 0:
                print(f"  {len(days)}/{len(dates)} days", flush=True)
    days.sort(key=lambda r: r["date"])
    T = len(days)
    sig0 = args.sigma[0]
    thr = bins[len(bins) // 2]

    write_csv(args.out / "daily.csv", days)

    ## --- by scale (all days) ---
    mse_e = np.mean([r["mse_e"] for r in days])
    by_scale = []
    for s in args.sigma:
        mL, ms, cr = (np.mean([r[f"{k}_s{s}"] for r in days]) for k in ("mse_L", "mse_s", "cross"))
        by_scale.append({"sigma_cells": s, "sigma_km": s * CELL_KM, "rmse_baseline": np.sqrt(mse_e),
                         "share_large_scale_L": mL / mse_e, "share_small_scale_s": ms / mse_e,
                         "share_cross": cr / mse_e})
    write_csv(args.out / "by_scale.csv", by_scale)

    ## --- by consistency bin (all days) ---
    g = acc["by_bin"]
    by_bin = [{"absL_bin": bin_label(bins, i), "share_cell_days": g["count"][i] / g["count"].sum(),
               "share_baseline_mse": g["e2"][i] / g["e2"].sum(),
               "rmse_baseline": np.sqrt(g["e2"][i] / max(g["count"][i], 1)),
               "share_L_in_bin_mse": g["L2"][i] / max(g["e2"][i], 1e-30)} for i in range(len(bins) - 1)]
    if "pred_by_bin" in acc:
        pg = acc["pred_by_bin"]
        for i, r in enumerate(by_bin):
            n = max(pg["count"][i], 1)
            r.update(test_share_cell_days=pg["count"][i] / pg["count"].sum(),
                     test_rmse_baseline=np.sqrt(pg["be2"][i] / n), test_rmse_model=np.sqrt(pg["pe2"][i] / n),
                     test_skill=1 - np.sqrt(pg["pe2"][i] / max(pg["be2"][i], 1e-30)),
                     test_share_of_mse_reduction=(pg["be2"][i] - pg["pe2"][i]) / (pg["be2"].sum() - pg["pe2"].sum()))
    write_csv(args.out / "by_bin.csv", by_bin)

    ## --- by distance to coast / to the open boundary ---
    for name, edges, unit in (("dist", DIST_BINS, " km"), ("edge", EDGE_BINS, " cells")):
        g = acc[f"by_{name}"]
        rows = [{"bin": bin_label(edges, i, unit), "share_cells": g["count"][i] / g["count"].sum(),
                 "share_baseline_mse": g["e2"][i] / g["e2"].sum(),
                 "rmse_baseline": np.sqrt(g["e2"][i] / max(g["count"][i], 1)),
                 "share_L": g["L2"][i] / max(g["e2"][i], 1e-30), "share_s": g["s2"][i] / max(g["e2"][i], 1e-30)}
                for i in range(len(edges) - 1)]
        if f"pred_by_{name}" in acc:
            pg = acc[f"pred_by_{name}"]
            for i, r in enumerate(rows):
                r["test_skill"] = 1 - np.sqrt(pg["pe2"][i] / max(pg["be2"][i], 1e-30))
        write_csv(args.out / f"by_{'distance' if name == 'dist' else 'edge'}.csv", rows)

    ## --- by month ---
    months = {}
    for r in days:
        months.setdefault(f"{r['date'][:4]}-{r['date'][4:6]}", []).append(r)
    by_month = []
    for m, rs in months.items():
        me = np.mean([r["mse_e"] for r in rs])
        row = {"month": m, "days": len(rs), "rmse_baseline": np.sqrt(me),
               "share_L": np.mean([r[f"mse_L_s{sig0}"] for r in rs]) / me,
               "share_s": np.mean([r[f"mse_s_s{sig0}"] for r in rs]) / me,
               "frac_inconsistent": np.mean([r["frac_incons"] for r in rs])}
        if all("mse_p" in r for r in rs):
            row["skill_model"] = 1 - np.sqrt(np.mean([r["mse_p"] for r in rs]) / me)
        by_month.append(row)
    write_csv(args.out / "by_month.csv", by_month)

    ## --- persistence: spatial correlation of the mismatch pattern L between day d and d+lag ---
    Lc = Lall - Lall.mean(1, keepdims=True)
    norms = np.linalg.norm(Lc, axis=1)
    pers = []
    for lag in LAGS:
        if lag >= T:
            break
        c = (Lc[:-lag] * Lc[lag:]).sum(1) / (norms[:-lag] * norms[lag:])
        pers.append({"lag_days": lag, "mean_corr": float(c.mean()), "p10": float(np.percentile(c, 10)),
                     "p90": float(np.percentile(c, 90))})
    write_csv(args.out / "persistence.csv", pers)

    ## --- systematic vs transient mismatch: per-cell mean L over the fit years, applied to the eval years ---
    ## fit = all years before the last one (the test year), eval = the last year. "interp + mean L" is a
    ## simple non-ML benchmark: bicubic + a fixed per-cell (or per-cell-per-month) correction of the large-scale
    ## mismatch, learned from the fit years only; its error is e - Lbar (since t = b - e and L ~ S(t) - S(b)).
    years = np.array([int(d[:4]) for d in dates])
    mons = np.array([int(d[4:6]) for d in dates])
    fit, ev = years < years.max(), years == years.max()
    syst = []
    if fit.any() and ev.any():
        Lbar = Lall[fit].mean(0)
        Lclim = {m: Lall[fit & (mons == m)].mean(0) if (fit & (mons == m)).any() else Lbar for m in range(1, 13)}
        corr_m = np.stack([Lclim[m] for m in mons[ev]])
        Ev, Lv = Eall[ev].astype(np.float64), Lall[ev].astype(np.float64)
        rmse = lambda a: float(np.sqrt((a ** 2).mean()))
        base = rmse(Ev)
        syst = [{"what": "baseline (bicubic)", "rmse": base, "skill": 0.0},
                {"what": "bicubic + per-cell mean L (fit years)", "rmse": rmse(Ev + Lbar), "skill": 1 - rmse(Ev + Lbar) / base},
                {"what": "bicubic + per-cell monthly mean L (fit years)", "rmse": rmse(Ev + corr_m),
                 "skill": 1 - rmse(Ev + corr_m) / base}]
        if pred_dates is not None:
            daily = {r["date"]: r for r in days}
            pm = [daily[d]["mse_p"] for d in np.array(dates)[ev] if "mse_p" in daily[d]]
            if len(pm) == ev.sum():
                syst.append({"what": "model (--pred_dir)", "rmse": float(np.sqrt(np.mean(pm))),
                             "skill": 1 - float(np.sqrt(np.mean(pm))) / base})
        ## how much of the eval-year L is explained by the fit-year mean / monthly mean
        syst += [{"what": "share of eval-year L^2 left after removing per-cell mean L", "rmse": float("nan"),
                  "skill": float(((Lv - Lbar) ** 2).mean() / (Lv ** 2).mean())},
                 {"what": "share of eval-year L^2 left after removing per-cell monthly mean L", "rmse": float("nan"),
                  "skill": float(((Lv - corr_m) ** 2).mean() / (Lv ** 2).mean())}]
        write_csv(args.out / "systematic.csv", syst)

    ## --- maps ---
    mean_L = np.zeros(sea.shape)
    mean_L[sea] = Lall.mean(0)
    out_maps = {"mean_L": mean_L, "rms_L": np.sqrt(maps["L2"] / T), "rms_s": np.sqrt(maps["s2"] / T), "rms_e": np.sqrt(maps["e2"] / T),
                "frac_inconsistent": maps["incons"] / T}
    if maps["n_pred"].max() > 0:
        n = np.maximum(maps["n_pred"], 1)
        out_maps.update(rms_model=np.sqrt(maps["pe2"] / n), rms_model_L=np.sqrt(maps["pL2"] / n),
                        rms_model_s=np.sqrt(maps["ps2"] / n))
    out_maps = {k: np.where(sea, v, np.nan).astype(np.float32) for k, v in out_maps.items()}
    np.savez_compressed(args.out / "maps.npz", **out_maps, sea=sea, edge_cells=edge_cells.astype(np.float32))

    ## --- summary ---
    s0 = by_scale[0]
    lines = [f"{var}: {T} days {days[0]['date']} -> {days[-1]['date']}, main scale sigma {sig0} cells "
             f"({sig0 * CELL_KM:.2f} km), inconsistent = |L| >= {thr:g}",
             f"baseline RMSE (all days) {np.sqrt(mse_e):.4f}",
             "",
             "share of baseline MSE by scale (L = large-scale mismatch, s = small-scale detail):"]
    lines += [f"  sigma {r['sigma_km']:.2f} km: L {r['share_large_scale_L']:.1%}  s {r['share_small_scale_s']:.1%}"
              f"  cross {r['share_cross']:+.1%}" for r in by_scale]
    lines += ["", "by consistency |L| (all days):",
              f"  {'bin':<10} {'cells':>7} {'base MSE':>9} {'base RMSE':>9}" +
              (f" {'test RMSE b':>11} {'test RMSE m':>11} {'skill':>7} {'gain share':>10}" if "pred_by_bin" in acc else "")]
    for r in by_bin:
        line = f"  {r['absL_bin']:<10} {r['share_cell_days']:>7.1%} {r['share_baseline_mse']:>9.1%} {r['rmse_baseline']:>9.4f}"
        if "test_skill" in r:
            line += (f" {r['test_rmse_baseline']:>11.4f} {r['test_rmse_model']:>11.4f} {r['test_skill']:>+7.1%}"
                     f" {r['test_share_of_mse_reduction']:>10.1%}")
        lines.append(line)
    if "pred_by_bin" in acc:
        pg = acc["pred_by_bin"]
        lines += ["", "model errors by scale (test days): "
                  f"L {pg['pL2'].sum() / pg['pe2'].sum():.1%}  s {pg['ps2'].sum() / pg['pe2'].sum():.1%} of model MSE; "
                  f"baseline -> model: L MSE {pg['bL2'].sum() / pg['count'].sum():.4f} -> {pg['pL2'].sum() / pg['count'].sum():.4f}, "
                  f"s MSE {pg['bs2'].sum() / pg['count'].sum():.4f} -> {pg['ps2'].sum() / pg['count'].sum():.4f}"]
    if syst:
        lines += ["", f"systematic part: per-cell mean L from {years[fit].min()}-{years[fit].max()}, "
                  f"applied to {years.max()} ({ev.sum()} days):"]
        lines += [f"  {r['what']:<52} RMSE {r['rmse']:.4f}  skill {r['skill']:+.1%}" if np.isfinite(r["rmse"])
                  else f"  {r['what']:<64} {r['skill']:.1%}" for r in syst]
    lines += ["", "persistence of the mismatch pattern (spatial corr of L, day d vs d+lag):"]
    lines += [f"  lag {r['lag_days']:>2} d: {r['mean_corr']:.2f} (p10 {r['p10']:.2f}, p90 {r['p90']:.2f})" for r in pers]
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    ## --- figures ---
    land = "#d9d8d4"
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(land)
    panels = [("rms_e", "baseline RMSE"), ("rms_L", f"RMS large-scale mismatch L (σ {sig0 * CELL_KM:.1f} km)"),
              ("rms_s", "RMS small-scale part s"), ("frac_inconsistent", f"fraction of days |L| ≥ {thr:g}")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    vmax = np.nanpercentile(out_maps["rms_e"], 99)
    for ax, (k, title) in zip(axes.flat, panels):
        im = ax.imshow(out_maps[k], origin="lower", cmap=cmap, vmin=0,
                       vmax=1 if k == "frac_inconsistent" else vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(args.out / "maps.png", dpi=110)
    plt.close(fig)

    ## time-mean L: where the 160m model is systematically warmer (red) / colder (blue) than the 800m one
    div = plt.get_cmap("RdBu_r").copy()
    div.set_bad(land)
    v = np.nanpercentile(np.abs(out_maps["mean_L"]), 99)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    im = ax.imshow(out_maps["mean_L"], origin="lower", cmap=div, vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(f"time-mean large-scale mismatch L = S(160m) − S(800m), {days[0]['date']}–{days[-1]['date']}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label=var)
    fig.savefig(args.out / "mean_mismatch.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    x = np.arange(T)
    me = np.array([r["mse_e"] for r in days])
    axes[0].plot(x, np.sqrt(me), color="#2a78d6", lw=1, label="baseline RMSE")
    axes[0].plot(x, np.sqrt([r[f"mse_L_s{sig0}"] for r in days]), color="#c2410c", lw=1, label="RMS L (large scale)")
    axes[0].plot(x, np.sqrt([r[f"mse_s_s{sig0}"] for r in days]), color="#15803d", lw=1, label="RMS s (small scale)")
    axes[0].legend(frameon=False); axes[0].set_ylabel(var)
    axes[1].plot(x, [r["frac_incons"] for r in days], color="#6b6a66", lw=1)
    axes[1].set_ylabel(f"fraction of sea cells |L| ≥ {thr:g}")
    ticks = [i for i, r in enumerate(days) if r["date"][6:] == "01" and r["date"][4:6] in ("01", "04", "07", "10")]
    axes[1].set_xticks(ticks, [days[i]["date"][:6] for i in ticks], rotation=45)
    fig.savefig(args.out / "timeseries.png", dpi=110)
    plt.close(fig)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
