## adjusted 160m targets: the 160m field with the day's large-scale mismatch to the 800m input removed, keeping
## its per-cell mean (the systematic, predictable part):
##   S(f)        Gaussian smoothing over sea cells only (land never leaks in), sigma = --sigma cells
##   L(day)      = S(t) - S(b)            large-scale mismatch of the day (t = 160m target, b = 800m on the 160m grid)
##   Lbar        = mean of L over the --fit_years days (never the test years)
##   target_adj  = t - (L - Lbar)          = the input's large scale + the 160m detail + the systematic offset Lbar
## NorKyst and NorFjords are separate simulations whose eddies drift apart; that wandering part (L - Lbar) cannot be
## predicted from the 800m input, so training on target_adj keeps it out of the loss. Train with
## data.target_dir: 160m_adj; score against the real target with evaluate.py --target_dir 160m.
## writes <processed_dir>/<--out_dir>/YYYYMMDD_<var>.npy (float32, land NaN) for every day of every year found,
## plus Lbar_<var>.npy and provenance.txt. No GPU.
## usage:
#python3 scripts/make_target_adj.py --processed_dir /data/prosjekt/15915_RAID/data/processed --vars u v
import argparse
import os
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from raid.data.file_utils import daily_filename

G = {}


def init(root, out, sea, sigma):
    G.update(root=root, out=out, sea=sea, sigma=sigma,
             wsum=np.maximum(ndimage.gaussian_filter(sea.astype(np.float64), sigma, mode="constant"), 1e-12))


def smooth(f):
    ## normalised convolution: mean over sea cells within the kernel only (as in consistency_stats.py)
    num = ndimage.gaussian_filter(np.where(G["sea"], np.nan_to_num(f), 0.0), G["sigma"], mode="constant")
    return num / G["wsum"]


def load(var, date):
    t = np.load(G["root"] / "160m" / daily_filename(var, date)).astype(np.float64)
    b = np.load(G["root"] / "800m_on_160m_grid" / daily_filename(var, date)).astype(np.float64)
    return t, b


def mismatch(job):
    var, date = job
    t, b = load(var, date)
    return smooth(t) - smooth(b)


def write(job):
    var, date = job
    if var not in G.setdefault("lbar", {}):                     # read once per worker
        G["lbar"][var] = np.nan_to_num(np.load(G["out"] / f"Lbar_{var}.npy").astype(np.float64))
    lbar = G["lbar"][var]
    path = G["out"] / daily_filename(var, date)
    if path.exists():
        return 0
    t, b = load(var, date)
    adj = t - (smooth(t) - smooth(b) - lbar)
    adj = np.where(G["sea"], adj, np.nan).astype(np.float32)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, adj)
    os.replace(tmp, path)                         # atomic: a restart never keeps a truncated file
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--vars", nargs="+", default=["u", "v"])
    parser.add_argument("--sigma", type=float, default=10, help="smoothing scale in 160m cells (10 = 1.6 km)")
    parser.add_argument("--fit_years", type=int, nargs="+", default=[2023, 2024, 2025],
                        help="years for the per-cell mean mismatch Lbar (never the test years)")
    parser.add_argument("--out_dir", type=str, default="160m_adj")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.processed_dir
    out = root / args.out_dir
    out.mkdir(exist_ok=True)
    sea = np.load(root / "static" / "land_mask.npy").astype(bool)
    with Pool(args.workers, initializer=init, initargs=(root, out, sea, args.sigma)) as pool:
        init(root, out, sea, args.sigma)
        for var in args.vars:
            days = sorted(p.name[:8] for p in (root / "160m").glob(daily_filename(var, "*"))
                          if (root / "800m_on_160m_grid" / p.name).exists())
            fit = [d for d in days if int(d[:4]) in args.fit_years]
            lbar = sum(pool.imap_unordered(mismatch, [(var, d) for d in fit], chunksize=8)) / len(fit)
            lbar = np.where(sea, lbar, np.nan)
            np.save(out / f"Lbar_{var}.npy", lbar.astype(np.float32))
            n = sum(pool.imap_unordered(write, [(var, d) for d in days], chunksize=8))
            print(f"{var}: Lbar from {len(fit)} days ({fit[0]}..{fit[-1]}), rms {np.sqrt(np.nanmean(lbar ** 2)):.4f}; "
                  f"{n} of {len(days)} days written to {out}")

    commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    with open(out / "provenance.txt", "a") as f:
        f.write(f"=== {datetime.now():%Y-%m-%d %H:%M:%S}  scripts/make_target_adj.py  git {commit}\n"
                f"target_adj = t - (S(t) - S(b) - Lbar), S = sea-only Gaussian sigma {args.sigma} cells, "
                f"Lbar = mean of S(t) - S(b) over years {args.fit_years}; vars {args.vars}; from {root}\n")


if __name__ == "__main__":
    main()
