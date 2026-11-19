## results table for every evaluated run under a folder: one row per (run, model, eval dir), read from the
## daily_metrics.npy that evaluate.py saves, plus the training settings from <run>/config.yaml when present.
## runs whose names differ only in "_seed<N>" (as written by edge_study.sh) are also pooled into
## mean / std / min / max of the skill over seeds. No GPU needed.
## usage:
#python3 scripts/summarize_runs.py /data/prosjekt/15915_RAID/train/step1B/temp/edge_study
#python3 scripts/summarize_runs.py <root> --eval eval_test_trim10      # one eval dir only
#python3 scripts/summarize_runs.py <root> --csv results.csv            # also write all rows to a csv
## skill = 1 - RMSE_model / RMSE_baseline, RMSE pooled over all days as in evaluate.py

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml

MODELS = ["val2024", "val2025", "ensemble", "full"]


def read_eval(eval_dir):
    ## columns as saved by evaluate.py: rmse_model, rmse_baseline, bias_model, bias_baseline
    d = np.load(eval_dir / "daily_metrics.npy")
    rmse_m, rmse_b = np.sqrt((d[:, :2] ** 2).mean(0))
    return dict(days=len(d), rmse_model=rmse_m, rmse_baseline=rmse_b, skill=1 - rmse_m / rmse_b,
                bias_model=d[:, 2].mean(), bias_baseline=d[:, 3].mean())


def read_settings(model_dir):
    ## config.yaml is written by train.py; ensemble folders have none, so fall back to a fold's
    for d in [model_dir, *(model_dir.parent / m for m in MODELS)]:
        if (d / "config.yaml").exists():
            t = yaml.safe_load((d / "config.yaml").read_text())["train"]
            return {k: t.get(k) for k in ("epochs", "seed", "edge_pixels", "val_edge_pixels")}
    return {}


def collect(root, only_eval=None):
    rows = []
    for f in sorted(root.rglob("daily_metrics.npy")):
        eval_dir = f.parent
        ## a run with several target variables has one subfolder per variable: eval_test/<var>/
        var = None
        if not eval_dir.name.startswith("eval_"):
            var, eval_dir = eval_dir.name, eval_dir.parent
        model_dir = eval_dir.parent
        if only_eval and eval_dir.name != only_eval:
            continue
        ## run = path between root and the model folder, e.g. edge0_seed0/temp (or "." for a single run)
        run = str(model_dir.parent.relative_to(root)) + (f" [{var}]" if var else "")
        rows.append(dict(run=run, model=model_dir.name, eval=eval_dir.name,
                         **read_settings(model_dir), **read_eval(f.parent)))
    return rows


def print_table(rows, cols, fmt):
    widths = {c: max(len(c), *(len(fmt.get(c, "{}").format(r.get(c, ""))) if r.get(c) is not None else 1
                                for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join((fmt.get(c, "{}").format(r[c]) if r.get(c) is not None else "-").ljust(widths[c])
                        for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="folder holding the runs (searched recursively)")
    parser.add_argument("--eval", default=None, help="only this eval dir name, e.g. eval_test or eval_test_trim10")
    parser.add_argument("--csv", type=Path, default=None, help="also write all rows to this csv")
    args = parser.parse_args()

    rows = collect(args.root, args.eval)
    if not rows:
        raise SystemExit(f"no daily_metrics.npy under {args.root}")
    order = {m: i for i, m in enumerate(MODELS)}
    rows.sort(key=lambda r: (r["eval"], r["run"], order.get(r["model"], 99)))

    fmt = {"rmse_model": "{:.4f}", "rmse_baseline": "{:.4f}", "skill": "{:+.1%}",
           "bias_model": "{:+.4f}", "bias_baseline": "{:+.4f}"}
    cols = ["eval", "run", "model", "epochs", "seed", "edge_pixels", "val_edge_pixels", "days",
            "rmse_model", "rmse_baseline", "skill", "bias_model", "bias_baseline"]
    print_table(rows, cols, fmt)

    ## pool seeds: same run name once "_seed<N>" is removed
    groups = {}
    for r in rows:
        key = (r["eval"], re.sub(r"_seed\d+", "", r["run"]), r["model"])
        groups.setdefault(key, []).append(r["skill"])
    pooled = [dict(eval=k[0], setting=k[1], model=k[2], n_seeds=len(v), mean=np.mean(v),
                   std=np.std(v, ddof=1) if len(v) > 1 else None, min=min(v), max=max(v))
              for k, v in groups.items() if len(v) > 1]
    if pooled:
        print("\nskill over seeds")
        print_table(pooled, ["eval", "setting", "model", "n_seeds", "mean", "std", "min", "max"],
                    {c: "{:+.1%}" for c in ("mean", "min", "max")} | {"std": "{:.1%}"})

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {args.csv}")


if __name__ == "__main__":
    main()
