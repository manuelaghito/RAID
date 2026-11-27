## write an evaluated run's fields to netCDF for browsing (ncview, Panoply, xarray): per day the 160m target,
## the bicubic baseline, the model prediction, their errors and where the model beats / loses to the baseline,
## plus per-cell RMSE maps per month and for the whole period. No GPU: reads what evaluate.py saved
## (<eval_dir>[/<var>]/pred_<var>.npy + dates.npy) and the preprocessed target/baseline files.
## usage:
#python3 scripts/eval_to_netcdf.py --config configs/step1b_unet.yaml --eval_dir experiments/step1b_unet/temp/ensemble/eval_test
## a period only (much smaller file), e.g. March 2026:
#python3 scripts/eval_to_netcdf.py --config configs/step1b_unet.yaml --eval_dir <...>/eval_test --start 20260301 --end 20260331
## several target variables (u+v): one file per variable, or --var u for one of them
## another region (evaluate.py --processed_dir): the region is read from the eval dir's evaluate.log ("data <root>"),
## its grid (lon/lat, projection offsets for --nk800) from the 160m reference file in <root>/preprocess_provenance.txt;
## --processed_dir / --reference_file override them, e.g. an A04 model scored on A05:
#python3 scripts/eval_to_netcdf.py --config configs/step1b_unet_uv.yaml --eval_dir <run>/u_v/ensemble/eval_test_trim10_processed_A05 --start 20260301 --end 20260331
## --nk800 adds the raw 800m NorKyst field before interpolation (read from data.raw_800m), each 800m cell copied
## onto the 5x5 160m cells it covers: blocky, no smoothing, NaN exactly where 800m is land (the 160m land mask
## is not applied, so the 800m coastline shows as it is)
## output: <eval_dir>[/<var>]/fields_<var>_<first>_<last>.nc unless --out is given (several variables: <out stem>_<var>.nc)

import argparse
import re
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
import yaml

from raid.data.file_utils import _extract_date, daily_filename, list_files
from raid.data.preprocessing import REFERENCE_FILE_160M, extract_variable, get_true_coords, set_rho_variables
from raid.metrics import UNITS

## the same folders as seen from the GPU/login nodes (/data/...) and from elsewhere (/gpfs/...)
ALIASES = [("/data/prosjekt/15915_RAID", "/gpfs/gpfs0/prosjekt/.prosjekt1/15915_RAID"),
           ("/data/osea/", "/gpfs/gpfs0/osea/")]


def resolve(path):
    """path as given if it exists, else its /data <-> /gpfs twin if that exists (else as given)."""
    p = str(path)
    if Path(p).exists():
        return Path(p)
    for a, b in ALIASES:
        for src, dst in ((a, b), (b, a)):
            if p.startswith(src) and Path(dst + p[len(src):]).exists():
                return Path(dst + p[len(src):])
    return Path(p)


def region_root(cfg, ev, override):
    ## the region evaluate.py scored: --processed_dir, else the last "data <root>" in the eval dir's log,
    ## else the config's (first) processed_dir
    if override:
        return resolve(override)
    for log in (ev / "evaluate.log", ev.parent / "evaluate.log"):
        if log.exists():
            roots = re.findall(r"^\s*data (\S+)\s*$", log.read_text(), re.M)
            if roots:
                return resolve(roots[-1])
    d = cfg["data"]["processed_dir"]
    return resolve(d if isinstance(d, str) else d[0])


def reference_160m(root, override):
    ## the region's 160m grid file: --reference_file, else the one preprocess.py recorded, else A04's
    if override:
        return resolve(override)
    prov = root / "preprocess_provenance.txt"
    if prov.exists():
        refs = re.findall(r"^raw_160m \S+\s+reference (\S+)", prov.read_text(), re.M)
        if refs:
            return resolve(refs[-1])
    return resolve(REFERENCE_FILE_160M)


def coords_160m(ref):
    ## 160m points in projection coordinates, with this file's own offsets (each region has its own)
    gm = ref["grid_mapping"].attrs if "grid_mapping" in ref.variables else {}
    if "false_easting" in gm and "false_northing" in gm:
        return ref["xi_rho"].values - gm["false_easting"], ref["eta_rho"].values - gm["false_northing"]
    return get_true_coords(ref, "160m")


def nk800_on_160m(path, name, level, x_tgt, y_tgt, margin=16):
    """Raw 800m field (surface by default, land NaN) sampled at the 160m points by nearest neighbour."""
    with xr.open_dataset(path) as ds:
        src = extract_variable(set_rho_variables(ds, "800m"), "800m", name, level)
        m = margin * float(src["xi_rho"][1] - src["xi_rho"][0])
        src = src.sel(xi_rho=slice(x_tgt.min() - m, x_tgt.max() + m),
                      eta_rho=slice(y_tgt.min() - m, y_tgt.max() + m))
        return src.sel(xi_rho=x_tgt, eta_rho=y_tgt, method="nearest").values.astype(np.float32)


def write_var(args, cfg, var, in_var, ev, root, ref_file, out):
    sea = np.load(root / "static" / "land_mask.npy").astype(bool)
    vdir = ev / var if (ev / var / f"pred_{var}.npy").exists() else ev
    dates = [str(d) for d in np.load(vdir / "dates.npy")]
    pred = np.load(vdir / f"pred_{var}.npy", mmap_mode="r")           # (T, H, W) physical units, land NaN
    if pred.shape[1:] != sea.shape:
        raise SystemExit(f"{vdir / f'pred_{var}.npy'} is {pred.shape[1:]}, {root} is {sea.shape}: "
                         f"wrong region, pass --processed_dir")
    keep = [i for i, d in enumerate(dates)
            if (args.start is None or d >= args.start) and (args.end is None or d <= args.end)]
    if not keep:
        raise SystemExit(f"no evaluated days in {args.start}..{args.end} (have {dates[0]}..{dates[-1]})")
    days = [dates[i] for i in keep]
    months = list(dict.fromkeys(d[:6] for d in days))
    H, W = sea.shape
    out = out or vdir / f"fields_{var}_{days[0]}_{days[-1]}.nc"

    nc = netCDF4.Dataset(out, "w")
    nc.title = f"RAID {var}: model vs bicubic baseline vs 160m target"
    nc.source = (f"eval_dir {ev.resolve()}, config {Path(args.config).resolve()}, data {root}, "
                 f"grid {ref_file}")
    nc.history = f"{datetime.now():%Y-%m-%d %H:%M} scripts/eval_to_netcdf.py"
    nc.createDimension("time", None)
    nc.createDimension("month", len(months))
    nc.createDimension("eta_rho", H)
    nc.createDimension("xi_rho", W)

    t = nc.createVariable("time", "f8", ("time",))
    t.units, t.calendar = "days since 1970-01-01", "standard"
    t[:] = netCDF4.date2num([datetime.strptime(d, "%Y%m%d") for d in days], t.units, t.calendar)
    m = nc.createVariable("month", "f8", ("month",))
    m.units, m.calendar, m.long_name = t.units, t.calendar, "first day of the month"
    m[:] = netCDF4.date2num([datetime.strptime(mm + "01", "%Y%m%d") for mm in months], m.units, m.calendar)

    ## lon/lat for map-aware viewers, from the region's 160m grid file
    coords = ""
    try:
        with xr.open_dataset(ref_file) as ref:
            if ref["lat_rho"].shape != (H, W):
                raise SystemExit(f"{ref_file} grid {ref['lat_rho'].shape} does not match {root} {(H, W)}; "
                                 f"pass --reference_file")
            for name in ("lon_rho", "lat_rho"):
                v = nc.createVariable(name, "f8", ("eta_rho", "xi_rho"))
                v.units = "degrees_east" if name == "lon_rho" else "degrees_north"
                v[:] = ref[name].values
            coords = "lon_rho lat_rho"
            if args.nk800:
                x_tgt, y_tgt = coords_160m(ref)
    except (OSError, KeyError) as e:
        if args.nk800:
            raise SystemExit(f"--nk800 needs the 160m grid file, could not read {ref_file}: {e}")
        print(f"no lon/lat (could not read {ref_file}: {e}); grid indices only")
    if args.nk800:
        years = sorted({int(d[:4]) for d in days})
        raw_cfg = dict(cfg["data"]["raw_800m"], path=str(resolve(cfg["data"]["raw_800m"]["path"])))
        raw_800m = {_extract_date(f): f for f in list_files(raw_cfg, years)}
        if missing := [d for d in days if d not in raw_800m]:
            raise SystemExit(f"no raw 800m file for {len(missing)} days, e.g. {missing[0]} ({raw_cfg['path']})")
        level = cfg["data"].get("levels", {}).get(in_var, -1)
    mask = nc.createVariable("mask_rho", "i1", ("eta_rho", "xi_rho"))
    mask.long_name, mask[:] = "sea mask (1 = sea)", sea.astype(np.int8)

    unit = UNITS.get(var, "")

    def field(name, dims, long_name, units=unit):
        ## float32, compressed, rounded to 0.001 units (invisible in plots, makes the file several times smaller)
        v = nc.createVariable(name, "f4", dims, zlib=True, complevel=4, least_significant_digit=3,
                              chunksizes=(1, H, W) if dims[0] in ("time", "month") else None,
                              fill_value=np.float32(np.nan))
        v.long_name, v.units = long_name, units
        if coords:
            v.coordinates = coords
        return v

    daily = ("time", "eta_rho", "xi_rho")
    v_t = field("target_160m", daily, f"160m NorFjords {var} (truth for training)")
    if args.nk800:
        v_n = field("nk800", daily, f"raw 800m NorKyst {in_var} before interpolation, nearest 800m cell "
                                     f"(NaN = 800m land; not masked with the 160m coast)")
    v_b = field("baseline_800int", daily, f"800m NorKyst {in_var} interpolated to 160m (bicubic)")
    v_p = field("model", daily, f"model prediction of {var}")
    v_c = field("correction", daily, "model - baseline: what the network adds")
    v_em = field("err_model", daily, "model - target")
    v_eb = field("err_baseline", daily, "baseline - target")
    v_d = field("abs_err_diff", daily, "|model - target| - |baseline - target|: < 0 model better, > 0 model worse")

    monthly = ("month", "eta_rho", "xi_rho")
    v_rm = field("rmse_model_month", monthly, "per-cell RMSE of the model over the month")
    v_rb = field("rmse_baseline_month", monthly, "per-cell RMSE of the baseline over the month")
    v_rd = field("rmse_diff_month", monthly, "rmse_model - rmse_baseline: < 0 model better")
    v_bm = field("bias_model_month", monthly, "per-cell mean of model - target over the month")
    v_bb = field("bias_baseline_month", monthly, "per-cell mean of baseline - target over the month")

    land = lambda a: np.where(sea, a, np.nan).astype(np.float32)
    sums = {k: np.zeros((H, W)) for k in ("sq_m", "sq_b", "e_m", "e_b")}
    total = {k: np.zeros((H, W)) for k in ("sq_m", "sq_b")}
    n = 0
    for j, (i, d) in enumerate(zip(keep, days)):
        tg = np.load(root / "160m" / daily_filename(var, d))
        b = np.load(root / "800m_on_160m_grid" / daily_filename(in_var, d))
        p = np.asarray(pred[i], dtype=np.float32)
        em, eb = p - tg, b - tg
        v_t[j], v_b[j], v_p[j] = land(tg), land(b), land(p)
        if args.nk800:
            v_n[j] = nk800_on_160m(raw_800m[d], in_var, level, x_tgt, y_tgt)      # 800m land mask only
        v_c[j], v_em[j], v_eb[j] = land(p - b), land(em), land(eb)
        v_d[j] = land(np.abs(em) - np.abs(eb))
        for k, a in (("sq_m", em ** 2), ("sq_b", eb ** 2), ("e_m", em), ("e_b", eb)):
            sums[k] += a
        n += 1
        ## month finished: write its maps and start the next one
        if j == len(days) - 1 or days[j + 1][:6] != d[:6]:
            k = months.index(d[:6])
            rm, rb = np.sqrt(sums["sq_m"] / n), np.sqrt(sums["sq_b"] / n)
            v_rm[k], v_rb[k], v_rd[k] = land(rm), land(rb), land(rm - rb)
            v_bm[k], v_bb[k] = land(sums["e_m"] / n), land(sums["e_b"] / n)
            total["sq_m"] += sums["sq_m"]; total["sq_b"] += sums["sq_b"]
            print(f"  {var} {d[:4]}-{d[4:6]}: {n} days, sea-mean RMSE model "
                  f"{np.sqrt((sums['sq_m'] / n)[sea].mean()):.4f} baseline {np.sqrt((sums['sq_b'] / n)[sea].mean()):.4f}")
            sums = {kk: np.zeros((H, W)) for kk in sums}
            n = 0

    ## whole selected period, no time dimension
    T = len(days)
    rm, rb = np.sqrt(total["sq_m"] / T), np.sqrt(total["sq_b"] / T)
    field("rmse_model", ("eta_rho", "xi_rho"), "per-cell RMSE of the model, whole period")[:] = land(rm)
    field("rmse_baseline", ("eta_rho", "xi_rho"), "per-cell RMSE of the baseline, whole period")[:] = land(rb)
    field("rmse_diff", ("eta_rho", "xi_rho"), "rmse_model - rmse_baseline, whole period: < 0 model better")[:] = land(rm - rb)
    nc.close()
    print(f"{var}: {T} days ({days[0]} -> {days[-1]}) written to {out} ({out.stat().st_size / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config the run was evaluated with")
    parser.add_argument("--eval_dir", type=str, required=True, help="evaluate.py output dir, e.g. <run>/eval_test")
    parser.add_argument("--start", type=str, default=None, help="first day YYYYMMDD (default: first evaluated day)")
    parser.add_argument("--end", type=str, default=None, help="last day YYYYMMDD (default: last evaluated day)")
    parser.add_argument("--var", type=str, default=None, help="one of the target variables (default: all)")
    parser.add_argument("--out", type=str, default=None, help="output .nc path")
    parser.add_argument("--processed_dir", type=str, default=None,
                        help="region the eval dir scored (default: from its evaluate.log)")
    parser.add_argument("--reference_file", type=str, default=None,
                        help="that region's 160m grid file (default: from its preprocess_provenance.txt)")
    parser.add_argument("--nk800", action="store_true",
                        help="also write the raw 800m input before interpolation (reads the raw NorKyst files)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    targets, inputs = cfg["data"]["target_vars"], cfg["data"]["input_vars"]
    if args.var is not None and args.var not in targets:
        parser.error(f"--var {args.var}: not a target variable of {args.config} ({targets})")
    ev = resolve(args.eval_dir)
    root = region_root(cfg, ev, args.processed_dir)
    ref_file = reference_160m(root, args.reference_file)
    print(f"region data {root}\ngrid {ref_file}")
    ## target k is the correction to input k, so its baseline is input_vars[k]
    todo = [(v, inputs[k]) for k, v in enumerate(targets) if args.var in (None, v)]
    for var, in_var in todo:
        if args.out is None:
            out = None
        elif len(todo) == 1:
            out = Path(args.out)
        else:
            out = Path(args.out).with_name(f"{Path(args.out).stem}_{var}.nc")
        write_var(args, cfg, var, in_var, ev, root, ref_file, out)


if __name__ == "__main__":
    main()
