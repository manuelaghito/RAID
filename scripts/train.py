import sys
import argparse
from pathlib import Path

import yaml
import torch

from raid.config import apply_fold, dict_to_namespace
from raid.data.dataset import build_dataloaders
from raid.log import tee_output
from raid.models import build_model
from raid.trainer import fit, seed_everything


# usage: CUDA_VISIBLE_DEVICES=1 
#        python3 scripts/train.py --config configs/step1a_cnn.yaml --fold val2025
#        python3 scripts/train.py --config configs/step1a_cnn.yaml --fold val2025 --output_dir ...
## --fold full trains the final model on split.full.train (no val; saves the last epoch as best.pt):
#        python3 scripts/train.py --config configs/step1a_cnn.yaml --fold full
#        python3 scripts/train.py --config configs/step1b_unet.yaml --fold val2024 --output_dir ...
## --seed overrides train.seed, e.g. several seeds of the same setup into separate folders:
#        bash scripts/train_folds.sh configs/step1b_unet.yaml --seed 1 --output_dir .../seed1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override train.seed")
    parser.add_argument("--train_edge_pixels", type=int, default=None,
                        help="Override train.edge_pixels (training loss only; not the same as evaluate.py --edge_pixels)")
    parser.add_argument("--val_edge_pixels", type=int, default=None,
                        help="Override train.val_edge_pixels (cells left out of the val loss that picks best.pt)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--fold", type=str, required=True,
                        help="CV fold name from split.folds, e.g. val2025 or val2024, or 'full'")    
    
    return parser.parse_args()


def load_config(config_path, args):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.train_edge_pixels is not None:
        cfg["train"]["edge_pixels"] = args.train_edge_pixels
    if args.val_edge_pixels is not None:
        cfg["train"]["val_edge_pixels"] = args.val_edge_pixels
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    apply_fold(cfg, args.fold)

    test_years = set(cfg["split"].get("test", {}).get("years", []))
    train_years, val_years = set(cfg["split"]["train"]["years"]), set(cfg["split"]["val"]["years"])
    assert not (train_years & val_years), f"years {train_years & val_years} in both train and val"
    used = train_years | val_years
    assert not (used & test_years), f"test years {used & test_years} used in train/val"

    return cfg


def main():
    args = parse_args()
    raw = load_config(args.config, args)
    cfg = dict_to_namespace(raw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    tee_output(out / "train.log")
    ## the config as actually run (fold years, output_dir and command-line overrides resolved),
    ## so a run folder always records what produced it; overwritten by a rerun in the same folder
    (out / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    t = cfg.train
    seed = getattr(t, "seed", None)
    print(f"device {device}  train {cfg.split.train.years}  val {cfg.split.val.years}  output_dir {cfg.output_dir}")
    print(f"seed {seed}  epochs {t.epochs}  lr {t.lr}  batch_size {t.batch_size}  "
          f"patch_size {getattr(t, 'patch_size', None)}  edge_pixels {getattr(t, 'edge_pixels', 0)}  "
          f"val_edge_pixels {getattr(t, 'val_edge_pixels', 0)}")
    if seed is not None:
        seed_everything(seed)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    fit(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()