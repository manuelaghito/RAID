# RAID



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.imr.no/ocenography/raid.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://git.imr.no/ocenography/raid/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
* [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***
# Data

## Processed data (data/processed/)

- 800m_on_160m_grid/, 160m/: daily surface fields, one file per day and variable (YYYYMMDD_<var>.npy),
  var = temp, salt, u, v (u/v moved onto the rho points)
- Units: °C, psu, m/s, NOT normalized
- 800m files: reprojected to the 160m grid, cropped to the
  NorFjords extent, interpolated (<method>) to 160m resolution
- Land cells: <NaN / fill value>; land mask in <file>, 1 = sea, 0 = land

## Normalization
- mean/std stored in <file>.npz (keys: <mean, std>)
- computed from sea cells in the training period only
- applied in Dataset at load time; model outputs are converted
  back to °C before computing metrics

## Train/val/test split
- by year, defined under `split` in the step1 configs
- `split.test`: held-out years, never used in training/CV
- `split.folds`: train/val years per CV fold, picked with `--fold` (required by train.py)

## Edge trimming
- `train.edge_pixels: N` in the step1 configs (default 0 = no trimming; CLI `--train_edge_pixels`)
- drops a frame of N cells along each grid edge from the **training** loss,
  to keep the 160m model's open-boundary zone from driving training
- loss mask only: inputs keep the edge cells (model still sees them as context),
  and normalization stats still use all sea cells
- with random patches, crops overlapping the frame just have fewer loss cells
  (masked_mse averages over the remaining sea cells)
- the validation loss that picks `best.pt` uses `train.val_edge_pixels` (default 0) instead,
  so runs with different `edge_pixels` select their checkpoint on the same cells
- evaluation scores all sea cells by default; `evaluate.py --edge_pixels N` scores the interior only
  (written to `eval_<split>_trim<N>/`), so runs trained with different `edge_pixels` compare on the same cells
- implemented by `trim_edges()` in `src/raid/data/dataset.py`

## Comparing runs
- `train.seed` (CLI `--seed`) fixes model init, crop positions and batch order; the spread between seeds
  is about as large as the effects tested so far, so compare settings over several seeds
- keep `train.epochs` identical: the cosine LR schedule is stretched over it, so it changes every epoch's LR
- train.py writes the resolved config to `<run>/config.yaml` and the settings to the top of `train.log`
- `--fold full` has no validation and keeps the last epoch; prefer the fold ensemble (`--fold ensemble`)
- results table over many runs, pooling runs that differ only in `_seed<N>`:
  `python3 scripts/summarize_runs.py <folder with runs> --eval eval_test_trim10`

## Diagnostics (CPU, from what evaluate.py saved)
- `scripts/consistency_stats.py`: splits the baseline (and model) error into a large-scale part, where 800m
  and 160m disagree at scales both resolve, and a small-scale part (sub-800m detail)
- `scripts/smallscale_stats.py`: does the model reproduce the 160m target's gradients and small-scale
  variance, and how are they tied to the bathymetry slope (RMSE alone rewards smooth fields)
- usage lines at the top of each script

## Metrics
- RMSE etc. over sea cells only, in physical units (°C, psu, m/s)
- baseline = interpolated 800m vs 160m

## Several target variables in one model (e.g. u and v)
- `data.target_vars: [u, v]` with `data.input_vars` starting with the same variables
  (`configs/step1b_unet_uv.yaml`): one model output per target variable, each a correction to the
  interpolated input of that variable
- the loss is the masked MSE over all outputs; each variable is normalized, so they weigh equally
- the run folder is named after all targets (`<output_dir>/u_v/<fold>`); evaluate.py writes one subfolder
  per variable (`eval_test/u/`, `eval_test/v/`), and summarize_runs.py shows one row per variable (`[u]`, `[v]`)

## Training on several regions
- `data.processed_dir` may be a list of regions (`configs/step1b_unet_uv_regions.yaml`: A04, A03, A05);
  training draws `train.crops_per_day` crops per day from every region, with the same year split in each,
  so the test years stay out everywhere
- normalization stats (also for the static fields) are pooled over all regions' training years and saved
  in `norm_stats.npz`; validation (picks `best.pt`) and evaluation use the first region
- `evaluate.py --processed_dir data/processed_A05` scores a model on another region's data, with the
  normalization of its training regions; results go to `eval_<split>[_trim<N>]_processed_A05/`

## Training on the predictable part only (target_adj)
- NorKyst and NorFjords are separate simulations: their eddies drift apart, and that large-scale mismatch
  (about 85 % of the interpolation error) cannot be predicted from the 800m input
- `scripts/make_target_adj.py` writes `<processed_dir>/160m_adj/`: target − (L − L̄), with L = S(target) − S(input)
  the day's large-scale mismatch (S = smoothing over sea, σ = 1.6 km) and L̄ its per-cell mean over the training
  years (the systematic, predictable part, kept)
- train with `data.target_dir: 160m_adj` (`configs/step1b_unet_uv_adj.yaml`); `evaluate.py --target_dir 160m`
  scores against the real target as well (`eval_<split>..._target160m/`)

## Run tools
- `scripts/seed_study.sh`: train (folds val2024 + val2025) and evaluate one config for several seeds
  (`LABEL`, `SEEDS`, `EPOCHS`); `scripts/queues/`: the run queues as launched, by date
- `scripts/eval_to_netcdf.py`: netCDF of an evaluation for ncview/Panoply, day by day target, baseline,
  model, errors and monthly RMSE maps, on the region that was scored (`--nk800` adds the raw 800m field)


***
# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
