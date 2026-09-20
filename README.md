# Path-specific FNA bounds

Code for the paper on estimating bounds for the path-specific **fraction negatively
affected (FNA)**. The bounds are covariate-assisted Makarov bounds, estimated
with a two-stage orthogonal estimator with cross-fitting,
and compared against a plug-in (g-formula) baseline.

## Repository layout

```
src/
  functions.py              nuisance models, EIF pseudo-outcomes, cross-fitting,
                            plug-in estimator, Makarov bounds, oracle quantities
  crps_mu_models.py         monotone neural conditional-CDF model trained with CRPS
                            (first-stage outcome CDF and second-stage regression)
experiments/
  simulation/fna_simulation.py   simulation study (all DGPs)
  mto/run_mto_fna.py             Moving to Opportunity application
slurm/
  run_simulation.slurm      array job: one task per DGP, paper settings
  run_mto.slurm             MTO job, paper settings
data/                       place mto.dta here (not distributed, see data/README.md)
results/                    outputs (git-ignored)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is used automatically when available; everything also runs on CPU, more slowly.


### Simulation study

Each run writes to `results/simulation/<dgp>/`. It produces a per-seed results
pickle, `final_statistics.pkl`, and a printed summary table.

| Design in paper | Flag |
|---|---|
| 3 covariates | `--dgp-type linear --cov 3cov` |
| 5 covariates | `--dgp-type linear --cov 5cov` |
| 10 covariates | `--dgp-type linear --cov 10cov` |
| Polynomial (appendix) | `--dgp-type polynomial` |
| Sinusoidal (appendix) | `--dgp-type sinusoidal` |

The paper settings for one design look like this:

```bash
python experiments/simulation/fna_simulation.py \
    --output-dir results/simulation --base-seed 2026 --n-sims 300 \
    --n-sample 5000 --n-mc 5 --epochs1 30 --epochs2 50 \
    --hidden 64 --n-layers 2 --lr 1e-3 --batch-size 256 --weight-decay 1e-4 \
    --dgp-type linear --cov 3cov
```

To run all five designs as a SLURM array, use `sbatch slurm/run_simulation.slurm`.


### MTO application

Place the data file at `data/mto.dta` (see `data/README.md`), then run:

```bash
python experiments/mto/run_mto_fna.py --data data/mto.dta --out results/mto \
    --K 5 --n-mc 10000 --epochs-stage1 50 --epochs-stage2 150 \
    --hidden 128 --n-grid 900 --seed 0
```

Alternatively, run `sbatch slurm/run_mto.slurm`. Results are written to
`results/mto/mto_fna_dr_results.json`.

## Hyperparameters used in the paper

| | Simulation | MTO |
|---|---|---|
| Cross-fitting folds K | 5 | 5 |
| Stage-1 / stage-2 epochs | 30 / 50 | 50 / 150 |
| CRPS net, stage 1 (hidden × layers) | 64 × 2 | 64 × 2 |
| CRPS net, stage 2 (hidden × layers) | 64 × 2 | 128 × 2 |
| Adam lr / weight decay / batch | 1e-3 / 1e-4 / 256 | 1e-3 / 1e-4 / 256 |
| MC draws for mediator integral | 5 (binary M: exact, unused) | 10000 |
| Replications / seeds | 300 (seeds 2026–2325) | 1 (seed 0) |

