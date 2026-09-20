#!/usr/bin/env python3
"""
MTO path-specific FNA bounds — orthogonal (DR) estimator only.

Estimates the Direct, Indirect, and Total Fraction-Negatively-Affected (FNA)
bounds on the Moving-to-Opportunity data using the one-step doubly-robust
covariate-assisted Makarov estimator. Reports per-pair train / held-out CRPS
pseudo-loss, the estimated bounds (with influence-function SEs), and the
conservative confidence sets.

No exploratory analysis and no plug-in estimator are included.

This is a packaged version of the FNA-bounds path in the experiment notebook
(sample construction + DR triplet + tighter_bounds_from_ccdf_triplet_dr). It
imports the shared `functions` / `crps_mu_models` modules from the project.

Example
-------
    python experiments/mto/run_mto_fna.py \
        --data data/mto.dta \
        --out  results/mto
"""

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MTO DR FNA bounds (direct/indirect/total).")
    p.add_argument("--data", default="data/mto.dta",
                   help="Path to the MTO Stata file (mto.dta).")
    p.add_argument("--code-dir", default=None,
                   help="Directory containing functions.py / crps_mu_models.py. "
                        "Defaults to <repo>/src.")
    p.add_argument("--out", default="results/mto",
                   help="Directory for the JSON + text results.")

    # Estimation hyper-parameters (defaults match the notebook).
    p.add_argument("--K", type=int, default=5, help="Cross-fitting folds.")
    p.add_argument("--n-mc", type=int, default=500, help="MC draws for the mediator integral.")
    p.add_argument("--epochs-stage1", type=int, default=50,
                   help="First-stage (pseudo-outcome) training epochs.")
    p.add_argument("--epochs-stage2", type=int, default=150,
                   help="Second-stage (conditional-CDF / CRPS) training epochs.")
    p.add_argument("--hidden", type=int, default=128, help="Hidden width of the CRPS net.")
    p.add_argument("--n-grid", type=int, default=900,
                   help="Approx. number of threshold-grid quantiles (central region).")
    p.add_argument("--seed", type=int, default=0, help="Seed (KFold shuffle + torch/np init).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Level for the conservative confidence set (1-alpha coverage).")
    p.add_argument("--print-every", type=int, default=50,
                   help="Stage-2 progress print frequency (used only if supported).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MTO sample construction (faithful to the notebook's bounds cell)
# ──────────────────────────────────────────────────────────────────────────────
def build_mto_sample(data_path):
    """Return X, A, M, Y, X_cols for the experimental-vs-control arms.

    A : 1 = experimental (low-poverty) voucher offer, 0 = control.
    M : 1{duration-weighted neighbourhood poverty < control median}.
    Y : standardised adult mental-health index (higher = better).
    X : site dummies (NY = reference) + baseline covariates.
    """
    df = pd.read_stata(data_path, convert_categoricals=False)

    # ra_group: 1 = Experimental voucher, 2 = Section 8, 3 = Control.
    data = df[df["ra_group"].isin([1, 2, 3])].copy()
    data["A"] = data["ra_group"].isin([1, 2]).astype(int)

    poverty = pd.to_numeric(data["ps_f_c9010t_perpov_dw"], errors="coerce")
    control_median = poverty[data["A"] == 0].median()
    # data["M"] = (poverty < control_median).astype(int)
    data["M"] = data["ps_f_c9010t_perpov_dw"]
    data["Y"] = data["ps_f_mh_idx_z_ad"]

    # Site dummies (drop New York, site 5, as the reference category).
    site_dummies = pd.get_dummies(data["ra_site"], prefix="X_site", dtype=int)
    if "X_site_5" in site_dummies.columns:
        site_dummies = site_dummies.drop(columns=["X_site_5"])
    site_dummies = site_dummies.rename(columns={
        "X_site_1": "X_site_baltimore",
        "X_site_2": "X_site_boston",
        "X_site_3": "X_site_chicago",
        "X_site_4": "X_site_los_angeles",
    })

    X_BASELINE_COLS = [
        "ps_x_rad_ad_male", "ps_x_rad_ad_le_35", "ps_x_rad_ad_36_40",
        "ps_x_rad_ad_41_45", "ps_x_rad_ad_46_50",
        "ps_x_rad_ad_ethrace_black_nh", "ps_x_rad_ad_ethrace_hisp",
        "ps_x_f_ad_working", "ps_x_f_ad_edged", "ps_x_f_ad_edgradhs",
        "ps_x_f_ad_edinsch", "ps_x_f_ad_nevmarr", "ps_x_f_ad_parentu18",
        "ps_x_f_hood_5y", "ps_x_f_hood_chat", "ps_x_f_hood_nbrkid",
        "ps_x_f_hood_nofamily", "ps_x_f_hood_nofriend", "ps_x_f_hood_unsafenit",
        "ps_x_f_hood_verydissat",
        "ps_x_f_hh_afdc", "ps_x_f_hh_car", "ps_x_f_hh_disabl",
        "ps_x_f_hh_noteens", "ps_x_f_hh_size2", "ps_x_f_hh_size3",
        "ps_x_f_hh_size4", "ps_x_f_hh_victim",
        "ps_x_f_hous_fndapt", "ps_x_f_hous_mov3tm", "ps_x_f_hous_movdrgs",
        "ps_x_f_hous_movschl", "ps_x_f_hous_sec8bef",
    ]
    X_BASELINE_COLS = [c for c in X_BASELINE_COLS if c in data.columns]

    baseline_X = data[X_BASELINE_COLS].copy()
    baseline_X = baseline_X.rename(
        columns={c: "X_" + c.replace("ps_", "") for c in X_BASELINE_COLS})

    X_df = pd.concat([site_dummies.reset_index(drop=True),
                      baseline_X.reset_index(drop=True)], axis=1)
    X_cols = list(X_df.columns)

    data = data.reset_index(drop=True)
    data = pd.concat([data, X_df], axis=1)

    xamy = data[X_cols + ["A", "M", "Y"]].copy()
    for col in xamy.columns:
        xamy[col] = pd.to_numeric(xamy[col], errors="coerce")
    xamy = xamy.dropna().reset_index(drop=True)

    X = xamy[X_cols].to_numpy(dtype=float)
    A = xamy["A"].to_numpy(dtype=int)
    M = xamy["M"].to_numpy(dtype=float)
    Y = xamy["Y"].to_numpy(dtype=float)
    return X, A, M, Y, X_cols


def build_threshold_grid(Y, n_central=900):
    """Quantile threshold grid: sparse tails, dense centre (notebook convention)."""
    Y_clean = np.asarray(Y, dtype=float)
    Y_clean = Y_clean[np.isfinite(Y_clean)]
    q_grid = np.concatenate([
        np.linspace(0.001, 0.01, 50),     # sparse lower tail
        np.linspace(0.01, 0.99, n_central),  # dense central region
        np.linspace(0.99, 0.999, 50),     # sparse upper tail
    ])
    q_grid = np.unique(q_grid)
    Tn = np.quantile(Y_clean, q_grid)
    return np.unique(Tn)


# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 call that works whether or not the installed version returns curves
# ──────────────────────────────────────────────────────────────────────────────
def fit_stage2(crossfit_fn, phi, X, Tn, splits, epochs, hidden, print_every):
    sig = inspect.signature(crossfit_fn)
    kwargs = {}
    if "epochs" in sig.parameters:
        kwargs["epochs"] = epochs
    if "hidden" in sig.parameters:
        kwargs["hidden"] = hidden
    if "print_every" in sig.parameters:
        kwargs["print_every"] = print_every

    res = crossfit_fn(phi, X, Tn, splits, **kwargs)
    if isinstance(res, tuple):
        dr_ccdfs = res[0]
        curves = res[1] if len(res) > 1 else None
    else:
        dr_ccdfs, curves = res, None
    return dr_ccdfs, curves


def summarize_curves(curves):
    """Pull final train / held-out CRPS pseudo-loss and held-out minimum per pair."""
    if not curves:
        return None
    out = {}
    for pair, c in curves.items():
        if not isinstance(c, dict) or "test_mean" not in c:
            continue
        train = np.asarray(c.get("train_mean", []), dtype=float)
        test = np.asarray(c["test_mean"], dtype=float)
        epoch = np.asarray(c.get("epoch", np.arange(len(test))))
        e_min = int(np.argmin(test))
        out[str(pair)] = {
            "train_final": float(train[-1]) if train.size else None,
            "heldout_final": float(test[-1]),
            "heldout_min": float(test[e_min]),
            "heldout_min_epoch": int(epoch[e_min]),
            "heldout_rise_after_min": float(test[-1] - test[e_min]),
        }
    return out


def conservative_set(L, U, se_L, se_U, alpha):
    """[max(0, L - z*se_L), min(1, U + z*se_U)] — covers the identified set."""
    from scipy.stats import norm
    z = norm.ppf(1.0 - alpha / 2.0)
    lo = max(0.0, L - z * se_L)
    hi = min(1.0, U + z * se_U)
    return lo, hi, z


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    code_dir = args.code_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src")
    sys.path.insert(0, code_dir)

    import torch  # noqa: F401  (imported for seeding + to surface device info)
    import functions as F
    import crps_mu_models as C
    from functions import crossfit_pseudo_outcomes, tighter_bounds_from_ccdf_triplet_dr
    from crps_mu_models import crossfit_conditional_cdfs_crps

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print("MTO path-specific FNA bounds — orthogonal (DR) estimator")
    print("=" * 78)
    print(f"device          : {F.DEVICE}")
    print(f"data            : {args.data}")
    print(f"code-dir        : {code_dir}")
    print(f"K / n_mc        : {args.K} / {args.n_mc}")
    print(f"epochs (s1/s2)  : {args.epochs_stage1} / {args.epochs_stage2}")
    print(f"hidden / seed   : {args.hidden} / {args.seed}")
    print("-" * 78)

    # 1. Sample -----------------------------------------------------------------
    X, A, M, Y, X_cols = build_mto_sample(args.data)
    Tn = build_threshold_grid(Y, n_central=args.n_grid)
    print(f"N               : {len(A)}")
    print(f"covariates      : {X.shape[1]}  ({', '.join(X_cols[:6])}, ...)")
    print(f"P(A=1)          : {A.mean():.4f}")
    print(f"P(M=1)          : {M.mean():.4f}")
    print(f"threshold grid  : {len(Tn)} points in [{Tn.min():.3f}, {Tn.max():.3f}]")
    print("-" * 78)

    t0 = time.time()

    # 2. Stage 1: cross-fitted DR pseudo-outcomes -------------------------------
    print("Stage 1: cross-fitted EIF pseudo-outcomes ...")
    phi, splits = crossfit_pseudo_outcomes(
        X, A, M, Y, Tn,
        K=args.K, n_mc=args.n_mc, epochs=args.epochs_stage1,
        t_learner=False, random_state=args.seed,
    )
    print(f"   phi keys      : {list(phi.keys())}  shapes {phi[(0, 0)].shape}")

    # 3. Stage 2: conditional CDFs via CRPS regression --------------------------
    print("Stage 2: conditional-CDF (CRPS) regression ...")
    dr_ccdfs, curves = fit_stage2(
        crossfit_conditional_cdfs_crps, phi, X, Tn, splits,
        epochs=args.epochs_stage2, hidden=args.hidden, print_every=args.print_every,
    )
    elapsed = time.time() - t0
    print(f"   fit time      : {elapsed:.1f}s")
    print("-" * 78)

    # 4. Train / held-out loss --------------------------------------------------
    loss_summary = summarize_curves(curves)
    print("Train / held-out CRPS pseudo-loss:")
    if loss_summary is None:
        print("   [installed crps_mu_models version does not return loss curves; "
              "skipping. Upgrade to the curve-returning version to report this.]")
    else:
        print(f"   {'pair':>6}  {'train':>10}  {'held-out':>10}  "
              f"{'min':>10}  {'min@ep':>7}  {'rise':>9}")
        for pair, s in loss_summary.items():
            tr = "n/a" if s["train_final"] is None else f"{s['train_final']:.5f}"
            print(f"   {pair:>6}  {tr:>10}  {s['heldout_final']:>10.5f}  "
                  f"{s['heldout_min']:>10.5f}  {s['heldout_min_epoch']:>7d}  "
                  f"{s['heldout_rise_after_min']:>9.5f}")
    print("-" * 78)

    # 5. DR bounds + conservative sets ------------------------------------------
    dr_bds = tighter_bounds_from_ccdf_triplet_dr(
        dr_ccdfs[(0, 0)], dr_ccdfs[(1, 0)], dr_ccdfs[(1, 1)], phi=phi)

    print(f"Estimated FNA bounds (DR) and {int((1 - args.alpha) * 100)}% "
          f"conservative sets:")
    print(f"   {'effect':>9}  {'L':>8}  {'U':>8}  {'se_L':>8}  {'se_U':>8}  "
          f"{'cons. set':>20}")
    results = {}
    for effect in ("Direct", "Indirect", "Total"):
        L, U, se_L, se_U = dr_bds[effect]
        lo, hi, z = conservative_set(L, U, se_L, se_U, args.alpha)
        crossed = L > U
        print(f"   {effect:>9}  {L:>8.4f}  {U:>8.4f}  {se_L:>8.4f}  {se_U:>8.4f}  "
              f"[{lo:.4f}, {hi:.4f}]" + ("  (L>U)" if crossed else ""))
        results[effect] = {
            "L": float(L), "U": float(U),
            "se_L": float(se_L), "se_U": float(se_U),
            "crossed": bool(crossed),
            "conservative_set": [float(lo), float(hi)],
        }
    print("=" * 78)

    # 6. Persist ----------------------------------------------------------------
    payload = {
        "config": vars(args),
        "device": F.DEVICE,
        "n": int(len(A)),
        "n_covariates": int(X.shape[1]),
        "covariates": X_cols,
        "p_A1": float(A.mean()),
        "p_M1": float(M.mean()),
        "n_thresholds": int(len(Tn)),
        "z_score": float(z),
        "fit_seconds": float(elapsed),
        "loss": loss_summary,
        "bounds": results,
    }
    json_path = os.path.join(args.out, "mto_fna_dr_results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results written to {json_path}")


if __name__ == "__main__":
    main()

