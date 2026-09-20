#!/usr/bin/env python3
"""Path-specific FNA Makarov Bounds - Simulation Study (GPU script).

Converted from the Colab notebook `FNA_Simulation_Colab_updated.ipynb` into a
plain, headless, GPU-runnable script.

Estimates direct / indirect / total FNA bounds via:
  - DR (proposed): cross-fit EIF pseudo-outcomes, second-stage conditional CDFs,
    and Wald endpoint CIs from influence-function standard errors.
  - Plug-in: g-formula conditional CDFs with Wald endpoint CIs.

Coverage is reported against the covariate-assisted oracle bounds and the true FNA.

----------------------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------------------
This script depends on two project modules in <repo>/src (or pointed to with
--code-dir):

    functions.py
    crps_mu_models.py

Python deps: torch, numpy, pandas, scipy, tqdm, matplotlib.

----------------------------------------------------------------------------
EXAMPLE
----------------------------------------------------------------------------
    # quick smoke test (a couple of seeds, small n)
    python experiments/simulation/fna_simulation.py --n-sims 2 --n-sample 5000 --n-mc 500 --epochs1 30 --epochs2 30 \
        --output-dir results/smoke

    # full study (paper settings: see slurm/run_simulation.slurm)
    python experiments/simulation/fna_simulation.py --n-sims 300 --n-sample 5000 --n-mc 5 --epochs1 30 --epochs2 50 \
        --dgp-type linear --cov 3cov --output-dir results/simulation
"""

import argparse
import os
import sys
import time
import pickle

# --- Headless plotting backend MUST be set before any pyplot import (incl. by
#     the project modules), otherwise a GPU/batch node with no display errors. ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Resolve where the project modules (functions.py, crps_mu_models.py) live, and
# put that directory on sys.path BEFORE importing them. We do a tiny manual scan
# of argv for --code-dir so this works regardless of the current directory.
# ---------------------------------------------------------------------------
def _resolve_code_dir(argv):
    code_dir = None
    for i, a in enumerate(argv):
        if a == "--code-dir" and i + 1 < len(argv):
            code_dir = argv[i + 1]
        elif a.startswith("--code-dir="):
            code_dir = a.split("=", 1)[1]
    if code_dir is None:
        # default: <repo>/src (this script lives in <repo>/experiments/simulation)
        code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "src")
    return os.path.abspath(code_dir)


_CODE_DIR = _resolve_code_dir(sys.argv)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
# also allow running from within the project dir
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# --- Standard scientific stack ---
import numpy as np
import pandas as pd  # noqa: F401  (kept for parity with the notebook)
import torch
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import norm as _norm
from scipy.special import expit
from scipy.integrate import quad

# --- Project modules (the GPU model code lives here) ---
try:
    import functions  # noqa: F401
    from functions import *          # noqa: F401,F403
    from crps_mu_models import *      # noqa: F401,F403
except ModuleNotFoundError as exc:
    sys.stderr.write(
        "\nERROR: could not import the project modules.\n"
        f"  Looked in: {_CODE_DIR}\n"
        "  Make sure 'functions.py' and 'crps_mu_models.py' are in that folder,\n"
        "  or pass --code-dir /path/to/those/files.\n"
        f"  Underlying error: {exc}\n\n"
    )
    raise


# ===========================================================================
# 1. Data-generating process  (notebook cell 7 -- the *tuned* DGP is kept)
# ===========================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ── shared structural parameters (same for both DGPs) ─────────────────────────
_DGP_SHARED = {
    "sigma_y": 0.25,
    "sigma_m": 0.15,
    "k":       6.0,
    "d_left":  3.00,  "d_right": -0.75,
    "q_left": -1.10,  "q_right":  1.20,
    "am_interaction": 0.10,
    "m_intercept": -0.90,
    "m_a_effect":   2.00,
}

# ── 1-covariate DGP ────────────────────────────────────────────────────────────
DGP_1COV = {
    **_DGP_SHARED,
    "m_x_effect":  0.15,
    "a_intercept": 0.20,
    "a_x_effect":  0.35,
    "y_x1_effect": 0.10,
}
DGP_3COV = {
    **_DGP_SHARED,
    # covariate generation
    "x3_on_x1":  0.40, "x3_on_x2": -0.30,   # X3 ~ Bern(sigmoid(...))
    # mediator model
    "m_x1_effect":  0.15, "m_x2_effect":  0.40, "m_x3_effect": -0.30,
    # treatment model
    "a_intercept":  0.20,
    "a_x1_effect":  0.35, "a_x2_effect": -0.25, "a_x3_effect":  0.50,
    # outcome baseline
    "y_x1_effect":  0.10, "y_x2_effect":  0.30, "y_x3_effect": -0.20,
}

def _p_med_3cov(x1, x2, x3, ap):
    d = DGP_3COV
    return sigmoid(
        d["m_intercept"] + d["m_a_effect"]*ap
        + d["m_x1_effect"]*x1 + d["m_x2_effect"]*x2 + d["m_x3_effect"]*x3
    )

def _bX_3cov(x1, x2, x3):
    d = DGP_3COV
    return d["y_x1_effect"]*x1 + d["y_x2_effect"]*x2 + d["y_x3_effect"]*x3

def _sample_data_3cov(n, seed=0, sigma_y=None):
    d  = DGP_3COV
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    X3 = rng.binomial(1, sigmoid(d["x3_on_x1"]*X1 + d["x3_on_x2"]*X2)).astype(float)

    A = rng.binomial(1, sigmoid(
        d["a_intercept"]
        + d["a_x1_effect"]*X1 + d["a_x2_effect"]*X2 + d["a_x3_effect"]*X3
    ))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_3cov(X1, X2, X3, 0);  pM1i = _p_med_3cov(X1, X2, X3, 1)
    M0   = (UM < pM0i).astype(int);     M1   = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(X1);  bX = _bX_3cov(X1, X2, X3)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    return dict(X1=X1, X2=X2, X3=X3,
                A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_3cov(t_grid, X, a=0, ap=0):
    """X: (n, 3) → (n, T)."""
    d   = DGP_3COV
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    X   = np.asarray(X)
    x1, x2, x3 = [X[:, i:i+1] for i in range(3)]
    t   = np.asarray(t_grid)[None, :]
    p        = _p_med_3cov(x1, x2, x3, ap)
    d_x, q_x = smooth_effects(x1)
    mu0 = _bX_3cov(x1, x2, x3) + d_x*a
    mu1 = mu0 + q_x + d["am_interaction"]*a
    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)

# ── 5-covariate DGP ────────────────────────────────────────────────────────────
# X1 ~ N(0,1)          drives heterogeneous effects via smooth_effects
# X2 ~ N(0,1)          continuous confounder
# X3 ~ Bernoulli(0.5)  binary covariate
# X4 ~ N(0,1)          continuous confounder
# X5 ~ Bernoulli(0.5)  binary covariate
DGP_5COV = {
    **_DGP_SHARED,
    "m_x1_effect":  0.15, "m_x2_effect":  0.40, "m_x3_effect": -0.30,
    "m_x4_effect":  0.20, "m_x5_effect": -0.10,
    "a_intercept":  0.20,
    "a_x1_effect":  0.35, "a_x2_effect": -0.25, "a_x3_effect":  0.50,
    "a_x4_effect": -0.15, "a_x5_effect":  0.30,
    "y_x1_effect":  0.10, "y_x2_effect":  0.30, "y_x3_effect": -0.20,
    "y_x4_effect":  0.25, "y_x5_effect": -0.15,
}

def smooth_effects(x1):
    """Heterogeneous d(x1) and q(x1) — shared by both DGPs."""
    w   = sigmoid(DGP["k"] * x1)
    d_x = DGP["d_left"] + (DGP["d_right"] - DGP["d_left"]) * w
    q_x = DGP["q_left"] + (DGP["q_right"] - DGP["q_left"]) * w
    return d_x, q_x


# ── 1-cov internals ────────────────────────────────────────────────────────────

def _p_med_1cov(x1, ap):
    d = DGP_1COV
    return sigmoid(d["m_intercept"] + d["m_a_effect"]*ap + d["m_x_effect"]*x1)

def _bX_1cov(x1):
    return DGP_1COV["y_x1_effect"] * x1

def _sample_data_1cov(n, seed=0, sigma_y=None):
    d  = DGP_1COV
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    X1 = rng.standard_normal(n)
    A  = rng.binomial(1, sigmoid(d["a_intercept"] + d["a_x_effect"]*X1))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_1cov(X1, 0);  pM1i = _p_med_1cov(X1, 1)
    M0   = (UM < pM0i).astype(int);  M1 = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(X1);  bX = _bX_1cov(X1)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    return dict(X1=X1, A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_1cov(t_grid, X1, a=0, ap=0):
    """Closed-form oracle CDF for 1-cov DGP. X1: (n,) → returns (n, T)."""
    d   = DGP_1COV
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    x   = np.asarray(X1)[:, None]       # (n, 1)
    t   = np.asarray(t_grid)[None, :]   # (1, T)
    p        = _p_med_1cov(x, ap)
    d_x, q_x = smooth_effects(x)
    mu0 = _bX_1cov(x) + d_x*a
    mu1 = mu0 + q_x + d["am_interaction"]*a
    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)


# ── 5-cov internals ────────────────────────────────────────────────────────────

def _p_med_5cov(x1, x2, x3, x4, x5, ap):
    d = DGP_5COV
    return sigmoid(
        d["m_intercept"] + d["m_a_effect"]*ap
        + d["m_x1_effect"]*x1 + d["m_x2_effect"]*x2 + d["m_x3_effect"]*x3
        + d["m_x4_effect"]*x4 + d["m_x5_effect"]*x5
    )

def _bX_5cov(x1, x2, x3, x4, x5):
    d = DGP_5COV
    return (d["y_x1_effect"]*x1 + d["y_x2_effect"]*x2 + d["y_x3_effect"]*x3
            + d["y_x4_effect"]*x4 + d["y_x5_effect"]*x5)

def _sample_data_5cov(n, seed=0, sigma_y=None):
    d  = DGP_5COV
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    X3 = rng.binomial(1, 0.5, size=n).astype(float)
    X4 = rng.standard_normal(n)
    X5 = rng.binomial(1, 0.5, size=n).astype(float)

    A = rng.binomial(1, sigmoid(
        d["a_intercept"]
        + d["a_x1_effect"]*X1 + d["a_x2_effect"]*X2 + d["a_x3_effect"]*X3
        + d["a_x4_effect"]*X4 + d["a_x5_effect"]*X5
    ))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_5cov(X1, X2, X3, X4, X5, 0)
    pM1i = _p_med_5cov(X1, X2, X3, X4, X5, 1)
    M0   = (UM < pM0i).astype(int);  M1 = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(X1)
    bX = _bX_5cov(X1, X2, X3, X4, X5)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    return dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5,
                A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_5cov(t_grid, X, a=0, ap=0):
    """Closed-form oracle CDF for 5-cov DGP. X: (n, 5) → returns (n, T)."""
    d   = DGP_5COV
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    X   = np.asarray(X)
    x1, x2, x3, x4, x5 = [X[:, i:i+1] for i in range(5)]   # each (n, 1)
    t   = np.asarray(t_grid)[None, :]   # (1, T)
    p        = _p_med_5cov(x1, x2, x3, x4, x5, ap)
    d_x, q_x = smooth_effects(x1)
    mu0 = _bX_5cov(x1, x2, x3, x4, x5) + d_x*a
    mu1 = mu0 + q_x + d["am_interaction"]*a
    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)

DGP_10COV = {
    **_DGP_SHARED,
    # covariate generation — dependency coefficients
    "x3_on_x1":   0.40, "x3_on_x2":  -0.30,   # X3 ~ Bern(sigmoid(...))
    "x4_on_x1":   0.50, "x4_on_x2":   0.30,   # X4 ~ N(mu, 1)
    "x6_on_x3":   0.60, "x6_on_x5":  -0.40,   # X6 ~ N(mu, 1)
    "x7_on_x4":   0.30, "x7_on_x6":   0.20,   # X7 ~ N(mu, 1)
    "x8_on_x2":   0.50, "x8_on_x7":   0.40,   # X8 ~ Bern(sigmoid(...))
    "x9_on_x1":   0.40, "x9_on_x8":  -0.30,   # X9 ~ N(mu, 1)
    "x10_on_x4":  0.30, "x10_on_x9": -0.50,   # X10 ~ Bern(sigmoid(...))
    # mediator model
    "m_x1_effect":  0.15, "m_x2_effect":  0.30, "m_x3_effect": -0.25,
    "m_x4_effect":  0.20, "m_x5_effect": -0.10, "m_x6_effect":  0.15,
    "m_x7_effect": -0.20, "m_x8_effect":  0.25, "m_x9_effect": -0.15,
    "m_x10_effect": 0.10,
    # treatment model
    "a_intercept":   0.20,
    "a_x1_effect":   0.35,  "a_x2_effect": -0.20, "a_x3_effect":  0.40,
    "a_x4_effect":  -0.15,  "a_x5_effect":  0.25, "a_x6_effect": -0.10,
    "a_x7_effect":   0.30,  "a_x8_effect": -0.20, "a_x9_effect":  0.15,
    "a_x10_effect": -0.25,
    # outcome baseline
    "y_x1_effect":  0.10,  "y_x2_effect":  0.25, "y_x3_effect": -0.15,
    "y_x4_effect":  0.20,  "y_x5_effect": -0.10, "y_x6_effect":  0.15,
    "y_x7_effect": -0.20,  "y_x8_effect":  0.10, "y_x9_effect":  0.25,
    "y_x10_effect":-0.15,
}


def _p_med_10cov(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, ap):
    d = DGP_10COV
    return sigmoid(
        d["m_intercept"] + d["m_a_effect"] * ap
        + d["m_x1_effect"]*x1  + d["m_x2_effect"]*x2  + d["m_x3_effect"]*x3
        + d["m_x4_effect"]*x4  + d["m_x5_effect"]*x5  + d["m_x6_effect"]*x6
        + d["m_x7_effect"]*x7  + d["m_x8_effect"]*x8  + d["m_x9_effect"]*x9
        + d["m_x10_effect"]*x10
    )

def _bX_10cov(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    d = DGP_10COV
    return (d["y_x1_effect"]*x1  + d["y_x2_effect"]*x2  + d["y_x3_effect"]*x3
          + d["y_x4_effect"]*x4  + d["y_x5_effect"]*x5  + d["y_x6_effect"]*x6
          + d["y_x7_effect"]*x7  + d["y_x8_effect"]*x8  + d["y_x9_effect"]*x9
          + d["y_x10_effect"]*x10)

def _sample_data_10cov(n, seed=0, sigma_y=None):
    d  = DGP_10COV
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    # exogenous roots
    X1  = rng.standard_normal(n)
    X2  = rng.standard_normal(n)
    X5  = rng.binomial(1, 0.5, size=n).astype(float)

    # level 1 — depend on X1, X2
    X3  = rng.binomial(1, sigmoid(d["x3_on_x1"]*X1 + d["x3_on_x2"]*X2)).astype(float)
    X4  = d["x4_on_x1"]*X1 + d["x4_on_x2"]*X2 + rng.standard_normal(n)

    # level 2 — depend on X3, X4, X5
    X6  = d["x6_on_x3"]*X3 + d["x6_on_x5"]*X5 + rng.standard_normal(n)
    X7  = d["x7_on_x4"]*X4 + d["x7_on_x6"]*X6 + rng.standard_normal(n)

    # level 3 — depend on X2, X7, X1
    X8  = rng.binomial(1, sigmoid(d["x8_on_x2"]*X2 + d["x8_on_x7"]*X7)).astype(float)
    X9  = d["x9_on_x1"]*X1 + d["x9_on_x8"]*X8 + rng.standard_normal(n)

    # level 4 — depend on X4, X9
    X10 = rng.binomial(1, sigmoid(d["x10_on_x4"]*X4 + d["x10_on_x9"]*X9)).astype(float)

    xs = (X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

    A = rng.binomial(1, sigmoid(
        d["a_intercept"]
        + d["a_x1_effect"]*X1  + d["a_x2_effect"]*X2  + d["a_x3_effect"]*X3
        + d["a_x4_effect"]*X4  + d["a_x5_effect"]*X5  + d["a_x6_effect"]*X6
        + d["a_x7_effect"]*X7  + d["a_x8_effect"]*X8  + d["a_x9_effect"]*X9
        + d["a_x10_effect"]*X10
    ))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_10cov(*xs, 0)
    pM1i = _p_med_10cov(*xs, 1)
    M0   = (UM < pM0i).astype(int);  M1 = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(X1)
    bX = _bX_10cov(*xs)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    return dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5,
                X6=X6, X7=X7, X8=X8, X9=X9, X10=X10,
                A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_10cov(t_grid, X, a=0, ap=0):
    """Closed-form oracle CDF for 10-cov DGP. X: (n, 10) → returns (n, T).
    The inter-covariate dependencies drop out when conditioning on observed X.
    """
    d   = DGP_10COV
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    X   = np.asarray(X)
    xs  = [X[:, i:i+1] for i in range(10)]   # each (n, 1)
    t   = np.asarray(t_grid)[None, :]         # (1, T)

    p        = _p_med_10cov(*xs, ap)
    d_x, q_x = smooth_effects(xs[0])          # driven by X1
    mu0 = _bX_10cov(*xs) + d_x*a
    mu1 = mu0 + q_x + d["am_interaction"]*a

    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)

DGP_20COV = {
    **_DGP_SHARED,
    # covariate generation coefficients
    "x3_on_x1":    0.40,  "x3_on_x2":   -0.30,
    "x4_on_x1":    0.50,  "x4_on_x2":    0.30,
    "x12_on_x11":  0.45,  "x12_on_x2":   0.20,
    "x13_on_x11":  0.35,  "x13_on_x15": -0.25,
    "x16_on_x1":   0.40,  "x16_on_x11": -0.30,
    "x6_on_x3":    0.60,  "x6_on_x5":   -0.40,
    "x7_on_x4":    0.30,  "x7_on_x6":    0.20,
    "x14_on_x12":  0.50,  "x14_on_x13": -0.30,
    "x17_on_x16":  0.40,  "x17_on_x5":   0.35,
    "x8_on_x2":    0.50,  "x8_on_x7":    0.40,
    "x9_on_x1":    0.40,  "x9_on_x8":   -0.30,
    "x18_on_x14":  0.45,  "x18_on_x17": -0.25,
    "x10_on_x4":   0.30,  "x10_on_x9":  -0.50,
    "x19_on_x16":  0.35,  "x19_on_x18":  0.40,
    "x20_on_x10":  0.25,  "x20_on_x19": -0.30, "x20_on_x7": 0.20,
    # model coefficient arrays — index i → covariate X_{i+1}, i=0..19
    "m_effects": np.array([
         0.15,  0.30, -0.25,  0.20, -0.10,   # X1–X5
         0.15, -0.20,  0.25, -0.15,  0.10,   # X6–X10
         0.20, -0.15,  0.30, -0.20,  0.10,   # X11–X15
         0.25, -0.10,  0.15, -0.20,  0.10,   # X16–X20
    ]),
    "a_intercept": 0.20,
    "a_effects": np.array([
         0.35, -0.20,  0.40, -0.15,  0.25,
        -0.10,  0.30, -0.20,  0.15, -0.25,
         0.30, -0.20,  0.10,  0.25, -0.15,
         0.20, -0.10,  0.30, -0.25,  0.15,
    ]),
    "y_effects": np.array([
         0.10,  0.25, -0.15,  0.20, -0.10,
         0.15, -0.20,  0.10,  0.25, -0.15,
         0.20, -0.10,  0.15, -0.20,  0.10,
         0.25, -0.15,  0.20, -0.10,  0.15,
    ]),
}

def _p_med_20cov(X, ap):
    """X: (n, 20) → (n,)."""
    d = DGP_20COV
    return sigmoid(d["m_intercept"] + d["m_a_effect"]*ap + X @ d["m_effects"])

def _bX_20cov(X):
    """X: (n, 20) → (n,)."""
    return X @ DGP_20COV["y_effects"]

def _gen_covariates_20cov(n, rng):
    """Generate all 20 covariates in topological order. Returns (n, 20)."""
    d = DGP_20COV
    # roots
    X1  = rng.standard_normal(n)
    X2  = rng.standard_normal(n)
    X5  = rng.binomial(1, 0.5, size=n).astype(float)
    X11 = rng.standard_normal(n)
    X15 = rng.binomial(1, 0.5, size=n).astype(float)
    # level 1
    X3  = rng.binomial(1, sigmoid(d["x3_on_x1"]*X1   + d["x3_on_x2"]*X2  )).astype(float)
    X4  = d["x4_on_x1"]*X1    + d["x4_on_x2"]*X2    + rng.standard_normal(n)
    X12 = d["x12_on_x11"]*X11 + d["x12_on_x2"]*X2   + rng.standard_normal(n)
    X13 = rng.binomial(1, sigmoid(d["x13_on_x11"]*X11 + d["x13_on_x15"]*X15)).astype(float)
    X16 = d["x16_on_x1"]*X1   + d["x16_on_x11"]*X11 + rng.standard_normal(n)
    # level 2
    X6  = d["x6_on_x3"]*X3    + d["x6_on_x5"]*X5    + rng.standard_normal(n)
    X7  = d["x7_on_x4"]*X4    + d["x7_on_x6"]*X6    + rng.standard_normal(n)
    X14 = d["x14_on_x12"]*X12 + d["x14_on_x13"]*X13 + rng.standard_normal(n)
    X17 = rng.binomial(1, sigmoid(d["x17_on_x16"]*X16 + d["x17_on_x5"]*X5)).astype(float)
    # level 3
    X8  = rng.binomial(1, sigmoid(d["x8_on_x2"]*X2   + d["x8_on_x7"]*X7  )).astype(float)
    X9  = d["x9_on_x1"]*X1    + d["x9_on_x8"]*X8    + rng.standard_normal(n)
    X18 = d["x18_on_x14"]*X14 + d["x18_on_x17"]*X17 + rng.standard_normal(n)
    # level 4
    X10 = rng.binomial(1, sigmoid(d["x10_on_x4"]*X4  + d["x10_on_x9"]*X9 )).astype(float)
    X19 = d["x19_on_x16"]*X16 + d["x19_on_x18"]*X18 + rng.standard_normal(n)
    # level 5
    X20 = rng.binomial(1, sigmoid(
        d["x20_on_x10"]*X10 + d["x20_on_x19"]*X19 + d["x20_on_x7"]*X7
    )).astype(float)

    return np.column_stack([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,
                            X11,X12,X13,X14,X15,X16,X17,X18,X19,X20])

def _sample_data_20cov(n, seed=0, sigma_y=None):
    d  = DGP_20COV
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    Xmat = _gen_covariates_20cov(n, rng)   # (n, 20)

    A = rng.binomial(1, sigmoid(d["a_intercept"] + Xmat @ d["a_effects"]))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_20cov(Xmat, 0);  pM1i = _p_med_20cov(Xmat, 1)
    M0   = (UM < pM0i).astype(int);  M1 = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(Xmat[:, 0])   # X1 is col 0
    bX = _bX_20cov(Xmat)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    # zero-padded keys so sorted() gives correct column order (X01 < X02 < ... < X20)
    Xcols = {f"X{i+1:02d}": Xmat[:, i] for i in range(20)}
    return dict(**Xcols, A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_20cov(t_grid, X, a=0, ap=0):
    """X: (n, 20) → (n, T). Inter-covariate dependencies drop out conditioning on X."""
    d   = DGP_20COV
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    X   = np.asarray(X)                          # (n, 20)
    t   = np.asarray(t_grid)[None, :]             # (1, T)
    p        = _p_med_20cov(X, ap)[:, None]       # (n, 1)
    d_x, q_x = smooth_effects(X[:, 0:1])          # (n, 1) — col 0 is X1
    mu0 = _bX_20cov(X)[:, None] + d_x*a           # (n, 1)
    mu1 = mu0 + q_x + d["am_interaction"]*a        # (n, 1)
    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)


# ===========================================================================
# Polynomial DGP  (3 covariates, X1, X2 continuous; X3 binary)
# Nonlinearity via X1², X2², and X1·X2 interaction terms in A, M, and Y.
# ===========================================================================
DGP_POLY = {
    **_DGP_SHARED,
    # covariate generation
    "x3_on_x1":  0.40, "x3_on_x2": -0.30,
    # mediator: linear + quadratic + interaction
    "m_x1_effect":    0.15, "m_x2_effect":    0.40, "m_x3_effect":   -0.30,
    "m_x1sq_effect": -0.10, "m_x1x2_effect":  0.20,
    # treatment: linear + quadratic + interaction
    "a_intercept":    0.20,
    "a_x1_effect":    0.35, "a_x2_effect":   -0.25, "a_x3_effect":    0.50,
    "a_x1sq_effect":  0.15, "a_x2sq_effect": -0.10, "a_x1x2_effect":  0.20,
    # outcome: linear + quadratic + interaction
    "y_x1_effect":    0.10, "y_x2_effect":    0.30, "y_x3_effect":   -0.20,
    "y_x1sq_effect":  0.15, "y_x1x2_effect": -0.10,
}

def _p_med_poly(x1, x2, x3, ap):
    d = DGP_POLY
    return sigmoid(
        d["m_intercept"] + d["m_a_effect"]*ap
        + d["m_x1_effect"]*x1 + d["m_x2_effect"]*x2 + d["m_x3_effect"]*x3
        + d["m_x1sq_effect"]*x1**2 + d["m_x1x2_effect"]*x1*x2
    )

def _bX_poly(x1, x2, x3):
    d = DGP_POLY
    return (d["y_x1_effect"]*x1 + d["y_x2_effect"]*x2 + d["y_x3_effect"]*x3
            + d["y_x1sq_effect"]*x1**2 + d["y_x1x2_effect"]*x1*x2)

def _sample_data_poly(n, seed=0, sigma_y=None):
    d  = DGP_POLY
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    X3 = rng.binomial(1, sigmoid(d["x3_on_x1"]*X1 + d["x3_on_x2"]*X2)).astype(float)

    A = rng.binomial(1, sigmoid(
        d["a_intercept"]
        + d["a_x1_effect"]*X1 + d["a_x2_effect"]*X2 + d["a_x3_effect"]*X3
        + d["a_x1sq_effect"]*X1**2 + d["a_x2sq_effect"]*X2**2 + d["a_x1x2_effect"]*X1*X2
    ))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_poly(X1, X2, X3, 0);  pM1i = _p_med_poly(X1, X2, X3, 1)
    M0   = (UM < pM0i).astype(int);     M1   = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(X1);  bX = _bX_poly(X1, X2, X3)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    return dict(X1=X1, X2=X2, X3=X3,
                A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_poly(t_grid, X, a=0, ap=0):
    """X: (n, 3) → (n, T). Oracle CDF for polynomial DGP."""
    d   = DGP_POLY
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    X   = np.asarray(X)
    x1, x2, x3 = [X[:, i:i+1] for i in range(3)]
    t   = np.asarray(t_grid)[None, :]
    p        = _p_med_poly(x1, x2, x3, ap)
    d_x, q_x = smooth_effects(x1)
    mu0 = _bX_poly(x1, x2, x3) + d_x*a
    mu1 = mu0 + q_x + d["am_interaction"]*a
    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)


# ===========================================================================
# Sinusoidal DGP  (3 covariates, X1, X2 continuous; X3 binary)
# Nonlinearity via sin(X1) and cos(X2) in A, M, and Y.
# ===========================================================================
DGP_SIN = {
    **_DGP_SHARED,
    # covariate generation
    "x3_on_x1":  0.40, "x3_on_x2": -0.30,
    # mediator: sinusoidal
    "m_sinx1_effect":  0.35, "m_cosx2_effect":  0.25, "m_x3_effect": -0.30,
    # treatment: sinusoidal
    "a_intercept":     0.20,
    "a_sinx1_effect":  0.55, "a_cosx2_effect": -0.40, "a_x3_effect":  0.50,
    # outcome: sinusoidal
    "y_sinx1_effect":  0.25, "y_cosx2_effect":  0.35, "y_x3_effect": -0.20,
}

def _p_med_sin(x1, x2, x3, ap):
    d = DGP_SIN
    return sigmoid(
        d["m_intercept"] + d["m_a_effect"]*ap
        + d["m_sinx1_effect"]*np.sin(x1) + d["m_cosx2_effect"]*np.cos(x2) + d["m_x3_effect"]*x3
    )

def _bX_sin(x1, x2, x3):
    d = DGP_SIN
    return (d["y_sinx1_effect"]*np.sin(x1) + d["y_cosx2_effect"]*np.cos(x2)
            + d["y_x3_effect"]*x3)

def _sample_data_sin(n, seed=0, sigma_y=None):
    d  = DGP_SIN
    sy = d["sigma_y"] if sigma_y is None else sigma_y
    rng = np.random.default_rng(seed)

    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    X3 = rng.binomial(1, sigmoid(d["x3_on_x1"]*X1 + d["x3_on_x2"]*X2)).astype(float)

    A = rng.binomial(1, sigmoid(
        d["a_intercept"]
        + d["a_sinx1_effect"]*np.sin(X1) + d["a_cosx2_effect"]*np.cos(X2) + d["a_x3_effect"]*X3
    ))

    UM   = rng.uniform(size=n)
    pM0i = _p_med_sin(X1, X2, X3, 0);  pM1i = _p_med_sin(X1, X2, X3, 1)
    M0   = (UM < pM0i).astype(int);    M1   = (UM < pM1i).astype(int)
    M    = np.where(A == 1, M1, M0)

    Ubase = rng.standard_normal(n);  Umed = rng.standard_normal(n)
    d_x, q_x = smooth_effects(X1);  bX = _bX_sin(X1, X2, X3)

    def Y_pot(a, m):
        return (bX + d_x*a + q_x*m + d["am_interaction"]*a*m
                + sy*Ubase + d["sigma_m"]*m*Umed)

    Y00 = Y_pot(0, M0);  Y10 = Y_pot(1, M0);  Y11 = Y_pot(1, M1)
    Y   = np.where(A==1, np.where(M==1, Y_pot(1,1), Y_pot(1,0)),
                         np.where(M==1, Y_pot(0,1), Y_pot(0,0)))

    return dict(X1=X1, X2=X2, X3=X3,
                A=A, M=M, Y=Y, Y00=Y00, Y10=Y10, Y11=Y11,
                bX=bX, d=d_x, q=q_x, c=np.full(n, d["am_interaction"]),
                pM0=pM0i, pM1=pM1i, sigma_y=sy, sigma_m_noise=d["sigma_m"])

def _oracle_F_sin(t_grid, X, a=0, ap=0):
    """X: (n, 3) → (n, T). Oracle CDF for sinusoidal DGP."""
    d   = DGP_SIN
    sd0 = d["sigma_y"];  sd1 = np.sqrt(d["sigma_y"]**2 + d["sigma_m"]**2)
    X   = np.asarray(X)
    x1, x2, x3 = [X[:, i:i+1] for i in range(3)]
    t   = np.asarray(t_grid)[None, :]
    p        = _p_med_sin(x1, x2, x3, ap)
    d_x, q_x = smooth_effects(x1)
    mu0 = _bX_sin(x1, x2, x3) + d_x*a
    mu1 = mu0 + q_x + d["am_interaction"]*a
    return (1-p)*norm.cdf((t-mu0)/sd0) + p*norm.cdf((t-mu1)/sd1)


# ── dispatch wrappers — the rest of the script calls only these ───────────────
def p_mediator(x, ap):
    X = np.asarray(x)
    if ACTIVE_DGP == "poly":  return _p_med_poly(X[:,0], X[:,1], X[:,2], ap)
    if ACTIVE_DGP == "sin":   return _p_med_sin(X[:,0], X[:,1], X[:,2], ap)
    if ACTIVE_DGP == "3cov":  return _p_med_3cov(X[:,0], X[:,1], X[:,2], ap)
    if ACTIVE_DGP == "5cov":  return _p_med_5cov(X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], ap)
    if ACTIVE_DGP == "10cov": return _p_med_10cov(*[X[:,i] for i in range(10)], ap)
    if ACTIVE_DGP == "20cov": return _p_med_20cov(X, ap)
    return _p_med_1cov(x, ap)

def sample_data(n, seed=0, sigma_y=None):
    if ACTIVE_DGP == "poly":  return _sample_data_poly(n, seed, sigma_y)
    if ACTIVE_DGP == "sin":   return _sample_data_sin(n, seed, sigma_y)
    if ACTIVE_DGP == "3cov":  return _sample_data_3cov(n, seed, sigma_y)
    if ACTIVE_DGP == "5cov":  return _sample_data_5cov(n, seed, sigma_y)
    if ACTIVE_DGP == "10cov": return _sample_data_10cov(n, seed, sigma_y)
    if ACTIVE_DGP == "20cov": return _sample_data_20cov(n, seed, sigma_y)
    return _sample_data_1cov(n, seed, sigma_y)

def oracle_F_closed_vec(t_grid, X, a=0, ap=0):
    """Dispatches on ACTIVE_DGP / X shape. Returns (n, T)."""
    X = np.asarray(X)
    if ACTIVE_DGP == "poly":  return _oracle_F_poly(t_grid, X, a, ap)
    if ACTIVE_DGP == "sin":   return _oracle_F_sin(t_grid, X, a, ap)
    if X.ndim == 1:           return _oracle_F_1cov(t_grid, X, a, ap)
    p = X.shape[1]
    if p == 3:  return _oracle_F_3cov(t_grid, X, a, ap)
    if p == 5:  return _oracle_F_5cov(t_grid, X, a, ap)
    if p == 10: return _oracle_F_10cov(t_grid, X, a, ap)
    if p == 20: return _oracle_F_20cov(t_grid, X, a, ap)
    raise ValueError(f"No oracle registered for {p} covariates")


# ===========================================================================
# 2. Estimators / aggregation  (notebook cells 9 & 10)
# ===========================================================================
EFFECTS = ["Direct", "Indirect", "Total"]
ESTIMATORS = ["dr", "plugin"]
# Set in main() via --dgp-type / --cov; default here is only for module-level use.
ACTIVE_DGP = "3cov"  # "1cov"|"3cov"|"5cov"|"10cov"|"20cov"|"poly"|"sin"
_DGP_MAP = {
    "1cov":  DGP_1COV,  "3cov":  DGP_3COV,  "5cov":  DGP_5COV,
    "10cov": DGP_10COV, "20cov": DGP_20COV,
    "poly":  DGP_POLY,  "sin":   DGP_SIN,
}
DGP = _DGP_MAP[ACTIVE_DGP]

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _wald_ci(theta, se, alpha=0.05):
    z = _norm.ppf(1 - alpha / 2)
    return (_clip01(theta - z * se), _clip01(theta + z * se))


def _dr_bounds_and_cis(delta, gamma_delta, alpha=0.05):
    """DR Makarov bounds with Wald CIs.

    delta       : (T, n) estimated conditional CDF difference F_u(t|X) - F_v(t|X)
    gamma_delta : (T, n) difference of EIF pseudo-outcomes Gamma_u(t) - Gamma_v(t)
    """
    n = delta.shape[1]
    obs = np.arange(n)

    s_L = delta.max(axis=0)
    idx_L = delta.argmax(axis=0)
    s_U = (-delta).max(axis=0)
    idx_U = (-delta).argmax(axis=0)

    G_L = gamma_delta[idx_L, obs]
    G_U = gamma_delta[idx_U, obs]

    theta_L_raw = float(np.mean((s_L > 0) * G_L))
    theta_U_raw = float(np.mean(1.0 + (s_U > 0) * G_U))

    phi_L = (s_L > 0) * G_L - theta_L_raw
    phi_U = 1.0 + (s_U > 0) * G_U - theta_U_raw

    se_L = float(np.sqrt(np.var(phi_L, ddof=1) / n))
    se_U = float(np.sqrt(np.var(phi_U, ddof=1) / n))

    ci_L = _wald_ci(theta_L_raw, se_L, alpha=alpha)
    ci_U = _wald_ci(theta_U_raw, se_U, alpha=alpha)

    return {
        "L": _clip01(theta_L_raw),
        "U": _clip01(theta_U_raw),
        "L_raw": theta_L_raw,
        "U_raw": theta_U_raw,
        "se_L": se_L,
        "se_U": se_U,
        "ci_L": ci_L,
        "ci_U": ci_U,
    }


def oracle_margin_summary(data, grid_size=800):
    """Oracle-only diagnostic for the kink and separation parts of the margin condition."""
    dat = data
    Tn = np.linspace(
        min(data["Y00"].min(), data["Y10"].min(), data["Y11"].min()),
        max(data["Y00"].max(), data["Y10"].max(), data["Y11"].max()),
        grid_size,
    )
    cond = oracle_cond_cdfs(dat, Tn)  # noqa: F405
    deltas = {
        "Direct": cond[(1, 0)] - cond[(0, 0)],
        "Indirect": cond[(1, 1)] - cond[(1, 0)],
        "Total": cond[(1, 1)] - cond[(0, 0)],
    }

    out = {}
    for eff, delta in deltas.items():
        s_L = delta.max(axis=0)
        s_U = (-delta).max(axis=0)
        sorted_L = np.sort(delta, axis=0)
        sorted_U = np.sort(-delta, axis=0)
        gap_L = sorted_L[-1] - sorted_L[-2]
        gap_U = sorted_U[-1] - sorted_U[-2]
        out[eff] = {
            "frac_L_at_kink": float(np.mean(s_L <= 1e-4)),
            "frac_U_at_kink": float(np.mean(s_U <= 1e-4)),
            "median_s_L": float(np.median(s_L)),
            "median_s_U": float(np.median(s_U)),
            "median_argmax_gap_L": float(np.median(gap_L)),
            "median_argmax_gap_U": float(np.median(gap_U)),
        }
    return out


def run_single_simulation(seed, n_sample=20000, n_mc=800, epochs1=40, epochs2=40,
                          hidden=64, n_layers=2, lr=1e-3, batch_size=256,
                          weight_decay=1e-4, alpha=0.05, sigma_y=.5,
                          save_plots=False, plot_dir=None):
    """Run one project-DGP simulation and return bound estimates, Wald CIs, and coverage."""
    data = sample_data(n=n_sample, seed=seed)
    X = np.column_stack([data[k] for k in sorted(data)
                         if k.startswith("X") and k[1:].isdigit()])

    A, M, Y = data["A"], data["M"], data["Y"]
    Y00, Y10, Y11 = data["Y00"], data["Y10"], data["Y11"]
    dat = data

    Y_clean = np.asarray(Y)
    Y_clean = Y_clean[np.isfinite(Y_clean)]

    q_grid = np.concatenate([
        np.linspace(0.001, 0.01, 10),    # sparse lower tail
        np.linspace(0.01, 0.99, 800),    # dense central region
        np.linspace(0.99, 0.999, 10),    # sparse upper tail
    ])
    q_grid = np.unique(q_grid)

    Tn = np.quantile(Y_clean, q_grid)
    Tn = np.unique(Tn)

    oracle_ccdfs = {
        (0, 0): oracle_F_closed_vec(Tn, X.squeeze(), 0, 0).T,
        (1, 0): oracle_F_closed_vec(Tn, X.squeeze(), 1, 0).T,
        (1, 1): oracle_F_closed_vec(Tn, X.squeeze(), 1, 1).T,
    }
    oracle_cdfs = oracle_marginal_cdfs(data, Tn)  # noqa: F405
    ora_bds_marginal = bounds_from_cdf_triplet(  # noqa: F405
        oracle_cdfs[(0, 0)], oracle_cdfs[(1, 0)], oracle_cdfs[(1, 1)])
    ora_bds = oracle_covariate_assisted_bounds(oracle_ccdfs, Tn)  # noqa: F405
    true_fna = oracle_true_fna(dat)  # noqa: F405

    phi, splits = crossfit_pseudo_outcomes(  # noqa: F405
        X, A, M, Y, Tn, K=5, n_mc=n_mc, epochs=epochs1, t_learner=False,
        hidden=hidden, n_layers=n_layers, lr=lr,
        batch_size=batch_size, weight_decay=weight_decay)
    dr_ccdfs, curves = crossfit_conditional_cdfs_crps(  # noqa: F405
        phi, X, Tn, splits, epochs=epochs2,
        hidden=hidden, n_layers=n_layers, lr=lr,
        batch_size=batch_size, weight_decay=weight_decay)

    # --- Optional diagnostic plot of the CRPS pseudo-loss curves. In the
    #     notebook this rendered inline; here it is off by default and saved
    #     to a file when enabled, so it never leaks memory over many seeds. ---
    if save_plots and plot_dir is not None:
        fig = plt.figure()
        for key, lbl in [((0, 0), "F00"), ((1, 0), "F10"), ((1, 1), "F11")]:
            c = curves[key]
            plt.plot(c["epoch"], c["train_mean"], label=f"{lbl} train")
            plt.plot(c["epoch"], c["test_mean"], label=f"{lbl} held-out")
            plt.fill_between(c["epoch"], c["test_mean"] - c["test_std"],
                             c["test_mean"] + c["test_std"], alpha=0.2)
        plt.xlabel("epoch")
        plt.ylabel("CRPS pseudo-loss")
        plt.legend()
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, f"crps_curves_seed{seed}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    phi00, phi10, phi11 = phi[(0, 0)], phi[(1, 0)], phi[(1, 1)]
    deltas = {
        "Direct": (dr_ccdfs[(1, 0)] - dr_ccdfs[(0, 0)], phi10 - phi00),
        "Indirect": (dr_ccdfs[(1, 1)] - dr_ccdfs[(1, 0)], phi11 - phi10),
        "Total": (dr_ccdfs[(1, 1)] - dr_ccdfs[(0, 0)], phi11 - phi00),
    }

    plugin_ccdfs = fit_plugin_cdfs(  # noqa: F405
        X, A, M, Y, Tn, n_mc=n_mc, epochs=epochs2,
        hidden=hidden, n_layers=n_layers, lr=lr,
        batch_size=batch_size, weight_decay=weight_decay)
    pl_bds = tighter_bounds_from_ccdf_triplet_plugin(  # noqa: F405
        plugin_ccdfs[(0, 0)], plugin_ccdfs[(1, 0)], plugin_ccdfs[(1, 1)])

    summary = {}

    print(f"\nSeed {seed}  n={n_sample}")
    hdr = (f"  {'Effect':<10} {'Est':<7} {'B':<2} "
           f"{'Estimate':>9} {'SE':>7} "
           f"{'CI_lo':>8} {'CI_hi':>8} "
           f"{'Oracle':>10} {'Cv?':>5} {'LCv?':>6} "
           f"{'TrueFNA':>8} {'CvFNA?':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for eff in EFFECTS:
        o_L, o_U = ora_bds[eff][0], ora_bds[eff][1]
        o_L_m, o_U_m = ora_bds_marginal[eff][0], ora_bds_marginal[eff][1]
        tv = true_fna[eff]

        delta_eff, gdelta_eff = deltas[eff]
        dr = _dr_bounds_and_cis(delta_eff, gdelta_eff, alpha=alpha)
        dr["wald_cs"] = (dr["ci_L"][0], dr["ci_U"][1])
        dr["covers_oracle_L"] = dr["ci_L"][0] <= o_L <= dr["ci_L"][1]
        dr["covers_oracle_U"] = dr["ci_U"][0] <= o_U <= dr["ci_U"][1]
        dr["lower_ci_covers_oracle_L"] = dr["ci_L"][0] <= o_L
        dr["upper_ci_covers_oracle_U"] = dr["ci_U"][1] >= o_U
        dr["covers_oracle_cs"] = dr["wald_cs"][0] <= o_L and o_U <= dr["wald_cs"][1]
        dr["covers_true_fna"] = dr["wald_cs"][0] <= tv <= dr["wald_cs"][1]

        e_L, e_U, se_L, se_U = pl_bds[eff]
        pl_ci_L = _wald_ci(e_L, se_L, alpha=alpha)
        pl_ci_U = _wald_ci(e_U, se_U, alpha=alpha)
        pl_cs = (pl_ci_L[0], pl_ci_U[1])
        plugin = {
            "L": _clip01(e_L),
            "U": _clip01(e_U),
            "se_L": float(se_L),
            "se_U": float(se_U),
            "ci_L": pl_ci_L,
            "ci_U": pl_ci_U,
            "wald_cs": pl_cs,
            "covers_oracle_L": pl_ci_L[0] <= o_L <= pl_ci_L[1],
            "covers_oracle_U": pl_ci_U[0] <= o_U <= pl_ci_U[1],
            "lower_ci_covers_oracle_L": pl_ci_L[0] <= o_L,
            "upper_ci_covers_oracle_U": pl_ci_U[1] >= o_U,
            "covers_oracle_cs": pl_cs[0] <= o_L and o_U <= pl_cs[1],
            "covers_true_fna": pl_cs[0] <= tv <= pl_cs[1],
        }

        summary[eff] = {
            "oracle_x": {"L": o_L, "U": o_U},
            "oracle_marginal": {"L": o_L_m, "U": o_U_m},
            "true_fna": tv,
            "dr": dr,
            "plugin": plugin,
        }

        for est_lbl, bds in [("DR", dr), ("Plugin", plugin)]:
            for bound, e_val, se_v, ci, o_val, covered, lower_covered in [
                ("L", bds["L"], bds["se_L"], bds["ci_L"], o_L,
                 bds["covers_oracle_L"], bds["lower_ci_covers_oracle_L"]),
                ("U", bds["U"], bds["se_U"], bds["ci_U"], o_U,
                 bds["covers_oracle_U"], bds["upper_ci_covers_oracle_U"]),
            ]:
                fna_str = (f"  {tv:>8.4f} {str(bds['covers_true_fna']):>8}"
                           if bound == "L" else "")
                print(f"  {eff:<10} {est_lbl:<7} {bound:<2} "
                      f"{e_val:>9.4f} {se_v:>7.4f} "
                      f"{ci[0]:>8.4f} {ci[1]:>8.4f} "
                      f"OrX={o_val:.4f} {str(covered):>5} {str(lower_covered):>6}"
                      + fna_str)
        print()

    return {"seed": seed, "summary": summary}


def summarize_simulations(results, base_seed=None):
    """Aggregate one list of independent simulation results."""
    if len(results) == 0:
        raise ValueError("No simulation results to summarize.")

    n_sims = len(results)
    agg = {"base_seed": base_seed, "n_sims": n_sims}

    for key in ("cover_oracle_L", "cover_oracle_U", "lower_ci_cover_oracle_L",
                "upper_ci_cover_oracle_U", "cover_oracle_cs", "cover_true_fna",
                "mean_L", "mean_U", "mean_se_L", "mean_se_U",
                "mean_ci_width_L", "mean_ci_width_U",
                "lb_bias", "ub_bias", "mean_interval_width"):
        agg[key] = {e: {} for e in ESTIMATORS}

    for key in ("oracle_L_x", "oracle_U_x", "oracle_L_m", "oracle_U_m", "oracle_width_x"):
        agg[key] = {}

    for eff in EFFECTS:
        agg["oracle_L_x"][eff] = float(np.mean([r["summary"][eff]["oracle_x"]["L"] for r in results]))
        agg["oracle_U_x"][eff] = float(np.mean([r["summary"][eff]["oracle_x"]["U"] for r in results]))
        agg["oracle_L_m"][eff] = float(np.mean([r["summary"][eff]["oracle_marginal"]["L"] for r in results]))
        agg["oracle_U_m"][eff] = float(np.mean([r["summary"][eff]["oracle_marginal"]["U"] for r in results]))
        agg["oracle_width_x"][eff] = agg["oracle_U_x"][eff] - agg["oracle_L_x"][eff]

        for est in ESTIMATORS:
            s = [r["summary"][eff][est] for r in results]
            ora_Ls_x = [r["summary"][eff]["oracle_x"]["L"] for r in results]
            ora_Us_x = [r["summary"][eff]["oracle_x"]["U"] for r in results]

            agg["cover_oracle_L"][est][eff] = float(np.mean([x["covers_oracle_L"] for x in s]))
            agg["cover_oracle_U"][est][eff] = float(np.mean([x["covers_oracle_U"] for x in s]))
            agg["lower_ci_cover_oracle_L"][est][eff] = float(np.mean([
                x.get("lower_ci_covers_oracle_L", x["ci_L"][0] <= ora_Ls_x[i])
                for i, x in enumerate(s)
            ]))
            agg["upper_ci_cover_oracle_U"][est][eff] = float(np.mean([
                x.get("upper_ci_covers_oracle_U", x["ci_U"][1] >= ora_Us_x[i])
                for i, x in enumerate(s)
            ]))
            agg["cover_oracle_cs"][est][eff] = float(np.mean([x["covers_oracle_cs"] for x in s]))
            agg["cover_true_fna"][est][eff] = float(np.mean([x["covers_true_fna"] for x in s]))
            agg["mean_L"][est][eff] = float(np.mean([x["L"] for x in s]))
            agg["mean_U"][est][eff] = float(np.mean([x["U"] for x in s]))
            agg["mean_se_L"][est][eff] = float(np.mean([x["se_L"] for x in s]))
            agg["mean_se_U"][est][eff] = float(np.mean([x["se_U"] for x in s]))
            agg["mean_ci_width_L"][est][eff] = float(np.mean([x["ci_L"][1] - x["ci_L"][0] for x in s]))
            agg["mean_ci_width_U"][est][eff] = float(np.mean([x["ci_U"][1] - x["ci_U"][0] for x in s]))
            agg["lb_bias"][est][eff] = float(np.mean([s[i]["L"] - ora_Ls_x[i] for i in range(n_sims)]))
            agg["ub_bias"][est][eff] = float(np.mean([s[i]["U"] - ora_Us_x[i] for i in range(n_sims)]))
            agg["mean_interval_width"][est][eff] = float(np.mean([x["U"] - x["L"] for x in s]))

    return agg


def run_simulation_study(base_seed, n_sims=10, n_sample=20000, n_mc=500, sigma_y=.5,
                         epochs1=40, epochs2=40, hidden=64, n_layers=2,
                         lr=1e-3, batch_size=256, weight_decay=1e-4):
    """Run n_sims independent seeds and aggregate Wald CI / coverage statistics."""
    results = []
    for i in tqdm(range(n_sims), desc=f"seed={base_seed}"):
        results.append(run_single_simulation(
            seed=base_seed + i,
            n_sample=n_sample, n_mc=n_mc, epochs1=epochs1, epochs2=epochs2,
            hidden=hidden, n_layers=n_layers, lr=lr,
            batch_size=batch_size, weight_decay=weight_decay))
    return summarize_simulations(results, base_seed=base_seed)


def print_aggregated(agg):
    """Print a Wald-inference summary table."""
    W = 156
    print(f"\n{'=' * W}")
    print(f"  base_seed={agg['base_seed']}  n_sims={agg['n_sims']}")
    print(f"{'=' * W}")

    hdr = (f"  {'Effect':<10} {'Est':<7} "
           f"{'CvOrL':>6} {'CvOrU':>6} {'LCvOrL':>7} {'UCvOrU':>7} {'CvOrCS':>7} {'CvFNA':>6} "
           f"{'mean_L':>8} {'mean_U':>8} "
           f"{'SE_L':>7} {'SE_U':>7} "
           f"{'wCI_L':>7} {'wCI_U':>7} "
           f"{'BiasL':>9} {'BiasU':>9} {'IW(U-L)':>9} "
           f"  [Oracle: {'OrX_L':>7} {'OrX_U':>7} {'OrX_W':>7} {'OrM_L':>7} {'OrM_U':>7}]")
    print(hdr)
    print("  " + "-" * (W - 2))

    for eff in EFFECTS:
        for est in ESTIMATORS:
            oracle_ref = ""
            if est == ESTIMATORS[0]:
                oracle_ref = (f"  [Oracle: {agg['oracle_L_x'][eff]:>7.4f} {agg['oracle_U_x'][eff]:>7.4f}"
                              f" {agg['oracle_width_x'][eff]:>7.4f}"
                              f" {agg['oracle_L_m'][eff]:>7.4f} {agg['oracle_U_m'][eff]:>7.4f}]")
            print(f"  {eff:<10} {est:<7}"
                  f" {agg['cover_oracle_L'][est][eff]:>6.3f}"
                  f" {agg['cover_oracle_U'][est][eff]:>6.3f}"
                  f" {agg['lower_ci_cover_oracle_L'][est][eff]:>7.3f}"
                  f" {agg['upper_ci_cover_oracle_U'][est][eff]:>7.3f}"
                  f" {agg['cover_oracle_cs'][est][eff]:>7.3f}"
                  f" {agg['cover_true_fna'][est][eff]:>6.3f}"
                  f" {agg['mean_L'][est][eff]:>8.4f}"
                  f" {agg['mean_U'][est][eff]:>8.4f}"
                  f" {agg['mean_se_L'][est][eff]:>7.4f}"
                  f" {agg['mean_se_U'][est][eff]:>7.4f}"
                  f" {agg['mean_ci_width_L'][est][eff]:>7.4f}"
                  f" {agg['mean_ci_width_U'][est][eff]:>7.4f}"
                  f" {agg['lb_bias'][est][eff]:>+9.4f}"
                  f" {agg['ub_bias'][est][eff]:>+9.4f}"
                  f" {agg['mean_interval_width'][est][eff]:>9.4f}"
                  + oracle_ref)
        print()

    print("=" * W)
    print("  CvOrL/CvOrU   : fraction of two-sided Wald endpoint CIs covering the oracle covariate-assisted bound")
    print("  LCvOrL/UCvOrU : one-sided checks ci_L[0] <= oracle_L and ci_U[0] >= oracle_U")
    print("  CvOrCS        : fraction of conservative Wald sets covering [OrX_L, OrX_U]")
    print("  CvFNA       : fraction of conservative Wald sets covering the true FNA")
    print("  Bias        : mean(est_bound - oracle_x_bound), paired per simulation")
    print("  IW(U-L)     : mean estimated interval width (mean_U - mean_L)")
    print("  OrM_L/OrM_U : marginal Makarov bounds shown for reference")


# ===========================================================================
# 3. Oracle NDE / NIE / ATE / FNA diagnostics  (notebook cells 15 & 17)
#    -- self-contained, CPU-only, run once with --diagnostics. Helper names are
#       underscore-prefixed so they never clash with `from functions import *`.
# ===========================================================================
def _smooth_effects_x(x, dgp):
    w = expit(dgp["k"] * x)
    d_x = dgp["d_left"] + (dgp["d_right"] - dgp["d_left"]) * w
    q_x = dgp["q_left"] + (dgp["q_right"] - dgp["q_left"]) * w
    return d_x, q_x


def _p_mediator_x(x, ap, dgp):
    return expit(dgp["m_intercept"] + dgp["m_a_effect"] * ap + dgp["m_x_effect"] * x)


def _normal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def oracle_nde_nie_ate_quad(dgp, lower=-10, upper=10):
    def nde_integrand(x):
        d_x, _ = _smooth_effects_x(x, dgp)
        p0 = _p_mediator_x(x, 0, dgp)
        return (d_x + dgp["am_interaction"] * p0) * _normal_pdf(x)

    def nie_integrand(x):
        _, q_x = _smooth_effects_x(x, dgp)
        p0 = _p_mediator_x(x, 0, dgp)
        p1 = _p_mediator_x(x, 1, dgp)
        return (p1 - p0) * (q_x + dgp["am_interaction"]) * _normal_pdf(x)

    nde, nde_err = quad(nde_integrand, lower, upper, epsabs=1e-10)
    nie, nie_err = quad(nie_integrand, lower, upper, epsabs=1e-10)
    return {"NDE": float(nde), "NIE": float(nie), "ATE": float(nde + nie),
            "NDE_err": float(nde_err), "NIE_err": float(nie_err)}


def oracle_fna_quad(dgp, lower=-10, upper=10):
    def fna_conditional_x(x):
        d_x, q_x = _smooth_effects_x(x, dgp)
        p0 = _p_mediator_x(x, 0, dgp)
        p1 = _p_mediator_x(x, 1, dgp)
        c = dgp["am_interaction"]
        sigma_m = dgp["sigma_m"]
        fna_dir_x = (1.0 - p0) * (d_x < 0) + p0 * (d_x + c < 0)
        fna_ind_x = (p1 - p0) * norm.cdf(-(q_x + c) / sigma_m)
        fna_tot_x = (
            (1.0 - p1) * (d_x < 0)
            + (p1 - p0) * norm.cdf(-(d_x + q_x + c) / sigma_m)
            + p0 * (d_x + c < 0)
        )
        return fna_dir_x, fna_ind_x, fna_tot_x

    dir_val, dir_err = quad(lambda x: fna_conditional_x(x)[0] * _normal_pdf(x), lower, upper, epsabs=1e-10)
    ind_val, ind_err = quad(lambda x: fna_conditional_x(x)[1] * _normal_pdf(x), lower, upper, epsabs=1e-10)
    tot_val, tot_err = quad(lambda x: fna_conditional_x(x)[2] * _normal_pdf(x), lower, upper, epsabs=1e-10)
    return {"Direct": float(dir_val), "Indirect": float(ind_val), "Total": float(tot_val),
            "Direct_err": float(dir_err), "Indirect_err": float(ind_err), "Total_err": float(tot_err)}


def print_oracle_diagnostics(dgp):
    eff = oracle_nde_nie_ate_quad(dgp)
    fna = oracle_fna_quad(dgp)
    print("-" * 60)
    print("Oracle NDE/NIE/ATE (numerical integration):")
    print(f"  NDE : {eff['NDE']:.6f}")
    print(f"  NIE : {eff['NIE']:.6f}")
    print(f"  ATE : {eff['ATE']:.6f}  (NDE+NIE check: {eff['NDE'] + eff['NIE']:.6f})")
    print("Oracle FNA:")
    print(f"  Direct  : {fna['Direct']:.6f}")
    print(f"  Indirect: {fna['Indirect']:.6f}")
    print(f"  Total   : {fna['Total']:.6f}")
    print("-" * 60)


# ===========================================================================
# 4. Main: resumable simulation loop  (notebook cells 21-24, 26)
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Path-specific FNA Makarov bounds simulation study (GPU script).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--code-dir", default=None,
                   help="Folder containing functions.py and crps_mu_models.py "
                        "(default: <repo>/src).")
    p.add_argument("--output-dir", default="./simulation_results",
                   help="Where to write result pickles (and plots, if enabled).")
    p.add_argument("--results-name", default="simulation_results.pkl",
                   help="Filename for the results pickle.")

    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument("--n-sims", type=int, default=500)
    p.add_argument("--n-sample", type=int, default=5000)
    p.add_argument("--n-mc", type=int, default=500)
    p.add_argument("--epochs1", type=int, default=30,
                   help="Epochs for stage 1: cross-fit pseudo outcomes.")
    p.add_argument("--epochs2", type=int, default=30,
                   help="Epochs for stage 2: conditional CDF / plugin regressions.")
    p.add_argument("--hidden", type=int, default=64,
                   help="Hidden layer width for CRPSMuNet.")
    p.add_argument("--n-layers", type=int, default=2,
                   help="Number of hidden layers in CRPSMuNet.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Mini-batch size.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="L2 weight decay (Adam).")
    p.add_argument("--dgp-type", default="linear",
                   choices=["linear", "polynomial", "sinusoidal"],
                   help="Family of DGP: linear (specify --cov for # covariates), "
                        "polynomial (3 covariates, X² and X1·X2 terms), "
                        "or sinusoidal (3 covariates, sin/cos terms).")
    p.add_argument("--cov", default="3cov",
                   choices=["1cov", "3cov", "5cov", "10cov", "20cov"],
                   help="Number of covariates for the linear DGP (ignored for "
                        "polynomial/sinusoidal).")
    p.add_argument("--print-every", type=int, default=50,
                   help="Print the aggregated table every N completed seeds.")

    p.add_argument("--save-plots", action="store_true",
                   help="Save per-seed CRPS pseudo-loss curves to <output-dir>/plots.")
    p.add_argument("--diagnostics", action="store_true",
                   help="Print oracle NDE/NIE/ATE and FNA values once, then continue.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    global ACTIVE_DGP, DGP
    if args.dgp_type == "polynomial":
        ACTIVE_DGP = "poly"
    elif args.dgp_type == "sinusoidal":
        ACTIVE_DGP = "sin"
    else:  # linear — use --cov to pick covariate count
        ACTIVE_DGP = args.cov
    DGP = _DGP_MAP[ACTIVE_DGP]

    dgp_tag = ACTIVE_DGP  # e.g. "3cov", "poly", "sin"
    output_dir = os.path.abspath(os.path.join(args.output_dir, dgp_tag))
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, args.results_name)
    plot_dir = os.path.join(output_dir, "plots")

    # --- GPU report (notebook cell 3) ---
    print("=" * 60)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({mem:.1f} GB)")
    else:
        print("No GPU detected -- this will run on CPU and be much slower.")
    print("=" * 60)

    print(f"Code dir   : {_CODE_DIR}")
    print(f"DGP        : type={args.dgp_type}  active={ACTIVE_DGP}")
    print(f"Simulations: {args.n_sims} seeds "
          f"({args.base_seed} to {args.base_seed + args.n_sims - 1})")
    print(f"n per sim  : {args.n_sample}   MC draws: {args.n_mc}   Epochs1: {args.epochs1}  Epochs2: {args.epochs2}")
    print(f"Model      : hidden={args.hidden}  n_layers={args.n_layers}  lr={args.lr}  batch={args.batch_size}  wd={args.weight_decay}")
    print(f"Output     : {output_dir}")
    print(f"Results    : {results_path}")
    print("=" * 60)

    if args.diagnostics:
        if "m_x_effect" in DGP:
            print_oracle_diagnostics(DGP)
        else:
            print("  [--diagnostics only supported for the 1-covariate linear DGP]")

    sim_results = []

    # --- Main loop (notebook cell 24) ---
    study_start = time.time()
    for sim_idx in range(args.n_sims):
        seed = args.base_seed + sim_idx
        print(f"\n{'=' * 70}")
        print(f"Simulation {sim_idx + 1}/{args.n_sims}  (seed={seed})")
        print(f"{'=' * 70}")

        result = run_single_simulation(
            seed=seed,
            n_sample=args.n_sample,
            n_mc=args.n_mc,
            epochs1=args.epochs1,
            epochs2=args.epochs2,
            hidden=args.hidden,
            n_layers=args.n_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            save_plots=args.save_plots,
            plot_dir=plot_dir,
        )
        sim_results.append(result)

        with open(results_path, "wb") as f:
            pickle.dump(sim_results, f)

        elapsed = time.time() - study_start
        remaining = (args.n_sims - sim_idx - 1) * elapsed / (sim_idx + 1)
        print(f"  Elapsed: {elapsed / 3600:.2f}h  |  Remaining: {remaining / 3600:.2f}h")

        if (sim_idx + 1) % args.print_every == 0 or (sim_idx + 1) == args.n_sims:
            print_aggregated(summarize_simulations(sim_results, base_seed=args.base_seed))

    # --- Final aggregate + save (notebook cell 26) ---
    final_stats = summarize_simulations(sim_results, base_seed=args.base_seed)
    with open(os.path.join(output_dir, "final_statistics.pkl"), "wb") as f:
        pickle.dump(final_stats, f)
    with open(os.path.join(output_dir, "all_simulation_results.pkl"), "wb") as f:
        pickle.dump(sim_results, f)
    print_aggregated(final_stats)

    print(f"\nALL DONE in {(time.time() - study_start) / 3600:.2f} hours")
    print(f"Saved final_statistics.pkl and all_simulation_results.pkl in {output_dir}")


if __name__ == "__main__":
    main()
