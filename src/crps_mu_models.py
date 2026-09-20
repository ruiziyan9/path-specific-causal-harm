"""
CRPS-based conditional CDF estimation — drop-in replacement for MuModels.

Key ideas
---------
1.  CRPSMuNet  (nn.Module)
    A small MLP whose output layer produces T values via a *monotone
    parameterisation*:

        raw    = Linear(hidden, T)          # unconstrained
        deltas = softplus(raw) + eps        # strictly positive increments
        cdf    = cumsum(deltas, dim=1)      # monotone, but unbounded
        cdf    = cdf / cdf[:, -1:]          # re-scale so last value → 1
                                            # (forces F(t_max|X) ≡ 1)

    This means monotonicity is a hard architectural constraint —
    no post-hoc isotonic regression needed.

2.  CRPS loss (discrete trapezoid approximation)
    CRPS(F, y) = ∫ (F(t) − 1{y≤t})² dt
               ≈ Σ_k w_k (F_k − 1{y≤t_k})²
    where w_k are trapezoid weights over the threshold grid Tn.
    Minimising E[CRPS] is a *strictly proper* scoring rule for the
    full conditional CDF, so the unique minimiser is the true F(t|X).

3.  CRPSMuModels  (dataclass, same public API as MuModels)
    .fit(X, A, M, Y, Tn, ...)   → CRPSMuModels instance
    .predict(a, M, X)           → (T, n)  ← identical to MuModels.predict
    .predict_batched(a, M_flat, X_rep) → (T, n*n_mc)

    fit_mu_crps(...)  factory — mirrors fit_mu() so you can swap
    "mu = fit_mu(...)" → "mu = fit_mu_crps(...)" with no other changes.

Usage
-----
    from functions import (fit_propensity, fit_mediator,
                           pseudo_outcomes_from_nuisances)
    from crps_mu_models import fit_mu_crps

    mu  = fit_mu_crps(X_tr, A_tr, M_tr, Y_tr, Tn, epochs=60)
    nuis = {"pi": pi, "g": g, "mu": mu}
    phi  = pseudo_outcomes_from_nuisances(X_te, A_te, M_te, Y_te, Tn, nuis)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (self-contained so this file can be used standalone)
# ──────────────────────────────────────────────────────────────────────────────

def _to_tensor(x, device=DEVICE):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

def _prepend_a(a_val, X):
    n = X.shape[0]
    col = np.full((n, 1), float(a_val)) if np.isscalar(a_val) \
          else np.asarray(a_val, dtype=float).reshape(n, 1)
    return np.column_stack([col, X])

def enforce_cdf_monotonicity(cdf, eps=1e-6):
    """Enforce monotonicity in threshold dimension (axis 0).
    Apply on shape Tn x n """
    cdf = np.clip(cdf, eps, 1 - eps)
    cdf = np.maximum.accumulate(cdf, axis=0)
    return np.clip(cdf, eps, 1 - eps)

# ──────────────────────────────────────────────────────────────────────────────
# Monotone CDF network
# ──────────────────────────────────────────────────────────────────────────────

class CRPSMuNet(nn.Module):
    """
    Small MLP that outputs a *strictly monotone* CDF vector for each input.

    Architecture
    ------------
    Input (n, p)  →  [Linear(hidden) → ReLU] × n_layers → Linear(T) [raw]
                  →  softplus(raw) + eps  [strictly positive deltas]
                  →  cumsum(dim=1)        [monotone, shape (n, T)]
                  →  / last_col           [last threshold maps to 1.0]

    Parameters
    ----------
    in_features : int   — number of input features (after prepending A, M)
    T           : int   — number of threshold grid points
    hidden      : int   — hidden layer width (default 64)
    n_layers    : int   — number of hidden layers (default 2)
    eps         : float — minimum increment to guarantee strict monotonicity
    """

    def __init__(self, in_features: int, T: int, hidden: int = 64,
                 n_layers: int = 2, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.T   = T
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, T))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (n, p)

        Returns
        -------
        cdf : (n, T)  — values in (0, 1], strictly increasing in dim=1
        """
        raw    = self.net(x)                                 # (n, T)
        deltas = torch.nn.functional.softplus(raw) + self.eps  # (n, T) > 0
        cdf    = torch.cumsum(deltas, dim=1)                 # (n, T) increasing
        # Normalise so the last threshold maps to 1
        cdf    = cdf / cdf[:, -1:].clamp(min=1e-8)          # (n, T) in (0, 1]
        return cdf


# ──────────────────────────────────────────────────────────────────────────────
# CRPS loss
# ──────────────────────────────────────────────────────────────────────────────

def crps_loss(cdf_pred: torch.Tensor,
              y: torch.Tensor,
              Tn_t: torch.Tensor) -> torch.Tensor:
    """
    Discrete CRPS loss (trapezoid rule over the threshold grid).

    CRPS(F, y) = ∫ (F(t) − 1{y ≤ t})² dt
               ≈ Σ_k w_k · (F_k − H_k)²

    where H_k = 1{y ≤ t_k} is the empirical CDF of the single observation y,
    and w_k are trapezoid weights:
        w_0 = (t_1 - t_0) / 2
        w_k = (t_{k+1} - t_{k-1}) / 2   for 0 < k < T-1
        w_{T-1} = (t_{T-1} - t_{T-2}) / 2

    Parameters
    ----------
    cdf_pred : (n, T)  — predicted CDF values, already in (0, 1]
    y        : (n,)    — observed outcomes
    Tn_t     : (T,)    — threshold grid (must be sorted ascending)

    Returns
    -------
    scalar loss (mean over n)
    """
    # Heaviside targets  H_{i,k} = 1{y_i <= t_k},  shape (n, T)
    H = (y.unsqueeze(1) <= Tn_t.unsqueeze(0)).float()   # (n, T)

    # Trapezoid weights, shape (T,)
    w = torch.zeros_like(Tn_t)
    w[0]    = (Tn_t[1]  - Tn_t[0])  / 2.0
    w[-1]   = (Tn_t[-1] - Tn_t[-2]) / 2.0
    w[1:-1] = (Tn_t[2:] - Tn_t[:-2]) / 2.0             # central differences

    # Weighted squared error, mean over n and sum over t
    sq = (cdf_pred - H) ** 2                             # (n, T)
    return (sq * w.unsqueeze(0)).sum(dim=1).mean()       # scalar


# ──────────────────────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────────────────────

def _fit_crps_net(features: np.ndarray,
                  Y: np.ndarray,
                  Tn: np.ndarray,
                  hidden: int = 64,
                  n_layers: int = 2,
                  lr: float = 1e-3,
                  epochs: int = 60,
                  batch_size: int = 512,
                  weight_decay: float = 1e-4,
                  device: str = DEVICE) -> CRPSMuNet:
    """
    Fit a CRPSMuNet on (features, Y) using mini-batch SGD.

    Parameters
    ----------
    features   : (n, p)  — input features (already includes A, M as needed)
    Y          : (n,)    — observed outcomes
    Tn         : (T,)    — sorted threshold grid
    hidden     : int     — hidden width
    n_layers   : int     — number of hidden layers
    lr         : float   — Adam learning rate
    epochs     : int     — number of passes over the data
    batch_size : int     — mini-batch size (use n if n < batch_size)
    weight_decay : float — L2 regularisation

    Returns
    -------
    Trained CRPSMuNet on `device`.
    """
    n, p = features.shape
    T    = len(Tn)

    X_t  = _to_tensor(features, device)
    Y_t  = _to_tensor(Y,        device)
    Tn_t = _to_tensor(Tn,       device)

    model = CRPSMuNet(in_features=p, T=T, hidden=hidden, n_layers=n_layers).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    bs = min(batch_size, n)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx     = perm[start: start + bs]
            cdf_hat = model(X_t[idx])                    # (bs, T)
            loss    = crps_loss(cdf_hat, Y_t[idx], Tn_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


# ──────────────────────────────────────────────────────────────────────────────
# CRPSMuModels — drop-in replacement for MuModels
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CRPSMuModels:
    """
    Drop-in replacement for MuModels.

    Trains CRPSMuNet(s) with the CRPS proper scoring rule.
    Monotonicity is enforced architecturally (cumsum of softplus increments),
    so no post-hoc isotonic regression is needed.

    Public API is identical to MuModels:
        .predict(a, M, X)               → (T, n)
        .predict_batched(a, M_flat, X_rep) → (T, n*n_mc)
    """
    t_learner: bool
    T:         int
    model:  Optional[CRPSMuNet] = field(default=None)   # S-learner
    model0: Optional[CRPSMuNet] = field(default=None)   # T-learner, A=0
    model1: Optional[CRPSMuNet] = field(default=None)   # T-learner, A=1

    @classmethod
    def fit(cls, X, A, M, Y, Tn,
            hidden: int = 64,
            n_layers: int = 2,
            lr: float = 1e-3,
            epochs: int = 60,
            batch_size: int = 256,
            weight_decay: float = 1e-4,
            t_learner: bool = False,
            device: str = DEVICE):
        """
        Parameters
        ----------
        X, A, M, Y : numpy arrays — covariates, treatment, mediator, outcome
        Tn         : (T,) sorted threshold grid
        t_learner  : if True fit separate models per arm (recommended for
                     heterogeneous treatment effects); if False, prepend A
                     to features and fit one S-learner
        hidden     : hidden layer width for CRPSMuNet
        n_layers   : number of hidden layers
        lr, epochs, batch_size, weight_decay : training hyper-parameters
        """
        T = len(Tn)
        if t_learner:
            feat0 = np.column_stack([M[A == 0], X[A == 0]])
            feat1 = np.column_stack([M[A == 1], X[A == 1]])
            m0 = _fit_crps_net(feat0, Y[A == 0], Tn,
                               hidden=hidden, n_layers=n_layers,
                               lr=lr, epochs=epochs,
                               batch_size=batch_size,
                               weight_decay=weight_decay, device=device)
            m1 = _fit_crps_net(feat1, Y[A == 1], Tn,
                               hidden=hidden, n_layers=n_layers,
                               lr=lr, epochs=epochs,
                               batch_size=batch_size,
                               weight_decay=weight_decay, device=device)
            return cls(t_learner=True, T=T, model0=m0, model1=m1)
        else:
            # S-learner: prepend (A, M) to X
            feat = np.column_stack([A, M, X])
            m = _fit_crps_net(feat, Y, Tn,
                              hidden=hidden, n_layers=n_layers,
                              lr=lr, epochs=epochs,
                              batch_size=batch_size,
                              weight_decay=weight_decay, device=device)
            return cls(t_learner=False, T=T, model=m)

    # ── internal ──────────────────────────────────────────────────────────────

    def _predict_raw(self, a, M, X) -> np.ndarray:
        """
        Returns (T, n) CDF array for treatment arm a.
        CRPSMuNet outputs (n, T); we transpose for consistency with MuModels.
        """
        n = X.shape[0]
        if self.t_learner:
            net  = self.model0 if a == 0 else self.model1
            feat = np.column_stack([M, X])
        else:
            net  = self.model
            feat = np.column_stack([np.full(n, float(a)), M, X])

        X_t = _to_tensor(feat, next(net.parameters()).device)
        with torch.no_grad():
            cdf = net(X_t).cpu().numpy()   # (n, T)
        return cdf.T                       # (T, n)

    # ── public API (identical to MuModels) ────────────────────────────────────

    def predict(self, a, M, X) -> np.ndarray:
        """Returns (T, n) array of P(Y ≤ t | A=a, M, X) for each threshold."""
        return self._predict_raw(a, M, X)

    def predict_batched(self, a, M_flat, X_rep) -> np.ndarray:
        """
        Batched prediction for MC integration.

        Parameters
        ----------
        M_flat : (n * n_mc,)    mediator draws, flattened
        X_rep  : (n * n_mc, p)  covariates, repeated n_mc times

        Returns
        -------
        (T, n * n_mc) — one GPU forward pass
        """
        return self._predict_raw(a, M_flat, X_rep)


# ──────────────────────────────────────────────────────────────────────────────
# Factory — mirrors fit_mu() for easy swapping
# ──────────────────────────────────────────────────────────────────────────────

def fit_mu_crps(X, A, M, Y, Tn,
                hidden: int = 64,
                n_layers: int = 2,
                lr: float = 1e-3,
                epochs: int = 60,
                batch_size: int = 256,
                weight_decay: float = 1e-4,
                t_learner: bool = False,
                device: str = DEVICE) -> CRPSMuModels:
    """
    Factory for CRPSMuModels.  Mirrors fit_mu() so you can swap

        mu = fit_mu(X, A, M, Y, Tn, epochs=epochs, t_learner=t_learner)

    to

        mu = fit_mu_crps(X, A, M, Y, Tn, epochs=epochs, t_learner=t_learner)

    with no other changes to the codebase.
    """
    return CRPSMuModels.fit(
        X, A, M, Y, Tn,
        hidden=hidden, n_layers=n_layers, lr=lr, epochs=epochs,
        batch_size=batch_size, weight_decay=weight_decay,
        t_learner=t_learner, device=device,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# CRPS loss for pseudo-outcome targets  (second stage)
# ──────────────────────────────────────────────────────────────────────────────

def crps_loss_pseudo(cdf_pred: torch.Tensor,
                     phi_t: torch.Tensor,
                     w: torch.Tensor) -> torch.Tensor:
    """
    CRPS loss where targets are pseudo-outcomes rather than binary indicators.

    In the second stage we have pseudo-outcome vectors phi_i ∈ R^T  (shape
    (n, T)) that satisfy E[phi_i(t) | X_i] = F_{aa'}(t | X_i).  The CRPS
    objective becomes:

        L = Σ_k w_k · E[(F_hat(t_k | X) − phi(t_k))²]

    which is still a strictly proper scoring rule for F_{aa'}(t | X) because
    the pseudo-outcomes are unbiased conditionally on X.

    Parameters
    ----------
    cdf_pred : (n, T)  — predicted CDF (monotone, from CRPSMuNet)
    phi_t    : (n, T)  — pseudo-outcome matrix (rows = obs, cols = thresholds)
    w        : (T,)    — trapezoid weights (pre-computed from Tn)

    Returns
    -------
    scalar loss
    """
    sq = (cdf_pred - phi_t) ** 2                         # (n, T)
    return (sq * w.unsqueeze(0)).sum(dim=1).mean()        # scalar


def _trapezoid_weights(Tn: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    """Pre-compute trapezoid weights for a given grid Tn."""
    Tn_t = _to_tensor(Tn, device)
    w    = torch.zeros_like(Tn_t)
    w[0]    = (Tn_t[1]  - Tn_t[0])  / 2.0
    w[-1]   = (Tn_t[-1] - Tn_t[-2]) / 2.0
    w[1:-1] = (Tn_t[2:] - Tn_t[:-2]) / 2.0
    return w                                              # (T,)


def _fit_crps_net_pseudo(X: np.ndarray,
                         phi_mat: np.ndarray,
                         Tn: np.ndarray,
                         hidden: int = 64,
                         n_layers: int = 2,
                         lr: float = 1e-3,
                         epochs: int = 60,
                         batch_size: int = 512,
                         weight_decay: float = 1e-4,
                         eval_X=None, eval_phi=None, print_every=10,
                         device: str = DEVICE) -> CRPSMuNet:
    """
    Fit a CRPSMuNet using pseudo-outcome targets.

    Parameters
    ----------
    X       : (n, p)  — covariates  (second-stage features, NOT including A/M)
    phi_mat : (n, T)  — pseudo-outcome matrix from crossfit_pseudo_outcomes
                        (already transposed from the (T, n) storage convention)
    Tn      : (T,)    — sorted threshold grid
    hidden, n_layers, lr, epochs, batch_size, weight_decay : training hyper-parameters

    Returns
    -------
    Trained CRPSMuNet.
    """
    n, p = X.shape
    T    = len(Tn)

    X_t   = _to_tensor(X,       device)
    phi_t = _to_tensor(phi_mat, device)       # (n, T)
    w     = _trapezoid_weights(Tn, device)    # (T,)

    have_eval = eval_X is not None and eval_phi is not None
    if have_eval:
        Xe_t   = _to_tensor(eval_X,   device)
        phie_t = _to_tensor(eval_phi, device)

    model = CRPSMuNet(in_features=p, T=T, hidden=hidden, n_layers=n_layers).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def _clean_loss(Xb, phib):
        # identical metric for train and eval; batched to stay memory-safe
        model.eval()
        tot, cnt = 0.0, 0
        for s in range(0, Xb.shape[0], bs):
            xb, pb = Xb[s:s + bs], phib[s:s + bs]
            tot += crps_loss_pseudo(model(xb), pb, w).item() * xb.shape[0]
            cnt += xb.shape[0]
        return tot / cnt

    history = {"epoch": [], "train_loss": []}
    if have_eval:
        history["test_loss"] = []

    bs = min(batch_size, n)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device) 

        for start in range(0, n, bs):
            idx     = perm[start: start + bs]
            cdf_hat = model(X_t[idx])                    # (bs, T)
            loss    = crps_loss_pseudo(cdf_hat, phi_t[idx], w)
            opt.zero_grad()
            loss.backward()
            opt.step()

        tr = _clean_loss(X_t, phi_t)          
        history["epoch"].append(epoch)
        history["train_loss"].append(tr)

        if have_eval:
            te = _clean_loss(Xe_t, phie_t)
            history["test_loss"].append(te)


        if print_every and (epoch % print_every == 0 or epoch == epochs - 1):
            msg = f"  epoch {epoch:03d}/{epochs - 1} | train={tr:.6f}"
            if have_eval:
                msg += f" | held-out={te:.6f}"
            print(msg)


    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Second-stage API — mirrors fit_conditional_cdfs / crossfit_conditional_cdfs
# ──────────────────────────────────────────────────────────────────────────────


def crossfit_conditional_cdfs_crps(phi, X, Tn, splits,
                                    hidden: int = 64,
                                    n_layers: int = 2,
                                    lr: float = 1e-3,
                                    epochs: int = 60,
                                    batch_size: int = 256,
                                    weight_decay: float = 1e-4,
                                    device: str = DEVICE,
                                    print_every: int = 10,
                                    return_curves: bool = True):
    """
    Second-stage cross-fitting with CRPS loss.

    Drop-in replacement for crossfit_conditional_cdfs().  Uses the same fold
    splits as the first stage to satisfy the double cross-fitting condition.
    Each fold model is trained on pseudo-outcomes from all *other* folds and
    predicts for the held-out fold.

    Parameters
    ----------
    phi    : dict  — from crossfit_pseudo_outcomes, keys (a,a'), values (T, n)
    X      : (n, p)
    Tn     : (T,)  sorted threshold grid
    splits : list of (train_idx, test_idx) — from crossfit_pseudo_outcomes
    hidden, n_layers, lr, epochs, batch_size, weight_decay : training hyper-parameters

    Returns
    -------
    cdfs : dict  — cdfs[pair] shape (T, n), fully out-of-sample fitted values
    """
    n     = X.shape[0]
    T     = len(Tn)
    pairs = list(phi.keys())
    cdfs  = {p: np.zeros((T, n)) for p in pairs}
    curves = {}

    for pair, phi_pair in phi.items():
        phi_mat = phi_pair.T     # (n, T)
        fold_train, fold_test = [], []  

        for k, (train_idx, test_idx) in enumerate(splits):
            if print_every:
                print(f"[pair {pair}  fold {k}]")
            model, hist = _fit_crps_net_pseudo(
                X[train_idx], phi_mat[train_idx], Tn,
                hidden=hidden, n_layers=n_layers, lr=lr, epochs=epochs,
                batch_size=batch_size, weight_decay=weight_decay,
                device=device,
                eval_X=X[test_idx], eval_phi=phi_mat[test_idx],
                print_every=print_every,
            )


            with torch.no_grad():
                preds = model(_to_tensor(X[test_idx], device))      # (T, n_te)
            cdfs[pair][:, test_idx] = preds.cpu().numpy().T 

            fold_train.append(hist["train_loss"])
            fold_test.append(hist["test_loss"])

        tr = np.array(fold_train)   # (K, E)
        te = np.array(fold_test)    # (K, E)
        curves[pair] = {
            "epoch":      np.arange(tr.shape[1]),
            "train_mean": tr.mean(0), "train_std": tr.std(0),
            "test_mean":  te.mean(0), "test_std":  te.std(0),
            "folds": [{"train_loss": fold_train[i], "test_loss": fold_test[i]}
                      for i in range(len(splits))],
        }
        # with torch.no_grad():
        #     preds = model(_to_tensor(x_grid, device).T)      # (T, n_te)
        #     cdfs[pair] = preds.cpu().numpy().T 

        #cdfs[pair] = enforce_cdf_monotonicity(cdfs[pair])


    return (cdfs, curves) if return_curves else cdfs




def predict_conditional_cdfs_crps(models, X, Tn, device: str = DEVICE):
    """
    Out-of-sample prediction using models from fit_conditional_cdfs_crps.

    Drop-in replacement for predict_conditional_cdfs().  No clipping or
    monotonicity enforcement needed — CRPSMuNet guarantees both.

    Parameters
    ----------
    models : dict  — models[pair] is a CRPSMuNet from fit_conditional_cdfs_crps
    X      : (n_new, p)
    Tn     : (T,)  (passed for API consistency; not used in forward pass)

    Returns
    -------
    cdfs : dict  — cdfs[pair] shape (T, n_new)
    """
    cdfs = {}
    for pair, model in models.items():
        X_t = _to_tensor(X, next(model.parameters()).device)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy().T            # (T, n_new)
        cdfs[pair] = preds
    return cdfs


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.normal(size=(n, p))
    A = rng.binomial(1, 0.5, n)
    M = rng.binomial(1, 0.4 + 0.2 * (A == 1), n)
    Y = 0.5 * X[:, 0] + 0.8 * A - 0.3 * M + rng.normal(scale=0.3, size=n)
    Tn = np.linspace(Y.min() - 0.5, Y.max() + 0.5, 40)

    print("Fitting CRPSMuModels (S-learner)...")
    mu = fit_mu_crps(X, A, M, Y, Tn, epochs=30, t_learner=False)

    # predict returns (T, n)
    pred = mu.predict(a=1, M=M, X=X)
    assert pred.shape == (len(Tn), n), f"unexpected shape {pred.shape}"

    # Check monotonicity: each column should be non-decreasing
    diffs = np.diff(pred, axis=0)
    assert (diffs >= -1e-5).all(), "monotonicity violated!"

    # Check values in (0, 1]
    assert pred.min() > 0 and pred.max() <= 1 + 1e-5

    # predict_batched (for MC eta integration)
    n_mc = 10
    M_flat = np.tile(M, n_mc)
    X_rep  = np.repeat(X, n_mc, axis=0)
    pred_b = mu.predict_batched(a=1, M_flat=M_flat, X_rep=X_rep)
    assert pred_b.shape == (len(Tn), n * n_mc)

    print(f"  predict shape     : {pred.shape}     ✓")
    print(f"  predict_batched   : {pred_b.shape}  ✓")
    print(f"  monotone          : {(diffs >= -1e-5).all()}  ✓")
    print(f"  range             : [{pred.min():.4f}, {pred.max():.4f}]  ✓")

    # ── Second-stage: simulate pseudo-outcomes ────────────────────────────────
    print("\nTesting second-stage CRPS conditional CDF fitting...")
    T = len(Tn)
    pairs = [(0, 0), (1, 0), (1, 1)]

    rng2 = np.random.default_rng(42)
    phi = {}
    for pair in pairs:
        true_cdf = np.clip(
            np.linspace(0.05, 0.95, T)[:, None]
            + 0.05 * rng2.normal(size=(T, n)), 0, 1
        )
        phi[pair] = true_cdf + 0.15 * rng2.normal(size=(T, n))

    # fit_conditional_cdfs_crps
    cdfs, models = fit_conditional_cdfs_crps(phi, X, Tn, epochs=20)
    for pair in pairs:
        arr = cdfs[pair]
        assert arr.shape == (T, n), f"wrong shape {arr.shape}"
        assert (np.diff(arr, axis=0) >= -1e-5).all(), f"non-monotone for {pair}"
        assert arr.min() > 0 and arr.max() <= 1 + 1e-5
    print("  fit_conditional_cdfs_crps        ✓")

    # predict_conditional_cdfs_crps (out-of-sample)
    X_new = rng2.normal(size=(50, p))
    cdfs_new = predict_conditional_cdfs_crps(models, X_new, Tn)
    for pair in pairs:
        arr = cdfs_new[pair]
        assert arr.shape == (T, 50)
        assert (np.diff(arr, axis=0) >= -1e-5).all(), f"non-monotone OOS {pair}"
    print("  predict_conditional_cdfs_crps    ✓")

    # crossfit_conditional_cdfs_crps
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=0)
    splits = list(kf.split(X))
    cdfs_cf = crossfit_conditional_cdfs_crps(phi, X, Tn, splits, epochs=20)
    for pair in pairs:
        arr = cdfs_cf[pair]
        assert arr.shape == (T, n)
        assert (np.diff(arr, axis=0) >= -1e-5).all(), f"non-monotone CF {pair}"
    print("  crossfit_conditional_cdfs_crps   ✓")

    print("\nAll checks passed.")