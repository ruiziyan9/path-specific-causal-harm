"""
Optimised utilities for algorithm2_minimax_policy_learner.

Key changes vs. original functions.py
--------------------------------------
1. MuModels.fit / MuModels.predict
     - Replaced T separate logistic models with ONE nn.Linear(features, T)
       trained end-to-end.  Eliminates T optimizer instantiations and T
       separate forward/backward passes.

2. fit_conditional_cdfs / predict_conditional_cdfs
     - Replaced T sequential fit_torch_linear calls with ONE nn.Linear(p, T)
       trained on the full (n, T) target matrix in a single loop.
       3 pairs × T=50 models → 3 models total.

3. pseudo_outcomes_from_nuisances  (and eta_mc inside fit_plugin_cdfs /
   predict_plugin_cdfs)
     - MC loop over n_mc draws replaced by a single batched forward pass:
       tile X and flatten M_draws → one call to mu.predict_batched → reshape.
     - Intermediate tensors stay on GPU; one .cpu().numpy() at the end.

4. MuModels.predict_batched
     - New method that accepts M of shape (n*n_mc,) and X of shape (n*n_mc, p)
       and returns (T, n*n_mc) in one GPU forward pass.

5. crossfit_pseudo_outcomes
     - Folds are now run in parallel via joblib (n_jobs=K).

Everything else is unchanged so the rest of the codebase is a drop-in.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field
from typing import Optional, List, Union
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SMOOTH_POTENTIAL = False



def simulate_path_specific_harm_dgp(n=10000, seed=2026, sigma_y=0.12):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n); X2 = rng.normal(size=n)
    X3 = rng.normal(size=n); X4 = rng.normal(size=n); X5 = rng.normal(size=n)
    S = (X1 > 0).astype(int)
    pA = expit(0.20 + 0.35 * X1 + 0.25 * X2 - 0.20 * X3)
    A  = (rng.uniform(size=n) < pA).astype(int)
    pM0 = expit(-0.20 + 0.15 * X1 - 0.25 * X2 + 0.15 * X1 * X2)
    pM1 = expit(-0.20 + 1.10 + 0.35 * X1 - 0.25 * X2 + 0.15 * X1 * X2)
    U_M = rng.uniform(size=n)
    def _M(a):
        return (U_M < (pM1 if a == 1 else pM0)).astype(int)
    M0, M1 = _M(0), _M(1)
    M = np.where(A == 1, M1, M0)
    bX = 0.1*X1 - 0.05*X2 + 0.01*X3 + 0.05*X4 - 0.05*X5
    d  = np.where(S == 1, -0.60, 1.10)
    q  = np.where(S == 1,  1.10,-1.00)
    c  = 0.1
    U_Y = rng.normal(size=n)
    def _Y(a, m):
        return bX + d * a + q * m + c * a * m + sigma_y * U_Y
    Y00, Y01, Y10, Y11 = _Y(0,M0), _Y(0,M1), _Y(1,M0), _Y(1,M1)
    Y = np.where(A == 1, Y11, Y00)
    return dict(X1=X1,X2=X2,X3=X3,X4=X4,X5=X5,S=S,A=A,M=M,Y=Y,
                pA=pA,pM0=pM0,pM1=pM1,M0=M0,M1=M1,
                Y00=Y00,Y01=Y01,Y10=Y10,Y11=Y11,
                sigma_y=sigma_y,bX=bX,d=d,q=q,c=c)

def simulate_path_specific_harm_dgp(n=10000, seed=2026, sigma_y=0.3, sigma_m_noise=0.25):
    rng = np.random.default_rng(seed)

    X1 = rng.normal(size=n)
    # X2 = rng.normal(size=n)
    S  = (X1 > 0).astype(int)

    # (1) Stable propensity — bounded well away from 0/1
    # pA  = expit(0.20 + 0.40 * X1 + 0.20 * X2)
    pA  = expit(0.20 + 0.40 * X1 )
    A   = (rng.uniform(size=n) < pA).astype(int)

    # (1) Stable mediator propensity — intercept shift of 1.1 guarantees
    #     meaningful treatment effect on M without extreme probabilities
    #pM0 = expit(-0.30 + 0.20 * X1 - 0.20 * X2)
    pM0 = expit(-0.30 + 0.20 * X1 )
    #pM1 = expit(-0.30 + 1.10 + 0.20 * X1 - 0.20 * X2)
    pM1 = expit(-0.30 + 1.10 + 0.20 * X1)
    U_M = rng.uniform(size=n)
    M0  = (U_M < pM0).astype(int)
    M1  = (U_M < pM1).astype(int)
    M   = np.where(A == 1, M1, M0)

    # (2) & (3) Outcome: shared noise + mediator-specific noise
    #     Shared noise keeps Direct bounds tight (Y10 vs Y00 are co-monotone)
    #     Mediator noise breaks co-monotonicity of Y11 vs Y10 → ind+dir ≠ total
    d  = np.where(S == 1, -0.60,  1.10)
    q  = np.where(S == 1,  1.10, -1.00)
    c  = 0.10
    # bX = 0.10 * X1 - 0.05 * X2
    bX = 0.10 * X1

    U_Y_base = rng.normal(size=n)   # shared: drives co-monotonicity of Y00, Y10
    U_Y_med  = rng.normal(size=n)   # M-specific: breaks Y10 vs Y11 co-monotonicity

    def _Y(a, m):
        return bX + d*a + q*m + c*a*m \
               + sigma_y * U_Y_base \
               + sigma_m_noise * m * U_Y_med

    Y00, Y10, Y11 = _Y(0, M0), _Y(1, M0), _Y(1, M1)
    Y = np.where(A == 1, Y11, Y00)

    return dict(
        X1=X1, #X2=X2,
        S=S, A=A, M=M, Y=Y,
        pA=pA, pM0=pM0, pM1=pM1, M0=M0, M1=M1,
        Y00=Y00, Y10=Y10, Y11=Y11,
        sigma_y=sigma_y, sigma_m_noise=sigma_m_noise,
        bX=bX, d=d, q=q, c=c
    )


def simulate_margin_friendly_harm_dgp(n=10000, seed=2026,
                                      sigma_y=0.6,
                                      sigma_m_noise=1.5):
    """
    DGP designed to be friendlier to the DR conditional-Makarov estimator.

    The current path-specific DGP often creates conditional CDF differences
    that are nearly one-sided stochastic shifts. Then either sup Delta_X(t)
    or sup -Delta_X(t) is close to the zero kink with positive probability,
    which is exactly the nonregular case excluded by the margin assumption.

    This DGP makes the relevant conditional CDFs cross for most X:
      * Direct: among M=0 units treatment shifts Y upward, while among M=1
        units the interaction shifts Y downward.
      * Indirect: treatment increases the probability of M=1, and the M=1
        outcome component has both a lower mean and a larger variance.

    The nuisance functions are nonlinear in X but overlap-safe. That makes
    naive plug-in more exposed to first-stage bias, while the DR pseudo-outcome
    correction has a better chance to help.
    """
    rng = np.random.default_rng(seed)

    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    X3 = rng.normal(size=n)

    pA = expit(0.15 + 0.35 * X1 - 0.25 * X2 + 0.20 * np.sin(X1 * X2))

    m_base = (
        -0.55
        + 0.35 * X1
        - 0.25 * X2
        + 0.20 * X3
        + 0.35 * np.sin(X1)
        - 0.15 * (X2**2 - 1.0)
    )
    pM0 = expit(m_base)
    pM1 = expit(m_base + 1.05)

    A = (rng.uniform(size=n) < pA).astype(int)
    U_M = rng.uniform(size=n)
    M0 = (U_M < pM0).astype(int)
    M1 = (U_M < pM1).astype(int)
    M = np.where(A == 1, M1, M0)

    bX = (
        0.45 * np.sin(X1)
        + 0.25 * (X2**2 - 1.0)
        - 0.20 * X3
        + 0.25 * X1 * X2
    )

    # Direct comparison:
    #   m=0 component: Y(1,m) - Y(0,m) = d > 0
    #   m=1 component: Y(1,m) - Y(0,m) = d + c < 0
    # This prevents the direct bound from sitting on a pure stochastic-order
    # boundary for most X.
    d = 0.65 + 0.10 * np.tanh(X1)
    q = 0.90 + 0.10 * np.tanh(X2)
    c = -1.35

    U_Y_base = rng.normal(size=n)
    U_Y_med = rng.normal(size=n)

    def _Y(a, m):
        return (
            bX
            + d * a
            + q * m
            + c * a * m
            + sigma_y * U_Y_base
            + sigma_m_noise * m * U_Y_med
        )

    Y00 = _Y(0, M0)
    Y10 = _Y(1, M0)
    Y11 = _Y(1, M1)
    Y = np.where(A == 1, Y11, Y00)

    return dict(
        X1=X1, X2=X2, X3=X3,
        A=A, M=M, Y=Y,
        pA=pA, pM0=pM0, pM1=pM1,
        M0=M0, M1=M1,
        Y00=Y00, Y10=Y10, Y11=Y11,
        sigma_y=sigma_y, sigma_m_noise=sigma_m_noise,
        bX=bX, d=d, q=q, c=c,
    )


def empirical_cdf(samples, grid):
    samples = np.asarray(samples)
    return np.array([(samples <= t).mean() for t in grid])


def makarov_bounds_from_cdfs(Fu, Fv):
    delta = Fu - Fv
    return max(delta.max(), 0.0), 1.0 - max((-delta).max(), 0.0)

def subset_dgp_dict(data, idx):
    # All keys whose values are scalars (not arrays) — extend if DGP adds more
    scalars = {"sigma_y", "sigma_m_noise", "c"}
    out = {}
    for k, v in data.items():
        out[k] = v if k in scalars else v[idx]
    # Build X from whatever Xi columns actually exist in data
    xi_cols = sorted(k for k in data if k.startswith("X") and k[1:].isdigit())
    out["X"] = np.column_stack([data[col][idx] for col in xi_cols])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Low-level torch helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_tensor(x, device=DEVICE):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

# Single-output helpers (propensity, mediator, policy) — kept as before
def fit_torch_linear(X, y, lr=1e-2, epochs=10, weight_decay=1e-4, device=DEVICE):
    X_t = to_tensor(X, device); y_t = to_tensor(y, device).view(-1, 1)
    model = nn.Linear(X_t.shape[1], 1).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(); nn.MSELoss()(model(X_t), y_t).backward(); opt.step()
    return model

def fit_torch_logistic(X, y, lr=1e-2, epochs=10, weight_decay=1e-4, device=DEVICE):
    X_t = to_tensor(X, device); y_t = to_tensor(y, device).view(-1, 1)
    model = nn.Linear(X_t.shape[1], 1).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(); nn.BCEWithLogitsLoss()(model(X_t), y_t).backward(); opt.step()
    return model

def predict_torch_linear(model, X, device=DEVICE):
    with torch.no_grad():
        return model(to_tensor(X, device)).squeeze(-1).cpu().numpy()

def predict_torch_logistic_proba(model, X, device=DEVICE):
    with torch.no_grad():
        return torch.sigmoid(model(to_tensor(X, device)).squeeze(-1)).cpu().numpy()

# ── NEW: multi-output helpers ──────────────────────────────────────────────────

def fit_torch_linear_multi(X, Y_mat, lr=1e-2, epochs=40,
                           weight_decay=1e-4, device=DEVICE):
    """
    Fit a single nn.Linear(p, T) on target matrix Y_mat (n, T).
    Replaces T sequential fit_torch_linear calls.
    """
    X_t = to_tensor(X, device)                       # (n, p)
    Y_t = to_tensor(Y_mat, device)                    # (n, T)
    T   = Y_t.shape[1]
    model = nn.Linear(X_t.shape[1], T).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X_t), Y_t).backward()
        opt.step()
    return model

def fit_torch_logistic_multi(X, Y_mat, lr=1e-2, epochs=40,
                              weight_decay=1e-4, device=DEVICE):
    """
    Fit a single nn.Linear(p, T) with BCE on binary target matrix Y_mat (n, T).
    Replaces T sequential fit_torch_logistic calls (MuModels).
    """
    X_t = to_tensor(X, device)
    Y_t = to_tensor(Y_mat, device)                    # (n, T)
    T   = Y_t.shape[1]
    model = nn.Linear(X_t.shape[1], T).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X_t), Y_t).backward()
        opt.step()
    return model

def predict_torch_logistic_proba_multi(model, X, device=DEVICE):
    """Returns (T, n) probability array — GPU → CPU in one call."""
    with torch.no_grad():
        logits = model(to_tensor(X, device))          # (n, T)
        return torch.sigmoid(logits).cpu().numpy().T  # (T, n)

def predict_torch_linear_multi(model, X, device=DEVICE):
    """Returns (T, n) prediction array."""
    with torch.no_grad():
        return model(to_tensor(X, device)).cpu().numpy().T  # (T, n)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_binary(arr):
    u = np.unique(arr)
    return len(u) == 2 and set(u).issubset({0, 1})

def _prepend_a(a_val_or_vec, X):
    n = X.shape[0]
    a_col = np.full((n, 1), float(a_val_or_vec)) if np.isscalar(a_val_or_vec) \
            else np.asarray(a_val_or_vec, dtype=float).reshape(n, 1)
    return np.column_stack([a_col, X])


# ──────────────────────────────────────────────────────────────────────────────
# Propensity model (unchanged — single-output is fine here)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PropensityModel:
    model: nn.Module

    @classmethod
    def fit(cls, X, A, epochs=10):
        return cls(model=fit_torch_logistic(X, A, epochs=epochs))

    def predict_proba(self, X):
        p1 = predict_torch_logistic_proba(self.model, X)
        return np.column_stack([1.0 - p1, p1])


# ──────────────────────────────────────────────────────────────────────────────
# Mediator models (unchanged — per-unit density/sample, single output)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GaussianMModel:
    t_learner: bool
    sigma_: float
    model:  Optional[nn.Module] = field(default=None)
    model0: Optional[nn.Module] = field(default=None)
    model1: Optional[nn.Module] = field(default=None)

    @classmethod
    def fit(cls, X, A, M, epochs=10, t_learner=False):
        if t_learner:
            m0 = fit_torch_linear(X[A==0], M[A==0], epochs=epochs)
            m1 = fit_torch_linear(X[A==1], M[A==1], epochs=epochs)
            resid = np.concatenate([M[A==0] - predict_torch_linear(m0, X[A==0]),
                                    M[A==1] - predict_torch_linear(m1, X[A==1])])
            return cls(t_learner=True, sigma_=float(np.std(resid))+1e-6,
                       model0=m0, model1=m1)
        else:
            XA = _prepend_a(A, X)
            m  = fit_torch_linear(XA, M, epochs=epochs)
            return cls(t_learner=False,
                       sigma_=float(np.std(M - predict_torch_linear(m, XA)))+1e-6,
                       model=m)

    def _mu(self, a, X):
        if self.t_learner:
            return predict_torch_linear(self.model0 if a==0 else self.model1, X)
        return predict_torch_linear(self.model, _prepend_a(a, X))

    def density(self, a, X, m):
        mu = self._mu(a, X)
        z  = (m - mu) / self.sigma_
        return (1./(np.sqrt(2*np.pi)*self.sigma_)) * np.exp(-0.5*z**2)

    def sample(self, a, X, n_draws=50, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        mu  = self._mu(a, X)
        return rng.normal(mu[:, None], self.sigma_, size=(len(X), n_draws))


@dataclass
class BernoulliMModel:
    t_learner: bool
    model:  Optional[nn.Module] = field(default=None)
    model0: Optional[nn.Module] = field(default=None)
    model1: Optional[nn.Module] = field(default=None)

    @classmethod
    def fit(cls, X, A, M, epochs=10, t_learner=False):
        if t_learner:
            return cls(t_learner=True,
                       model0=fit_torch_logistic(X[A==0], M[A==0], epochs=epochs),
                       model1=fit_torch_logistic(X[A==1], M[A==1], epochs=epochs))
        return cls(t_learner=False,
                   model=fit_torch_logistic(_prepend_a(A, X), M, epochs=epochs))

    def _proba(self, a, X):
        if self.t_learner:
            return predict_torch_logistic_proba(self.model0 if a==0 else self.model1, X)
        return predict_torch_logistic_proba(self.model, _prepend_a(a, X))

    def density(self, a, X, m):
        p = self._proba(a, X)
        return np.where(m == 1, p, 1.0 - p)

    def sample(self, a, X, n_draws=50, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        p   = self._proba(a, X)
        return rng.binomial(1, p[:, None], size=(len(X), n_draws))

    def eta_exact(self, a, a_prime, X, mu):
        """
        Exact marginalisation over M ~ Bernoulli(p_{a'}(X)):

            eta_{a,a'}(t, X) = p_{a'}(X) * mu(t, a, M=1, X)
                             + (1 - p_{a'}(X)) * mu(t, a, M=0, X)

        Returns shape (T, n).  Replaces MC integration for binary M;
        exact, faster, and zero MC variance.

        Parameters
        ----------
        a       : int  -- treatment arm for the outcome model
        a_prime : int  -- treatment arm whose mediator distribution is used
        X       : (n, p)
        mu      : MuModels instance
        """
        p   = self._proba(a_prime, X)                 # (n,)
        m1  = mu.predict(a, np.ones(len(X)),  X)      # (T, n)
        m0  = mu.predict(a, np.zeros(len(X)), X)      # (T, n)
        return p[None, :] * m1 + (1.0 - p[None, :]) * m0  # (T, n)


# ──────────────────────────────────────────────────────────────────────────────
# *** OPTIMISED *** MuModels — single multi-output model per arm
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MuModels:
    """
    Models mu(t, A, M, X) = P(Y <= t | A, M, X) for each t in Tn.

    OPTIMISED: one nn.Linear(features, T) per arm instead of T separate models.
      t_learner=False: one model on (A, M, X)  →  shape (n, T) logits.
      t_learner=True:  two models on (M, X) per arm.

    Public API is identical to the original so downstream code is unchanged.
    """
    t_learner: bool
    T:         int
    model:  Optional[nn.Module] = field(default=None)   # S-learner
    model0: Optional[nn.Module] = field(default=None)   # T-learner, A=0
    model1: Optional[nn.Module] = field(default=None)   # T-learner, A=1

    @classmethod
    def fit(cls, X, A, M, Y, Tn, epochs=10, t_learner=False):
        T = len(Tn)
        if t_learner:
            MX0 = np.column_stack([M[A==0], X[A==0]])
            MX1 = np.column_stack([M[A==1], X[A==1]])
            # Target matrices: (n_arm, T)
            Y0_mat = np.column_stack([(Y[A==0] <= t).astype(float) for t in Tn])
            Y1_mat = np.column_stack([(Y[A==1] <= t).astype(float) for t in Tn])
            m0 = fit_torch_logistic_multi(MX0, Y0_mat, epochs=epochs)
            m1 = fit_torch_logistic_multi(MX1, Y1_mat, epochs=epochs)
            return cls(t_learner=True, T=T, model0=m0, model1=m1)
        else:
            AMX   = np.column_stack([A, M, X])
            Y_mat = np.column_stack([(Y <= t).astype(float) for t in Tn])
            m     = fit_torch_logistic_multi(AMX, Y_mat, epochs=epochs)
            return cls(t_learner=False, T=T, model=m)

    def predict(self, a, M, X):
        """Returns (T, n) array of P(Y <= t | A=a, M, X) for each threshold."""
        n = X.shape[0]
        if self.t_learner:
            clf = self.model0 if a == 0 else self.model1
            return predict_torch_logistic_proba_multi(clf, np.column_stack([M, X]))
        else:
            return predict_torch_logistic_proba_multi(
                self.model, np.column_stack([np.full(n, a), M, X]))

    def predict_batched(self, a, M_flat, X_rep):
        """
        Batched prediction for MC integration.

        Parameters
        ----------
        M_flat : (n * n_mc,)   mediator draws, flattened
        X_rep  : (n * n_mc, p) covariates, repeated

        Returns
        -------
        (T, n * n_mc)  probabilities — stays in numpy, one GPU call.
        """
        return self.predict(a, M_flat, X_rep)

# Alternative regression for outcome Y
@dataclass
class MuRegressionModel:
    """
    Models E[Y | A, M, X] via linear regression.
    Used in the plugin estimator: CDF is derived by thresholding predicted Y samples.
    """
    t_learner: bool
    model:  Optional[nn.Module] = field(default=None)
    model0: Optional[nn.Module] = field(default=None)
    model1: Optional[nn.Module] = field(default=None)

    @classmethod
    def fit(cls, X, A, M, Y, epochs=10, t_learner=False):
        if t_learner:
            m0 = fit_torch_linear(np.column_stack([M[A==0], X[A==0]]), Y[A==0], epochs=epochs)
            m1 = fit_torch_linear(np.column_stack([M[A==1], X[A==1]]), Y[A==1], epochs=epochs)
            return cls(t_learner=True, model0=m0, model1=m1)
        else:
            m = fit_torch_linear(np.column_stack([A, M, X]), Y, epochs=epochs)
            return cls(t_learner=False, model=m)

    def predict_y(self, a, M, X):
        """Returns (n,) predicted E[Y | A=a, M, X]."""
        n = X.shape[0]
        if self.t_learner:
            clf = self.model0 if a == 0 else self.model1
            return predict_torch_linear(clf, np.column_stack([M, X]))
        else:
            return predict_torch_linear(self.model, np.column_stack([np.full(n, a), M, X]))


# ──────────────────────────────────────────────────────────────────────────────
# Factories (unchanged interface)
# ──────────────────────────────────────────────────────────────────────────────

def fit_propensity(X, A, epochs=10):
    return PropensityModel.fit(X, A, epochs=epochs)

def fit_mediator(X, A, M, epochs=10, t_learner=False):
    cls = BernoulliMModel if is_binary(M) else GaussianMModel
    return cls.fit(X, A, M, epochs=epochs, t_learner=t_learner)

def fit_mu(X, A, M, Y, Tn, epochs=10, t_learner=False):
    return MuModels.fit(X, A, M, Y, Tn, epochs=epochs, t_learner=t_learner)


# ──────────────────────────────────────────────────────────────────────────────
# Nuisance estimation (unchanged interface)
# ──────────────────────────────────────────────────────────────────────────────
from crps_mu_models import fit_mu_crps
def estimate_nuisances(X_tr, A_tr, M_tr, Y_tr, Tn, epochs=10, weight_decay=1e-4,
                       hidden=64, n_layers=2, lr=1e-3, batch_size=256,
                       t_learner=False):
    pi = fit_propensity(X_tr, A_tr, epochs=epochs)
    g  = fit_mediator(X_tr, A_tr, M_tr, epochs=epochs, t_learner=t_learner)
    mu = fit_mu_crps(X_tr, A_tr, M_tr, Y_tr, Tn,
                     hidden=hidden, n_layers=n_layers, lr=lr,
                     batch_size=batch_size, weight_decay=weight_decay,
                     epochs=epochs)
    return {"pi": pi, "g": g, "mu": mu}


# ──────────────────────────────────────────────────────────────────────────────
# *** OPTIMISED *** pseudo_outcomes_from_nuisances
# ──────────────────────────────────────────────────────────────────────────────

def pseudo_outcomes_from_nuisances(X_te, A_te, M_te, Y_te, Tn, nuis,
                                   n_mc=50, rng=None):
    """
    EIF pseudo-outcomes — MC loop replaced by a single batched forward pass.

    For binary mediator (the typical case) MC is unnecessary; we keep it for
    generality (Gaussian mediator) but make it ~n_mc× faster by tiling the
    entire draw matrix into one big batch.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    T, n = len(Tn), len(Y_te)

    pi    = nuis["pi"]; g = nuis["g"]; mu = nuis["mu"]
    pi_te = pi.predict_proba(X_te)
    pi0   = np.clip(pi_te[:, 0], 1e-3, 1 - 1e-3)
    pi1   = np.clip(pi_te[:, 1], 1e-3, 1 - 1e-3)

    # Outcome CDFs at observed (M, X) — one call, shape (T, n)
    mu0_te = mu.predict(0, M_te, X_te)   # (T, n)
    mu1_te = mu.predict(1, M_te, X_te)   # (T, n)

    # ── eta integration: exact for binary M, batched MC for continuous M ──────
    if isinstance(g, BernoulliMModel):
        # Exact: two mu.predict calls per (a, a') pair, no sampling
        eta00 = g.eta_exact(0, 0, X_te, mu)
        eta10 = g.eta_exact(1, 0, X_te, mu)
        eta11 = g.eta_exact(1, 1, X_te, mu)
    else:
        def eta_mc_batched(a, M_draws):
            """
            M_draws : (n, n_mc)
            Returns (T, n)  via ONE batched forward pass.
            """
            n_mc_ = M_draws.shape[1]
            M_flat = M_draws.reshape(-1, order='C')          # (n*n_mc,)
            X_rep  = np.repeat(X_te, n_mc_, axis=0)          # (n*n_mc, p)
            pred   = mu.predict_batched(a, M_flat, X_rep)    # (T, n*n_mc)
            return pred.reshape(T, n, n_mc_).mean(axis=2)    # (T, n)

        Mdraws_g0 = g.sample(0, X_te, n_draws=n_mc, rng=rng)
        Mdraws_g1 = g.sample(1, X_te, n_draws=n_mc, rng=rng)
        eta00 = eta_mc_batched(0, Mdraws_g0)
        eta10 = eta_mc_batched(1, Mdraws_g0)
        eta11 = eta_mc_batched(1, Mdraws_g1)
    # ──────────────────────────────────────────────────────────────────────

    g0_te   = np.clip(g.density(0, X_te, M_te), 1e-6, None)
    g1_te   = np.clip(g.density(1, X_te, M_te), 1e-6, None)
    A0      = (A_te == 0).astype(float)
    A1      = (A_te == 1).astype(float)
    Yt_te   = (Y_te[None, :] <= Tn[:, None]).astype(float)   # (T, n)

    phi_00  = (A0/pi0) * (Yt_te - mu0_te) + (A0/pi0) * (mu0_te - eta00) + eta00
    ratio10 = g0_te / g1_te
    phi_10  = (A1/pi1) * ratio10 * (Yt_te - mu1_te) + (A0/pi0) * (mu1_te - eta10) + eta10
    phi_11  = (A1/pi1) * (Yt_te - mu1_te) + (A1/pi1) * (mu1_te - eta11) + eta11

    return {(0,0): phi_00, (1,0): phi_10, (1,1): phi_11}


# ──────────────────────────────────────────────────────────────────────────────
# *** OPTIMISED *** crossfit_pseudo_outcomes — parallel folds
# ──────────────────────────────────────────────────────────────────────────────

def _fit_one_fold(X, A, M, Y, Tn, train_idx, test_idx,
                  epochs, n_mc, t_learner, seed,
                  hidden=64, n_layers=2, lr=1e-3, batch_size=256,
                  weight_decay=1e-4):
    """Worker function for one cross-fit fold (picklable)."""
    rng  = np.random.default_rng(seed)
    nuis = estimate_nuisances(X[train_idx], A[train_idx],
                              M[train_idx], Y[train_idx], Tn,
                              epochs=epochs, weight_decay=weight_decay,
                              hidden=hidden, n_layers=n_layers,
                              lr=lr, batch_size=batch_size,
                              t_learner=t_learner)
    phi_te = pseudo_outcomes_from_nuisances(
        X[test_idx], A[test_idx], M[test_idx], Y[test_idx], Tn,
        nuis, n_mc=n_mc, rng=rng)
    return test_idx, phi_te


def crossfit_pseudo_outcomes(X, A, M, Y, Tn, K=2, n_mc=200,
                             epochs=40, t_learner=False, rng=None,
                             n_jobs=None, random_state=0,
                             hidden=64, n_layers=2, lr=1e-3,
                             batch_size=256, weight_decay=1e-4):
    """
    K-fold cross-fitting with optional parallel fold execution.

    Returns phi and the fold splits so that crossfit_conditional_cdfs can
    reuse the *same* partition in the second stage, satisfying the double
    cross-fitting condition.

    Parameters
    ----------
    n_jobs : int or None
        Number of parallel jobs for joblib.  None → serial (safe on GPU);
        set to K for CPU-bound workloads or when using multiple GPUs.
        Note: parallel jobs share the GPU poorly — leave None unless you have
        multiple GPUs or are CPU-bound.
    n_mc : int
        Default reduced to 200 (was 800).  For binary mediators ~100 is enough.
    random_state : int
        Seed for KFold shuffle.  Must match the value passed to
        crossfit_conditional_cdfs (default 0 for both).

    Returns
    -------
    phi    : dict  -- keys (a, a'), values shape (T, n)
    splits : list of (train_idx, test_idx) arrays, length K
    """
    rng    = np.random.default_rng(0) if rng is None else rng
    pairs  = [(0,0), (1,0), (1,1)]
    phi    = {p: np.zeros((len(Tn), len(Y))) for p in pairs}
    kf     = KFold(n_splits=K, shuffle=True, random_state=random_state)
    splits = list(kf.split(X))
    seeds  = rng.integers(0, 2**31, size=K)

    fold_kwargs = dict(hidden=hidden, n_layers=n_layers, lr=lr,
                       batch_size=batch_size, weight_decay=weight_decay)
    if n_jobs is not None and n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one_fold)(
                X, A, M, Y, Tn, tr, te, epochs, n_mc, t_learner, int(seeds[i]),
                **fold_kwargs)
            for i, (tr, te) in enumerate(splits)
        )
    else:
        results = [
            _fit_one_fold(X, A, M, Y, Tn, tr, te, epochs, n_mc, t_learner, int(seeds[i]),
                          **fold_kwargs)
            for i, (tr, te) in enumerate(splits)
        ]

    for test_idx, phi_te in results:
        for p in pairs:
            phi[p][:, test_idx] = phi_te[p]

    return phi, splits


# ──────────────────────────────────────────────────────────────────────────────
# *** OPTIMISED *** fit_conditional_cdfs / predict_conditional_cdfs
# ──────────────────────────────────────────────────────────────────────────────

def enforce_cdf_monotonicity(cdf, eps=1e-6):
    """Enforce monotonicity in threshold dimension (axis 0)."""
    cdf = np.clip(cdf, eps, 1 - eps)
    cdf = np.maximum.accumulate(cdf, axis=0)
    return np.clip(cdf, eps, 1 - eps)

def crossfit_conditional_cdfs(phi, X, Tn, splits, epochs=40, eps=1e-4,
                               enforce_mono=True):
    """
    Second-stage cross-fitting: produces fully out-of-sample fitted values of
    F_{a,a'}(t | X) for bias/coverage evaluation only.

    For each fold k the second-stage model is trained on pseudo-outcomes from
    all *other* folds and predicts for fold k, using the *same* partition as
    the first stage (crossfit_pseudo_outcomes).  This completes the double
    cross-fitting condition required by Double ML.

    NOTE: fold models are not returned — each was trained on only 1 - 1/K of
    the data and is not suitable for deployment.  After validating with this
    function, call fit_conditional_cdfs(phi, X, Tn) to obtain a single
    full-sample model for out-of-sample prediction (e.g. for policy learning).

    Parameters
    ----------
    phi    : dict  -- from crossfit_pseudo_outcomes, keys (a,a'), shape (T, n)
    X      : shape (n, p)
    Tn     : shape (T,)
    splits : list of (train_idx, test_idx)  -- returned by crossfit_pseudo_outcomes
    epochs : int
    eps    : float
    enforce_mono : bool

    Returns
    -------
    cdfs : dict  -- cdfs[pair] shape (T, n), fully out-of-sample fitted values
    """
    n, T  = X.shape[0], len(Tn)
    pairs = list(phi.keys())
    cdfs  = {p: np.zeros((T, n)) for p in pairs}

    for pair, phi_pair in phi.items():
        # phi_pair : (T, n) — rows are thresholds, cols are observations
        phi_mat = phi_pair.T   # (n, T) for fit_torch_linear_multi
        for train_idx, test_idx in splits:
            model = fit_torch_linear_multi(X[train_idx], phi_mat[train_idx],
                                           epochs=epochs)
            preds = predict_torch_linear_multi(model, X[test_idx])  # (T, n_te)
            cdfs[pair][:, test_idx] = np.clip(preds, eps, 1 - eps)

        #if enforce_mono:
        #    cdfs[pair] = enforce_cdf_monotonicity(cdfs[pair], eps=eps)

    return cdfs


def fit_conditional_cdfs(phi, X, Tn, epochs=40, eps=1e-4, enforce_mono=True):
    """
    Second-stage regression: fit F_{a,a'}(t | X) on the full sample.

    This is the deployment model for out-of-sample prediction, e.g. for
    computing CATE and FNA upper bounds for policy learning.  Call this
    *after* crossfit_conditional_cdfs has validated estimation quality.

    Uses a single nn.Linear(p, T) per pair — one model scores all T thresholds
    in a single forward pass.

    Use predict_conditional_cdfs(models, X_new, Tn) to score new individuals.

    Parameters
    ----------
    phi : dict  -- from crossfit_pseudo_outcomes, keys (a,a'), shape (T, n)
    X   : shape (n, p)
    Tn  : shape (T,)

    Returns
    -------
    cdfs   : dict  -- cdfs[pair] shape (T, n), in-sample fitted values
    models : dict  -- models[pair] single nn.Linear(p, T);
                      pass to predict_conditional_cdfs for new data
    """
    cdfs, models = {}, {}
    for pair, phi_pair in phi.items():
        phi_mat = phi_pair.T                                          # (n, T)
        model   = fit_torch_linear_multi(X, phi_mat, epochs=epochs)
        preds   = predict_torch_linear_multi(model, X)               # (T, n)
        preds   = np.clip(preds, eps, 1 - eps)
        if enforce_mono:
            preds = enforce_cdf_monotonicity(preds, eps=eps)
        cdfs[pair]   = preds
        models[pair] = model
    return cdfs, models



def predict_conditional_cdfs(models, X, Tn, eps=1e-4, enforce_mono=True):
    """
    Out-of-sample prediction using models from fit_conditional_cdfs.

    Parameters
    ----------
    models : dict  -- models[pair] is a single nn.Linear(p, T) per pair
    X      : shape (n_new, p)
    Tn     : shape (T,)

    Returns
    -------
    cdfs : dict  -- cdfs[pair] shape (T, n_new)
    """
    cdfs = {}
    for pair, model in models.items():
        with torch.no_grad():
            preds = model(to_tensor(X)).cpu().numpy().T   # (T, n)
        preds = np.clip(preds, eps, 1 - eps)
        if enforce_mono:
            preds = enforce_cdf_monotonicity(preds, eps=eps)
        cdfs[pair] = preds
    return cdfs


# ──────────────────────────────────────────────────────────────────────────────
# *** OPTIMISED *** fit_plugin_cdfs / predict_plugin_cdfs
# ──────────────────────────────────────────────────────────────────────────────

def fit_plugin_cdfs(X, A, M, Y, Tn, n_mc=200, epochs=10, weight_decay=1e-4,
                    t_learner=False, rng=None, hidden=64, n_layers=2,
                    lr=1e-3, batch_size=256, enforce_mono=True, eps=1e-4):
    """
    Plug-in (g-formula) estimator.

    Uses exact marginalisation for binary M (BernoulliMModel) and
    batched MC integration for continuous M (GaussianMModel).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n, T = len(Y), len(Tn)

    g  = fit_mediator(X, A, M, epochs=epochs, t_learner=t_learner)
    mu = fit_mu_crps(X, A, M, Y, Tn, hidden=hidden, n_layers=n_layers,
                     lr=lr, batch_size=batch_size, weight_decay=weight_decay,
                     epochs=epochs)
    if isinstance(g, BernoulliMModel):
        def eta(a, a_prime):
            return g.eta_exact(a, a_prime, X, mu)
    else:
        def eta(a, a_prime):
            M_draws = g.sample(a_prime, X, n_draws=n_mc, rng=rng)
            M_flat  = M_draws.reshape(-1, order='C')
            X_rep   = np.repeat(X, n_mc, axis=0)
            pred    = mu.predict_batched(a, M_flat, X_rep)
            return pred.reshape(T, n, n_mc).mean(axis=2)

    cdfs = {(0,0): eta(0,0), (1,0): eta(1,0), (1,1): eta(1,1)}

    if enforce_mono:
      cdfs = {pair: enforce_cdf_monotonicity(arr, eps=eps)
              for pair, arr in cdfs.items()}

    return cdfs#, {"g": g, "mu": mu, "n_mc": n_mc}


def predict_plugin_cdfs(models, X, Tn, n_mc=None, rng=None):
    rng  = np.random.default_rng(0) if rng is None else rng
    g    = models["g"]; mu = models["mu"]
    n_mc = models["n_mc"] if n_mc is None else n_mc
    n, T = X.shape[0], len(Tn)

    if isinstance(g, BernoulliMModel):
        def eta(a, a_prime):
            return g.eta_exact(a, a_prime, X, mu)
    else:
        def eta(a, a_prime):
            M_draws = g.sample(a_prime, X, n_draws=n_mc, rng=rng)
            M_flat  = M_draws.reshape(-1, order='C')
            X_rep   = np.repeat(X, n_mc, axis=0)
            pred    = mu.predict_batched(a, M_flat, X_rep)
            return pred.reshape(T, n, n_mc).mean(axis=2)

    return {(0,0): eta(0,0), (1,0): eta(1,0), (1,1): eta(1,1)}


# ──────────────────────────────────────────────────────────────────────────────
# Policy training helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def makarov_bounds_per_x(delta):
    L = np.maximum(delta.max(axis=0), 0.0)
    U = 1.0 - np.maximum((-delta).max(axis=0), 0.0)
    return L, U

def compute_bounds(cdfs, Tn):
    F00, F10, F11 = cdfs[(0,0)], cdfs[(1,0)], cdfs[(1,1)]
    _, U_dir = makarov_bounds_per_x(F10 - F00)
    _, U_ind = makarov_bounds_per_x(F11 - F10)
    _, U_tot = makarov_bounds_per_x(F11 - F00)
    return {"U_dir": U_dir, "U_ind": U_ind, "U_tot": U_tot}






# ──────────────────────────────────────────────────────────────────────────────
# Oracle pseudo-outcomes (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def oracle_pseudo_outcomes(Y, A, Tn, pi0, pi1, g0, g1,
                           mu0_obs, mu1_obs, eta00, eta10, eta11):
    A0  = (A == 0).astype(float); A1 = (A == 1).astype(float)
    Yt  = (Y[None,:] <= Tn[:,None]).astype(float)
    return {
        (0,0): (A0/pi0)*(Yt-mu0_obs) + (A0/pi0)*(mu0_obs-eta00) + eta00,
        (1,0): (A1/pi1)*(g0/g1)*(Yt-mu1_obs) + (A0/pi0)*(mu1_obs-eta10) + eta10,
        (1,1): (A1/pi1)*(Yt-mu1_obs) + (A1/pi1)*(mu1_obs-eta11) + eta11,
    }

def rmse_phi(phi_hat, phi_oracle):
    return {p: float(np.sqrt(np.mean((phi_hat[p]-phi_oracle[p])**2)))
            for p in phi_hat}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation / plotting helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def empirical_cdf(y, Tn):          # redefined above; keep for compat
    return np.mean(y[:,None] <= Tn[None,:], axis=0)

def makarov_bounds_from_cdfs(Fu, Fv):
    delta = Fu - Fv
    L = max(delta.max(), 0.0)
    U = 1.0 - max((-delta).max(), 0.0)
    return L, U

def makarov_bounds_from_ccdf_plugin(Fu_x,Fv_x):
    n = Fu_x.shape[1]
    delta_hat = Fu_x-Fv_x
    gamma_L = np.clip(np.maximum(delta_hat.max(axis=0), 0), 0, 1)
    gamma_U = np.clip(1 + np.minimum(delta_hat.min(axis=0), 0), 0, 1)
    L = np.mean(gamma_L)
    U = np.mean(gamma_U)

    se_L = np.sqrt(np.var(gamma_L, ddof=1) / n)
    se_U = np.sqrt(np.var(gamma_U, ddof=1) / n)

    #return {"L":L, "U":U, 
    #        "se_L":se_L, "se_U": se_U}
    return L, U, se_L, se_U

def makarov_bounds_from_ccdf_dr(delta, gamma_delta):
    """One-step DR Makarov bounds from (T, n) arrays.

    The DR estimator is:
        theta_L = mean_i [ 1{s_L(i)>0} * Gamma_Delta(t_L(i), i) ]
        theta_U = mean_i [ 1 + 1{s_U(i)>0} * Gamma_Delta(t_U(i), i) ]

    where t_L(i) = argmax_t delta_i(t)  and  t_U(i) = argmax_t -delta_i(t)
    are the per-observation conditional maximisers.

    SE is computed from the estimated influence functions:
        phi_L(i) = 1{s_L(i)>0} * Gamma_Delta(t_L(i), i) - theta_L
        phi_U(i) = 1 + 1{s_U(i)>0} * Gamma_Delta(t_U(i), i) - theta_U
        se = sqrt( Var_n(phi) / n )

    Returns (theta_L, theta_U, se_L, se_U).
    """
    n = delta.shape[1]
    j = np.arange(n)

    s_L   = delta.max(axis=0);      idx_L = delta.argmax(axis=0)
    s_U   = (-delta).max(axis=0);   idx_U = (-delta).argmax(axis=0)

    G_L = gamma_delta[idx_L, j]    # Gamma_Delta evaluated at per-obs argmax
    G_U = gamma_delta[idx_U, j]

    theta_L = np.clip(float(np.mean((s_L > 0) * G_L)),0,1)
    theta_U = np.clip(float(np.mean(1.0 + (s_U > 0) * G_U)),0, 1)

    phi_L = (s_L > 0) * G_L - theta_L
    phi_U = 1.0 + (s_U > 0) * G_U - theta_U

    se_L = float(np.sqrt(np.var(phi_L, ddof=1) / n))
    se_U = float(np.sqrt(np.var(phi_U, ddof=1) / n))

    #return {"L": theta_L, "U":theta_U, 
    #        "se_L":se_L, "se_U": se_U}
    return theta_L, theta_U, se_L, se_U



def bounds_from_cdf_triplet(F00, F10, F11):
    dir_L, dir_U = makarov_bounds_from_cdfs(F10, F00)
    ind_L, ind_U = makarov_bounds_from_cdfs(F11, F10)
    tot_L, tot_U = makarov_bounds_from_cdfs(F11, F00)

    return {"Direct":   makarov_bounds_from_cdfs(F10, F00),
            "Indirect": makarov_bounds_from_cdfs(F11, F10),
            "Total":    makarov_bounds_from_cdfs(F11, F00)}

def tighter_bounds_from_ccdf_triplet_plugin(F00_x, F10_x, F11_x):
    return {"Direct":   makarov_bounds_from_ccdf_plugin(F10_x, F00_x),
            "Indirect": makarov_bounds_from_ccdf_plugin(F11_x, F10_x),
            "Total":    makarov_bounds_from_ccdf_plugin(F11_x, F00_x)}

def tighter_bounds_from_ccdf_triplet_dr(F00_x, F10_x, F11_x,phi):
    phi00 = phi[(0,0)]
    phi10 = phi[(1,0)]
    phi11 = phi[(1,1)]
    return {"Direct":   makarov_bounds_from_ccdf_dr(F10_x-F00_x, phi10-phi00),
            "Indirect": makarov_bounds_from_ccdf_dr(F11_x-F10_x, phi11-phi10),
            "Total":    makarov_bounds_from_ccdf_dr(F11_x-F00_x,phi11-phi00)}

def oracle_pi_a(dat, a):
    return dat["pA"] if a==1 else 1.0-dat["pA"]

def oracle_g_a(dat, a, m_obs):
    pMa = dat["pM1"] if a==1 else dat["pM0"]
    return np.where(m_obs==1, pMa, 1.0-pMa)

def oracle_mu_a_tmx(dat, a, t, m):
    mean_amx = dat["bX"] + dat["d"]*a + dat["q"]*m + dat["c"]*a*m
    # BUG FIX: DGP has heteroskedastic noise in m.
    # Var(Y | a, m=1, X) = sigma_y^2 + sigma_m_noise^2 (both noise terms active)
    # Var(Y | a, m=0, X) = sigma_y^2                   (sigma_m_noise * 0 = 0)
    sigma_m = dat.get("sigma_m_noise", 0.0)
    sigma   = np.sqrt(dat["sigma_y"]**2 + (m * sigma_m)**2)
    return norm.cdf((t - mean_amx) / sigma)


def oracle_cond_cdfs(dat, Tn):
    """
    Oracle conditional CDFs F_{a,a'}(t | X_i) for each individual i.

    Uses the true DGP parameters (bX, d, q, c, pM0, pM1, sigma_y,
    sigma_m_noise) to compute F_{aa'}(t|X_i) analytically:

        F_{aa'}(t | X_i) = sum_{m in {0,1}} P(M_{a'} = m | X_i)
                           * Phi( (t - mu(a, m, X_i)) / sigma(m) )

    where
        mu(a, m, X_i) = bX_i + d_i*a + q_i*m + c*a*m
        sigma(m=0)    = sigma_y
        sigma(m=1)    = sqrt(sigma_y^2 + sigma_m_noise^2)

    Returns
    -------
    dict  keyed by (a, a'), each value shape (T, n)
    """
    sigma_y = dat["sigma_y"]
    sigma_m = dat.get("sigma_m_noise", 0.0)
    sigma0  = sigma_y
    sigma1  = np.sqrt(sigma_y**2 + sigma_m**2)

    bX   = dat["bX"]; d = dat["d"]; q = dat["q"]; c = dat["c"]
    pM0  = dat["pM0"];  pM1 = dat["pM1"]

    t = Tn[:, None]                          # (T, 1)

    def _F(a, ap):
        pMa = pM1 if ap == 1 else pM0        # (n,)  mediator dist for a'
        mu_m0 = bX + d*a                     # (n,)  q*0 + c*a*0 = 0
        mu_m1 = bX + d*a + q + c*a           # (n,)  q*1 + c*a*1
        F_m0  = norm.cdf((t - mu_m0[None, :]) / sigma0)   # (T, n)
        F_m1  = norm.cdf((t - mu_m1[None, :]) / sigma1)   # (T, n)
        return (1.0 - pMa)[None, :] * F_m0 + pMa[None, :] * F_m1

    return {(0, 0): _F(0, 0),
            (1, 0): _F(1, 0),
            (1, 1): _F(1, 1)}


def oracle_covariate_assisted_bounds(oracle_ccdfs, Tn):
    """
    Oracle covariate-assisted Makarov bounds.

    The estimands targeted by the DR / plugin estimators are:

        theta_L = E_X[ max_t  [F_u(t|X) - F_v(t|X)] ]   (averaged lower bound)
        theta_U = E_X[ 1 + min_t [F_u(t|X) - F_v(t|X)] ] (averaged upper bound)

    These are strictly tighter than the marginal Makarov bounds (which operate
    on F_u(t) = E_X[F_u(t|X)] and Jensen's inequality goes the wrong way).

    This function computes them using the *true* DGP conditional CDFs, giving
    the correct oracle comparator for coverage checks.

    Returns
    -------
    dict  keyed by "Direct" / "Indirect" / "Total", each value (L, U, se_L, se_U)
        where se is the oracle Monte-Carlo SE = sqrt(Var_n(gamma) / n).
    """
    #cond = oracle_cond_cdfs(dat, Tn)   # {(a,a'): (T, n)}
    #n    = cond[(0, 0)].shape[1]

    def _bounds(Fu_x, Fv_x):
        n = Fu_x.shape[1]
        delta   = Fu_x - Fv_x                          # (T, n)
        gamma_L = np.maximum(delta.max(axis=0), 0.0)   # (n,)
        gamma_U = 1.0 + np.minimum(delta.min(axis=0), 0.0)
        L    = float(np.mean(gamma_L))
        U    = float(np.mean(gamma_U))
        se_L = float(np.sqrt(np.var(gamma_L, ddof=1) / n))
        se_U = float(np.sqrt(np.var(gamma_U, ddof=1) / n))
        return L, U, se_L, se_U

    return {
        "Direct":   _bounds(oracle_ccdfs[(1, 0)], oracle_ccdfs[(0, 0)]),
        "Indirect": _bounds(oracle_ccdfs[(1, 1)], oracle_ccdfs[(1, 0)]),
        "Total":    _bounds(oracle_ccdfs[(1, 1)], oracle_ccdfs[(0, 0)]),
    }

def oracle_eta_aa_prime_t(dat, a, a_prime, t):
    pM_ap = dat["pM1"] if a_prime==1 else dat["pM0"]
    return (1.0-pM_ap)*oracle_mu_a_tmx(dat,a,t,m=0) + pM_ap*oracle_mu_a_tmx(dat,a,t,m=1)

def oracle_eif_F_aa_prime_t(dat, a, a_prime, t):
    A, M, Y = dat["A"], dat["M"], dat["Y"]
    pi_a  = oracle_pi_a(dat, a);        pi_ap = oracle_pi_a(dat, a_prime)
    g_a   = oracle_g_a(dat, a, M);      g_ap  = oracle_g_a(dat, a_prime, M)
    mu_obsM = oracle_mu_a_tmx(dat, a, t, M)
    eta     = oracle_eta_aa_prime_t(dat, a, a_prime, t)
    F_true  = float(np.mean(dat[f"Y{a}{a_prime}"] <= t))
    Yt      = (Y <= t).astype(float)
    term1   = (A==a).astype(float)  / pi_a  * (g_ap/g_a) * (Yt - mu_obsM)
    term2   = (A==a_prime).astype(float) / pi_ap * (mu_obsM - eta)
    D = term1 + term2 + eta
    return D - F_true, D

def oracle_marginal_cdfs(dat, Tn):
    return {(0,0): empirical_cdf(dat["Y00"],Tn),
            (1,0): empirical_cdf(dat["Y10"],Tn),
            (1,1): empirical_cdf(dat["Y11"],Tn)}

def oracle_true_fna(dat):
    return {"Direct":   float(np.mean(dat["Y10"] < dat["Y00"])),
            "Indirect": float(np.mean(dat["Y11"] < dat["Y10"])),
            "Total":    float(np.mean(dat["Y11"] < dat["Y00"]))}

def oracle_subgroup_cdfs(dat, Tn, x1_col=0):
    x1 = dat["X"][:,x1_col]; masks = {"X1>0": x1>0, "X1<=0": x1<=0}
    return {lbl: {"n": int(mask.sum()),
                  (0,0): empirical_cdf(dat["Y00"][mask],Tn),
                  (1,0): empirical_cdf(dat["Y10"][mask],Tn),
                  (1,1): empirical_cdf(dat["Y11"][mask],Tn)}
            for lbl, mask in masks.items()}

def oracle_subgroup_fna(dat, x1_col=0):
    x1 = dat["X"][:,x1_col]; masks = {"X1>0": x1>0, "X1<=0": x1<=0}
    return {lbl: {"Direct":   float(np.mean(dat["Y10"][mask]<dat["Y00"][mask])),
                  "Indirect": float(np.mean(dat["Y11"][mask]<dat["Y10"][mask])),
                  "Total":    float(np.mean(dat["Y11"][mask]<dat["Y00"][mask]))}
            for lbl, mask in masks.items()}

def oracle_objects(dat, Tn, x1_col=0):
    marginal = oracle_marginal_cdfs(dat, Tn)
    true_fna = oracle_true_fna(dat)
    bounds_marginal = bounds_from_cdf_triplet(marginal[(0,0)], marginal[(1,0)], marginal[(1,1)])
    # Covariate-assisted oracle bounds — correct comparator for DR / plugin estimators,
    # which average per-individual conditional Makarov bounds over X.
    bounds_cov_assisted = oracle_covariate_assisted_bounds(dat, Tn)
    return {"marginal":        marginal,
            "bounds":          bounds_marginal,      # marginal Makarov (for reference)
            "bounds_oracle_x": bounds_cov_assisted,  # comparable to DR / plugin
            "true_fna":        true_fna}

def estimated_marginal_cdfs(cdf_hat, from_cond=False):
    return {pair: arr.mean(axis=1) for pair, arr in cdf_hat.items()}

def estimated_subgroup_cdfs(cdf_hat, X, x1_col=0):
    x1 = X[:,x1_col]; masks = {"X1>0": x1>0, "X1<=0": x1<=0}
    return {lbl: {"n": int(mask.sum()),
                  **{pair: arr[:,mask].mean(axis=1) for pair,arr in cdf_hat.items()}}
            for lbl, mask in masks.items()}

def estimated_objects(cdf_hat, X, Tn,  is_dr=False, phi = [], x1_col=0):
    if is_dr == True:
        #marginal  = estimated_marginal_cdfs(phi)
        bm = tighter_bounds_from_ccdf_triplet_dr(cdf_hat[(0,0)],cdf_hat[(1,0)],cdf_hat[(1,1)],phi)
    else :
        #marginal  = estimated_marginal_cdfs(cdf_hat)
        bm = tighter_bounds_from_ccdf_triplet_plugin(cdf_hat[(0,0)],cdf_hat[(1,0)],cdf_hat[(1,1)])
    #marginal  = estimated_marginal_cdfs(phi if is_dr else cdf_hat)
    #avg_cond  = estimated_subgroup_cdfs(cdf_hat, X, x1_col=x1_col)
    #bm = bounds_from_cdf_triplet(marginal[(0,0)],marginal[(1,0)],marginal[(1,1)])

    return bm


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

_PAIR_LABELS = {(0,0): r"$F_{00}$", (1,0): r"$F_{10}$", (1,1): r"$F_{11}$"}
_PAIRS       = [(0,0),(1,0),(1,1)]
_EFFECTS     = ["Direct","Indirect","Total"]
_GROUPS      = ["X1>0","X1<=0"]

def plot_cdfs(Tn, oracle_obj, dr_obj, plugin_obj, scope="marginal", figsize=None):
    key   = "marginal" if scope=="marginal" else scope
    oc    = oracle_obj["marginal"] if scope=="marginal" else oracle_obj["subgroup"][scope]
    dc    = dr_obj["marginal"]     if scope=="marginal" else dr_obj["subgroup"][scope]
    pc    = plugin_obj["marginal"] if scope=="marginal" else plugin_obj["subgroup"][scope]
    title = "Marginal" if scope=="marginal" else scope
    fig, axes = plt.subplots(1,3,figsize=figsize or (13,4),sharey=True)
    for ax, pair in zip(axes, _PAIRS):
        ax.plot(Tn, oc[pair], color="black",      lw=2,   label="Oracle")
        ax.plot(Tn, dc[pair], color="tab:blue",   lw=1.5, ls="--", label="DR")
        ax.plot(Tn, pc[pair], color="tab:orange", lw=1.5, ls=":",  label="Plug-in")
        ax.set_title(f"{title}  {_PAIR_LABELS[pair]}"); ax.set_xlabel("t")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    axes[0].set_ylabel("P(Y ≤ t)"); plt.tight_layout(); return fig

def plot_bounds(Tn, oracle_obj, dr_obj, plugin_obj, scope="marginal", figsize=None):
    ob = oracle_obj["bounds"][scope]; db = dr_obj["bounds"][scope]
    pb = plugin_obj["bounds"][scope]; tf = oracle_obj["true_fna"][scope]
    method_info = [("Oracle",ob,2,"black"),("DR",db,1,"tab:blue"),("Plug-in",pb,0,"tab:orange")]
    fig, axes = plt.subplots(1,3,figsize=figsize or (12,3),sharey=True)
    for ax, effect in zip(axes, _EFFECTS):
        for lbl, bdict, ypos, color in method_info:
            L, U = bdict[effect]
            ax.hlines(ypos,L,U,lw=3,color=color,label=lbl if effect==_EFFECTS[0] else None)
            ax.vlines([L,U],ypos-.12,ypos+.12,lw=1.5,color=color)
        tv = tf[effect]
        ax.axvline(tv,ls="--",color="grey",alpha=.7,label="True FNA" if effect==_EFFECTS[0] else None)
        ax.scatter(tv,1,marker="D",s=55,color="tab:red",zorder=5)
        ax.set_title(f"{effect}  ({scope})"); ax.set_yticks([2,1,0])
        ax.set_yticklabels(["Oracle","DR","Plug-in"]); ax.set_xlim(0,1)
        ax.grid(axis="x",alpha=.3); ax.set_xlabel("FNA / Bound value")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen: seen[l] = h
    fig.legend(seen.values(),seen.keys(),loc="upper center",ncol=len(seen),frameon=True)
    plt.tight_layout(rect=[0,0,1,.88]); return fig

def save_fig(fig, name):
    fig.savefig(f"outputs/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"outputs/{name}.pdf", bbox_inches="tight")

def bin_by_x1(x1, n_bins=10):
    qs   = np.quantile(x1, np.linspace(0,1,n_bins+1))
    bins = np.digitize(x1, qs[1:-1], right=True)
    return bins, qs

def mean_from_cdf_grid(T, F):
    return np.trapz(1.0 - F, T, axis=0)
