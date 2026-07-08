"""
gp_vol_engine.py — Gaussian Process Volatility Forecasting Engine
========================================================================

Theory
------
**Gaussian Process regression.** A GP places a distribution over
functions, fully specified by a mean function (taken as zero here after
centering) and a covariance/kernel function k(t,t'). Given training data
(t_i, y_i), the posterior at a new point t* is exactly:

    mu*    = k*^T (K + sigma_n^2 I)^-1 y
    sigma*^2 = k(t*,t*) - k*^T (K + sigma_n^2 I)^-1 k*

This is EXACT Bayesian inference (closed-form), not an approximation or a
trained neural network — the only "learning" is fitting 3 scalar
hyperparameters by maximum likelihood.

**Matern 5/2 kernel** (closed form, no Bessel functions needed):

    k(r) = sigma_f^2 * (1 + sqrt5*r/l + 5r^2/(3l^2)) * exp(-sqrt5*r/l)

where r = |t - t'|. This is the standard practical Matern choice: genuinely
a Matern kernel (not RBF), but tractable without special functions.

**Hyperparameter fitting via exact marginal likelihood gradient.** The
negative log marginal likelihood:

    NLL = 0.5 y^T alpha + 0.5 log|K_y| + (N/2) log(2*pi),   alpha = K_y^-1 y

has an EXACT gradient (Rasmussen & Williams 2006, eq. 5.9):

    dNLL/dtheta = -0.5 * ( alpha^T (dK_y/dtheta) alpha - tr(K_y^-1 dK_y/dtheta) )

This is closed-form matrix calculus specific to the GP marginal likelihood
— not backprop through a computation graph. Optimized via Adam on the
log-hyperparameters (kept positive by construction).

**Leave-one-out (LOO) cross-validation, in closed form.** Given the fitted
Cholesky factors, every training point's leave-one-out predictive mean and
variance is available WITHOUT refitting N times (Rasmussen & Williams
2006, eq. 5.12-5.13):

    mu_LOO_i  = y_i - alpha_i / [K_y^-1]_ii
    var_LOO_i = 1 / [K_y^-1]_ii

This is what makes "today's vol is anomalous relative to the GP" a
genuinely out-of-sample-style statement, not a trivial in-sample residual
(which would be near-zero by construction since the GP interpolates its
own training points closely).

**Core signal.** ETFs where CURRENT realized vol sits in the tail of the
GP's LOO-implied posterior are flagged anomalous. The posterior WIDTH
itself (how uncertain the GP's forward vol forecast is) is treated as a
separate regime-stability signal.

**Score construction**

    score = 0.50*(-anomaly_z) + 0.25*(-regime_width)*sign(-anomaly_z) + 0.25*fit_quality

| Component     | Meaning                                                                  |
|-----------------|------------------------------------------------------------------------------|
| anomaly_z       | Standardized LOO residual at the most recent point. Negative (calmer than expected) is treated as favorable — a risk-preference ranking convention (standard in low-volatility investing), not a specific return-direction causal claim. |
| regime_width    | Average forecast posterior std over the horizon, normalized by historical vol level — the posterior's own width as a regime-stability signal. |
| fit_quality     | LOO-CV R^2-style diagnostic: does the fitted GP genuinely explain this ticker's historical vol dynamics? |

**Distinct from GPLVM-ANOMALY** elsewhere in this suite: that engine uses
a GP for LATENT VARIABLE modelling (unsupervised representation learning).
This engine uses a GP directly as a TIME SERIES REGRESSION model on
realized volatility — a different application of the same underlying
machinery.

References
----------
- Rasmussen, C. & Williams, C. (2006). Gaussian Processes for Machine
  Learning. MIT Press. (Chapters 2, 5.)
- Matern, B. (1960). Spatial Variation. Springer Lecture Notes in
  Statistics.
"""

import numpy as np
import pandas as pd
from typing import List

import config


# ── Matern 5/2 kernel and its exact gradients ─────────────────────────────────

def matern52_and_grads(r: np.ndarray, ell: float, sf2: float):
    """
    r: distances (any shape). Returns (K, dK_dlogell, dK_dlogsf2), same shape as r.
    Gradients are w.r.t. LOG-hyperparameters (log ell, log sf2), so that
    optimization keeps ell, sf2 positive by construction.
    """
    u = r / ell
    sqrt5 = np.sqrt(5.0)
    su = sqrt5 * u
    exp_term = np.exp(-su)

    K = sf2 * (1.0 + su + (5.0 * u ** 2) / 3.0) * exp_term
    dK_dlogsf2 = K.copy()                                    # dK/d(sf2) * sf2 = K
    dK_dlogell = sf2 * (5.0 / 3.0) * (u ** 2) * (1.0 + su) * exp_term
    return K, dK_dlogell, dK_dlogsf2


class GPVolModel:
    """GP regression on a realized-vol time series with a Matern 5/2 kernel,
    fit by exact marginal-likelihood gradient (Adam on log-hyperparameters)."""

    def __init__(self, t: np.ndarray, y: np.ndarray):
        self.t = t.astype(np.float64)
        self.y = y.astype(np.float64)
        self.y_mean = float(y.mean())
        self.y_centered = self.y - self.y_mean

        y_var = float(np.var(y)) + 1e-8
        self.log_ell = np.log(max(len(t) / 10.0, 1.0))
        self.log_sf2 = np.log(y_var)
        self.log_sn2 = np.log(0.1 * y_var + 1e-8)

    def _abs_dists(self, t1, t2):
        return np.abs(t1[:, None] - t2[None, :])

    def _nll_and_grad(self):
        ell = np.exp(self.log_ell)
        sf2 = np.exp(self.log_sf2)
        sn2 = np.exp(self.log_sn2)
        N = len(self.t)

        r = self._abs_dists(self.t, self.t)
        K, dK_dlogell, dK_dlogsf2 = matern52_and_grads(r, ell, sf2)
        Ky = K + (sn2 + config.JITTER) * np.eye(N)

        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_centered))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * self.y_centered @ alpha + 0.5 * logdet + 0.5 * N * np.log(2 * np.pi)

        Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))
        dK_dlogsn2 = sn2 * np.eye(N)

        def grad_term(dK):
            return -0.5 * (alpha @ dK @ alpha - np.sum(Kinv * dK))

        grads = np.array([
            grad_term(dK_dlogell),
            grad_term(dK_dlogsf2),
            grad_term(dK_dlogsn2),
        ])
        return float(nll), grads, Kinv, alpha, L

    def fit(self, epochs: int = None, lr: float = None):
        epochs = epochs or config.GP_EPOCHS
        lr = lr or config.GP_LR
        theta = np.array([self.log_ell, self.log_sf2, self.log_sn2])
        m, v = np.zeros(3), np.zeros(3)
        b1, b2, eps = 0.9, 0.999, 1e-8

        last_nll = None
        for step in range(1, epochs + 1):
            self.log_ell, self.log_sf2, self.log_sn2 = theta
            nll, grads, Kinv, alpha, L = self._nll_and_grad()
            last_nll = nll

            m = b1 * m + (1 - b1) * grads
            v = b2 * v + (1 - b2) * grads ** 2
            mh = m / (1 - b1 ** step)
            vh = v / (1 - b2 ** step)
            theta = theta - lr * mh / (np.sqrt(vh) + eps)

        self.log_ell, self.log_sf2, self.log_sn2 = theta
        _, _, self.Kinv, self.alpha, self.L = self._nll_and_grad()
        return last_nll

    def loo_predictive(self):
        """Exact closed-form leave-one-out mean/variance for every training point."""
        Kinv_diag = np.diag(self.Kinv)
        mu_loo = self.y_centered - self.alpha / Kinv_diag
        var_loo = 1.0 / Kinv_diag
        return mu_loo + self.y_mean, var_loo

    def predict(self, t_star: np.ndarray):
        """GP posterior mean/variance at new time indices t_star."""
        ell, sf2 = np.exp(self.log_ell), np.exp(self.log_sf2)
        r_star = self._abs_dists(t_star, self.t)
        k_star, _, _ = matern52_and_grads(r_star, ell, sf2)

        mean_star = k_star @ self.alpha + self.y_mean

        v = np.linalg.solve(self.L, k_star.T)
        k_ss = sf2   # matern52 at r=0
        var_star = k_ss - np.sum(v ** 2, axis=0)
        var_star = np.clip(var_star, 1e-10, None)
        return mean_star, var_star


# ── Per-ticker forecast + diagnostics ───────────────────────────────────────────

def compute_realized_vol(log_ret: np.ndarray, rv_window: int) -> np.ndarray:
    """Rolling realized vol (std of returns) over rv_window days, one value per
    valid endpoint. Returns array of length len(log_ret) - rv_window + 1."""
    n = len(log_ret) - rv_window + 1
    rv = np.zeros(n)
    for i in range(n):
        rv[i] = np.std(log_ret[i:i + rv_window])
    return rv


def forecast_and_diagnose(prices: pd.DataFrame, ticker: str, window: int):
    """
    Fit a GP to one ticker's realized-vol series using only data present in
    `prices`, and return {anomaly_z, regime_width, fit_quality} for the
    state as of the LAST row of `prices`. Returns None on failure.
    """
    H, rv_w = config.PRED_HORIZON, config.RV_WINDOW

    ps = prices[ticker].dropna()
    min_needed = window + rv_w + 10
    if len(ps) < min_needed:
        return None

    log_ret_full = np.log(ps / ps.shift(1)).dropna().values
    log_ret = log_ret_full[-window:]
    if len(log_ret) < rv_w + 10:
        return None

    rv = compute_realized_vol(log_ret, rv_w)
    N = len(rv)
    if N < 20:
        return None

    t = np.arange(N, dtype=np.float64)

    try:
        model = GPVolModel(t, rv)
        model.fit()
    except Exception as e:
        print(f"    Failed {ticker}: {e}")
        return None

    mu_loo, var_loo = model.loo_predictive()
    resid = rv - mu_loo
    std_loo = np.sqrt(var_loo)

    anomaly_z_today = float(resid[-1] / (std_loo[-1] + 1e-10))

    fit_quality = float(1.0 - np.clip(
        np.sum(resid ** 2) / (np.sum((rv - rv.mean()) ** 2) + 1e-10), 0.0, 1.0
    ))

    t_future = np.arange(N, N + H, dtype=np.float64)
    _, var_future = model.predict(t_future)
    std_future = np.sqrt(var_future)
    regime_width = float(np.mean(std_future) / (rv.mean() + 1e-10))

    return {
        "anomaly_z": anomaly_z_today,
        "regime_width": regime_width,
        "fit_quality": fit_quality,
        "current_vol": float(rv[-1]),
        "gp_expected_vol": float(mu_loo[-1]),
    }


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_gp_vol_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.DataFrame:
    """
    Fit a GP volatility model per ETF (pure univariate — no macro
    conditioning) and extract a vol-anomaly + regime-stability signal.
    Returns a DataFrame of score + diagnostics (cross-sectional z-scored
    on the composite).
    """
    cols = ["score", "anomaly_z", "regime_width", "fit_quality",
            "current_vol", "gp_expected_vol"]
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.DataFrame(columns=cols)

    raw_scores = {}

    for ticker in avail:
        print(f"    Fitting GP vol model for {ticker}")
        diag = forecast_and_diagnose(prices, ticker, window)
        if diag is None:
            continue

        anomaly_z    = diag["anomaly_z"]
        regime_width = diag["regime_width"]
        fit_quality  = diag["fit_quality"]

        print(f"    {ticker}: anomaly_z={anomaly_z:.3f}  "
              f"regime_width={regime_width:.3f}  fit={fit_quality:.3f}")

        favorable = -anomaly_z
        sign = np.sign(favorable) if favorable != 0 else 1.0

        composite = (
            config.WEIGHT_ANOMALY * favorable
            + config.WEIGHT_REGIME  * (-regime_width) * sign
            + config.WEIGHT_FIT      * fit_quality
        )
        raw_scores[ticker] = {"composite": composite, **diag}

    if not raw_scores:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(raw_scores).T
    mu_s, std_s = df["composite"].mean(), df["composite"].std()
    if std_s < 1e-10:
        df["score"] = 0.0
    else:
        df["score"] = (df["composite"] - mu_s) / std_s
    return df[cols]
