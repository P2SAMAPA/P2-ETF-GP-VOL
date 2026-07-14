import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-gp-vol-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Gaussian Process Volatility Forecasting hyperparameters ──────────────────
# A GP places a distribution over functions, fully specified by a mean
# function and a covariance (kernel) function. Given a realized-volatility
# time series, GP regression gives an exact posterior mean (forecast) AND
# posterior variance (uncertainty) — a genuinely different mechanism from
# every other engine in this suite: no gradient descent through a neural
# network, just Cholesky-based exact linear algebra optimizing a marginal
# likelihood over 3 scalar hyperparameters.
#
# MATERN 5/2 KERNEL specifically (not the general Matern-nu family): this is
# the standard practical choice because it has a CLOSED FORM (no modified
# Bessel functions needed), while still being a genuine Matern kernel rather
# than an approximation:
#
#     k(r) = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
#
# Hyperparameters (lengthscale l, signal variance sigma_f^2, noise variance
# sigma_n^2) are fit via maximum likelihood — minimizing the negative log
# marginal likelihood via gradient descent on the log-hyperparameters (kept
# positive by construction), using the EXACT analytical GP gradient formula
# (Rasmussen & Williams, 2006, eq. 5.9), not backprop through a computation
# graph.
#
# LEAVE-ONE-OUT (LOO) CROSS-VALIDATION, in closed form (no need to refit N
# times): given the fitted Cholesky factors, every training point's
# leave-one-out predictive mean and variance is available directly:
#
#     mu_LOO_i  = y_i - alpha_i / [K_y^-1]_ii
#     var_LOO_i = 1 / [K_y^-1]_ii
#
# This is what makes "today's vol is anomalous relative to the GP" a
# genuinely out-of-sample-style statement rather than a trivial in-sample
# fit (which would show near-zero residual by construction).
#
# Distinct from GPLVM-ANOMALY elsewhere in this suite: that engine uses a
# GP for LATENT VARIABLE modelling (an unsupervised dimensionality-reduction
# / representation-learning use of GPs). This engine uses a GP directly as a
# TIME SERIES REGRESSION model on realized volatility — a different
# application of the same underlying machinery, not the same mechanism.

RV_WINDOW = 10          # rolling window (days) used to estimate realized vol at each point
MATERN_NU = 2.5          # documents the kernel family used (closed-form Matern 5/2)
GP_EPOCHS = 100          # hyperparameter optimization steps (only 3 scalars — cheap)
GP_LR     = 0.05
JITTER    = 1e-6          # numerical stability constant added to the kernel diagonal

PRED_HORIZON = 21        # H: forecast horizon for forward vol prediction

# ── Score construction ────────────────────────────────────────────────────────
# anomaly_z       : standardized LOO residual at the most recent point —
#                   how anomalous is TODAY's realized vol relative to what
#                   the GP (fit on the surrounding temporal structure)
#                   would have expected? Negative (calmer than expected) is
#                   treated as the favorable direction, matching standard
#                   low-volatility investing convention: this is a
#                   risk-preference ranking, not a specific return-direction
#                   causal claim.
# regime_width    : average forecast posterior std over the horizon,
#                   normalized by the historical realized-vol level — the
#                   posterior WIDTH ITSELF is treated as a regime signal:
#                   a wide posterior means the GP is uncertain about
#                   near-term vol (unstable regime); narrow means confident
#                   (stable regime). Used as a confirmation term, scaled by
#                   the sign of anomaly_z's contribution.
# fit_quality     : LOO-CV R^2-style diagnostic (1 - normalized LOO residual
#                   sum of squares) — does the fitted GP genuinely explain
#                   this ticker's historical vol dynamics?

WEIGHT_ANOMALY = 0.50
WEIGHT_REGIME    = 0.25
WEIGHT_FIT        = 0.25

TOP_N = 3
