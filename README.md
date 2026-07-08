# 📈 P2-ETF-GP-VOL

**Gaussian Process Volatility Forecasting Engine**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine fits a **Gaussian Process** with a Matérn 5/2 kernel to each
ETF's realized volatility time series — giving an EXACT posterior mean
(forecast) and posterior variance (uncertainty), not an approximation and
not a trained neural network. ETFs whose current realized vol sits in the
tail of the GP's leave-one-out-implied posterior are flagged anomalous;
the posterior's own width is treated as a separate regime-stability
signal.

---

## Theory

### Gaussian Process Regression

Given training data (t_i, y_i), the GP posterior at a new point t* is
exact closed-form Bayesian inference:

```
mu*      = k*^T (K + sigma_n^2 I)^-1 y
sigma*^2 = k(t*,t*) - k*^T (K + sigma_n^2 I)^-1 k*
```

### Matérn 5/2 Kernel

Chosen specifically because it has a **closed form** (no modified Bessel
functions needed), while still being a genuine Matérn kernel rather than
an approximation:

```
k(r) = sigma_f^2 * (1 + sqrt5*r/l + 5r^2/(3l^2)) * exp(-sqrt5*r/l)
```

### Hyperparameter Fitting via Exact Marginal Likelihood

```
NLL = 0.5 y^T alpha + 0.5 log|K_y| + (N/2) log(2*pi),   alpha = K_y^-1 y
```

has an exact gradient (Rasmussen & Williams 2006, eq. 5.9):

```
dNLL/dtheta = -0.5 * ( alpha^T (dK_y/dtheta) alpha - tr(K_y^-1 dK_y/dtheta) )
```

This is closed-form matrix calculus specific to the GP marginal
likelihood — not backprop through a computation graph. Optimized via Adam
on 3 log-hyperparameters (lengthscale, signal variance, noise variance),
kept positive by construction.

### Leave-One-Out Cross-Validation, in Closed Form

Given the fitted Cholesky factors, every training point's leave-one-out
predictive mean and variance is available WITHOUT refitting N times
(Rasmussen & Williams 2006, eq. 5.12-5.13):

```
mu_LOO_i  = y_i - alpha_i / [K_y^-1]_ii
var_LOO_i = 1 / [K_y^-1]_ii
```

This is what makes "today's vol is anomalous relative to the GP" a
genuinely out-of-sample-style statement, not a trivial in-sample residual
(which would be near-zero by construction).

### Score Construction

```
score = 0.50*(-anomaly_z) + 0.25*(-regime_width)*sign(-anomaly_z) + 0.25*fit_quality
```

| Component | Meaning |
|-----------|---------|
| anomaly_z | Standardized LOO residual at today — how anomalous is current vol relative to GP expectation? Negative (calmer than expected) is favorable — a risk-preference ranking convention (standard in low-volatility investing), **not** a specific return-direction causal claim. |
| regime_width | Average forecast posterior std over the horizon, normalized by historical vol level — the posterior's own width as a regime-stability signal. |
| fit_quality | LOO-CV R²-style diagnostic on the GP's own fit quality. |

### Validation

Before shipping:
- **Kernel gradients** (w.r.t. log-lengthscale, log-signal-variance)
  checked against finite differences — exact to ~1e-11.
- **Full marginal likelihood gradient** (including the trace term)
  checked against finite differences — exact to ~1e-11.
- **LOO closed-form formula validated against actual brute-force
  leave-one-out refitting** — mean matched to ~1e-16, variance matched to
  the deliberately-added jitter constant's own magnitude (1e-6), after
  correctly accounting for the fixed-mean convention and the predictive
  (noise-inclusive) variance convention the closed-form formula uses.
- **Synthetic vol-spike test**: a genuine, known vol regime shift was
  correctly flagged (`anomaly_z ≈ 11`, strongly positive) against a calm
  control ticker with no spike (`anomaly_z ≈ -0.3`, near zero).

---

## Distinction from Other GP-Based Engines in the Suite

| Engine | GP's role |
|--------|-------------|
| GPLVM-ANOMALY | Latent variable modelling (unsupervised representation learning) |
| **GP Vol Forecasting (this engine)** | **Direct time series regression on realized volatility** |

Both use Gaussian Process machinery, but for genuinely different
purposes: GPLVM-ANOMALY learns a latent representation; this engine
forecasts an observed time series directly and quantifies uncertainty
about it.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-GP-VOL/
├── config.py          # Universes, GP hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── gp_vol_engine.py     # Core: Matern 5/2 kernel, marginal likelihood, LOO, forecasting
├── trainer.py            # Orchestrator
├── push_results.py       # HfApi.upload_file wrapper
├── streamlit_app.py       # Two-tab Streamlit dashboard
├── us_calendar.py        # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml     # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-GP-VOL
cd P2-ETF-GP-VOL
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-gp-vol-results`

---

## References

- Rasmussen, C. & Williams, C. (2006). Gaussian Processes for Machine
  Learning. MIT Press. (Chapters 2, 5.)
- Matérn, B. (1960). Spatial Variation. Springer Lecture Notes in
  Statistics.
