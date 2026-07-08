import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="GP Vol Forecasting Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1a2a33; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #3d7a8c; }
.etf-card    { background:linear-gradient(135deg,#1a2a33 0%,#3d7a8c 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1a2a33 0%,#254855 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 GP Vol Forecasting Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Gaussian Process regression (Matern 5/2 kernel) on realized volatility · '
    'Exact posterior mean (forecast) + posterior variance (uncertainty) · '
    'Closed-form leave-one-out anomaly detection, no neural network · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## GP Vol Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Realized vol window:** {config.RV_WINDOW}d")
st.sidebar.markdown(f"**Kernel:** Matern {config.MATERN_NU}")
st.sidebar.markdown(
    f"**Fitting:** {config.GP_EPOCHS} Adam steps on 3 log-hyperparameters | lr={config.GP_LR}")
st.sidebar.markdown(f"**Forecast horizon:** {config.PRED_HORIZON}d")
st.sidebar.markdown(
    f"**Weights:** Anomaly {config.WEIGHT_ANOMALY:.0%} | "
    f"Regime {config.WEIGHT_REGIME:.0%} | "
    f"Fit {config.WEIGHT_FIT:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        files = [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                           detail=True, recursive=True)
                 if f["type"] == "file"]
        return files, None
    except Exception as e:
        return [], str(e)


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files, list_error = list_repo_files()

with st.expander("🔧 Debug: what the dashboard sees on HuggingFace", expanded=bool(list_error)):
    st.markdown(f"**Repo:** `{OUTPUT_REPO}`  ·  **Token set:** {'yes' if bool(HF_TOKEN) else 'no'}")
    if list_error:
        st.error(f"Could not list repo files: {list_error}")
    else:
        st.write(f"{len(files)} file(s) found:")
        st.code("\n".join(sorted(files)) if files else "(empty)")

tab1_path = find_latest(files, "gp_vol_engine_2")
tab2_path = find_latest(files, "gp_vol_engine_windows_")

if not tab1_path:
    if list_error:
        st.error("Could not reach HuggingFace to look for results (see 🔧 Debug above).")
    else:
        st.error(
            "Connected to HuggingFace successfully, but no file matching "
            "`gp_vol_engine_2*.json` was found (see 🔧 Debug above for the "
            "exact file list). Run trainer.py, or check the filename it "
            "actually pushed."
        )
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Volatility Anomaly Signal")

    with st.expander("GP Vol Forecasting Methodology", expanded=True):
        st.markdown("""
A **Gaussian Process** places a distribution over functions, giving an
EXACT posterior mean (forecast) and posterior variance (uncertainty) —
not an approximation, not a trained neural network:

```
mu*     = k*^T (K + sigma_n^2 I)^-1 y
sigma*^2 = k(t*,t*) - k*^T (K + sigma_n^2 I)^-1 k*
```

**Matern 5/2 kernel** (closed form, no Bessel functions needed):

```
k(r) = sigma_f^2 * (1 + sqrt5*r/l + 5r^2/(3l^2)) * exp(-sqrt5*r/l)
```

**Hyperparameters fit by exact marginal likelihood gradient** — Adam on
3 scalar log-hyperparameters (lengthscale, signal variance, noise
variance), using the closed-form GP gradient (Rasmussen & Williams 2006),
not backprop through a computation graph.

**Leave-one-out (LOO) cross-validation, in closed form** — every training
point's out-of-sample-style predictive mean/variance is available WITHOUT
refitting N times:

```
mu_LOO_i  = y_i - alpha_i / [K_y^-1]_ii
var_LOO_i = 1 / [K_y^-1]_ii
```

This is what makes "today's vol is anomalous" a genuine out-of-sample
statement — a naive in-sample residual would be near-zero by construction
since a GP fits its own training points closely.

**Signal:**

```
score = 0.50*(-anomaly_z) + 0.25*(-regime_width)*sign(-anomaly_z) + 0.25*fit_quality
```

- `anomaly_z` — standardized LOO residual at today: how anomalous is
  CURRENT realized vol relative to the GP's expectation? Negative (calmer
  than expected) is treated as favorable — a risk-preference convention
  standard in low-volatility investing, **not** a specific return-direction
  causal claim.
- `regime_width` — average forecast posterior std over the horizon,
  normalized by historical vol level: the posterior's own WIDTH is a
  regime-stability signal (wide = uncertain/unstable, narrow = confident).
- `fit_quality` — LOO-CV R²-style diagnostic on the GP's own fit.

**Distinct from GPLVM-ANOMALY** elsewhere in this suite: that engine uses
a GP for latent variable modelling (unsupervised representation learning).
This engine uses a GP directly as a time series regression model on
realized volatility — a different application of the same machinery.

**Validated before shipping:** kernel gradients, the full marginal
likelihood gradient (trace formula), and the LOO closed-form shortcut
were all checked against finite differences / brute-force refitting to
numerical precision. A synthetic vol spike was correctly flagged
(anomaly_z ≈ 11) against a calm control ticker (anomaly_z ≈ -0.3).
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">GP score = {etf['gp_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
  <div class="etf-score">anomaly z = {etf.get('anomaly_z', float('nan')):.2f}</div>
  <div class="etf-score">regime width = {etf.get('regime_width', float('nan')):.2f}</div>
  <div class="etf-score">lengthscale = {etf.get('fitted_lengthscale', float('nan')):.1f}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "GP Score": info.get("score"),
                        "Best Window (d)": info.get("best_window", "N/A"),
                        "Anomaly Z": info.get("anomaly_z"),
                        "Regime Width": info.get("regime_width"),
                        "Fit Quality": info.get("fit_quality"),
                        "Fitted Lengthscale (d)": info.get("fitted_lengthscale"),
                        "Current Vol": info.get("current_vol"),
                        "GP Expected Vol": info.get("gp_expected_vol"),
                    })
                df = pd.DataFrame(rows).sort_values("GP Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Gaussian Process regression, Matern 5/2 kernel · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore GP Vol Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — short vol history; fewer points for the GP to fit; reactive, noisier hyperparameters
- **126d** — 6-month window; recommended minimum for a stable GP fit
- **252d** — 1-year window; most stable hyperparameter estimates; recommended primary signal
- **504d** — 2-year window; more history, but vol regimes may shift within the window, which a single stationary kernel can blur
        """)

    st.markdown(f"### GP Vol Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">GP score = {etf['gp_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
  <div class="etf-score">anomaly z = {etf.get('anomaly_z', float('nan')):.2f}</div>
  <div class="etf-score">regime width = {etf.get('regime_width', float('nan')):.2f}</div>
  <div class="etf-score">lengthscale = {etf.get('fitted_lengthscale', float('nan')):.1f}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["ETF", "GP Score", "Anomaly Z", "Regime Width",
                             "Fit Quality", "Fitted Lengthscale (d)"],
                )
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
