"""
Monte Carlo Risk Studio — guided web UI.

Run:  streamlit run app.py
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from mc.models.breach import BreachLikelihoodModel, SECTOR_RISK, factor_contributions
from mc.models.cyber_cost import CyberCostModel
from mc.models.dcf import DCFModel
from mc.models.var import VaRModel

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Monte Carlo Risk Studio", page_icon="📊", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Risk Studio")
st.sidebar.caption("Probability-first financial and cyber risk analysis.")

mode = st.sidebar.radio(
    "Choose a model",
    ["🔐 Breach Likelihood", "💸 Cyber Breach Cost", "📈 DCF Valuation", "📉 Portfolio VaR"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Outputs are probability distributions, not predictions. Not financial advice.")

# ── Constants ─────────────────────────────────────────────────────────────────
SECTORS = sorted(SECTOR_RISK.keys())

SIZE_TIERS = {
    "Startup  (< $500M)":       200,
    "Small    ($500M – $5B)": 2_000,
    "Mid      ($5B – $50B)": 15_000,
    "Large    ($50B – $500B)": 100_000,
    "Mega     (> $500B)":     600_000,
}

N = 10_000


# ── Helper: histogram ─────────────────────────────────────────────────────────
def _hist(values, title, xlabel, p10, p50, p90, vlines=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=80, color="#4C78A8", alpha=0.85, edgecolor="none")
    for val, lbl, col in [
        (p10, f"p10  {p10:,.1f}", "#F58518"),
        (p50, f"p50  {p50:,.1f}", "#E45756"),
        (p90, f"p90  {p90:,.1f}", "#54A24B"),
    ]:
        ax.axvline(val, color=col, linewidth=2, linestyle="--", label=lbl)
    if vlines:
        ymax = ax.get_ylim()[1]
        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        for x, lbl, col in vlines:
            ax.axvline(x, color=col, linewidth=1.5, linestyle=":", alpha=0.6)
            ax.text(x + xrange * 0.005, ymax * 0.85, lbl, color=col, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Simulations", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Helper: factor breakdown chart ────────────────────────────────────────────
def _factor_chart(contribs: dict[str, float]) -> object:
    labels = list(contribs.keys())
    values = list(contribs.values())
    colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B", "#B279A2"]
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    ax.set_xlabel("Score contribution (pts)", fontsize=10)
    ax.set_title("What's driving the score?", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 35)
    for bar, val in zip(bars, values):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val} pts", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 🔐 BREACH LIKELIHOOD
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔐 Breach Likelihood":
    st.title("🔐 Breach Likelihood")
    st.markdown(
        "How likely is this company to suffer a material cyber breach? "
        "Enter what you know — the model handles the uncertainty."
    )

    c1, c2 = st.columns(2)
    with c1:
        name   = st.text_input("Company name", "My Company")
        sector     = st.selectbox("Industry", SECTORS, index=SECTORS.index("Healthcare"))
        market_cap = st.number_input(
            "Company market cap ($M)",
            min_value=1.0, value=500.0, step=50.0,
            help="Market capitalization in millions. E.g. $12M company → enter 12",
        )
    with c2:
        beta   = st.slider("Market beta",        0.2, 3.0, 1.0, 0.1,
                           help="How volatile is the stock vs the market? 1.0 = average.")
        growth = st.slider("Annual revenue growth", -10, 60, 10, 1, format="%d%%")
        peers  = st.slider("Peer companies with known breaches", 0, 15, 3)

    if st.button("Run Model", type="primary", use_container_width=True):
        company_info = {
            "name": name, "sector": sector,
            "market_cap_m": market_cap,
            "beta": beta, "rev_growth": growth / 100,
        }
        industry_rows = [{"ticker": f"P{i}"} for i in range(peers)]

        with st.spinner("Running 10,000 simulations…"):
            res = BreachLikelihoodModel(
                company_info=company_info,
                industry_rows=industry_rows,
                n=N,
            ).run()

        ICON = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}
        st.markdown(f"### {ICON[res['tier']]} {res['tier']} tier — {name}")

        # Histogram first — the distribution IS the point
        fig = _hist(
            res["scores"],
            f"Breach Score Distribution — {name}",
            "Score  (0 = very safe · 100 = certain breach)",
            res["p10"], res["p50"], res["p90"],
            vlines=[(30, "Moderate", "#F58518"), (50, "High", "#E45756"), (70, "Critical", "#B40000")],
        )
        st.pyplot(fig)
        plt.close()

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pessimistic (p10)",  f"{res['p10']}/100")
        c2.metric("Median (p50)",       f"{res['p50']}/100")
        c3.metric("Worst-case (p90)",   f"{res['p90']}/100")
        c4.metric("Prob. Critical (70+)", f"{res['prob_critical_pct']}%")

        # Factor breakdown
        contribs = factor_contributions(company_info, industry_rows)
        fig2 = _factor_chart(contribs)
        st.pyplot(fig2)
        plt.close()

        st.info(res["narrative"])

        # Pass score forward to cost model
        st.session_state.update({
            "_score": res["p50"], "_sector": sector, "_market_cap": market_cap,
        })
        st.caption("Switch to **💸 Cyber Breach Cost** to see the dollar impact.")

        st.markdown("---")
        st.markdown(
            "💼 **Want a full BreachIQ assessment?** "
            "[breachiq.io](https://breachiq.io) — deeper analysis, custom reporting."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 💸 CYBER BREACH COST
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "💸 Cyber Breach Cost":
    st.title("💸 Cyber Breach Cost")
    st.markdown(
        "If a breach occurs, what does it cost? "
        "Calibrated to IBM Cost of a Data Breach 2023. Run **Breach Likelihood** first to pre-fill the score."
    )

    prev_score  = st.session_state.get("_score")
    prev_sector = st.session_state.get("_sector", "Healthcare")
    prev_market_cap = st.session_state.get("_market_cap", 500.0)

    c1, c2 = st.columns(2)
    with c1:
        sector = st.selectbox("Industry", SECTORS,
                              index=SECTORS.index(prev_sector) if prev_sector in SECTORS else 0)
        market_cap = st.number_input(
            "Company market cap ($M)",
            min_value=1.0,
            value=float(prev_market_cap),
            step=50.0,
            help="Market capitalization in millions. E.g. $12M company → enter 12",
        )
    with c2:
        score = st.slider(
            "Breach likelihood score (0–100)",
            0, 100, int(prev_score) if prev_score else 50,
            help="From the Breach Likelihood model, or enter manually.",
        )

    if st.button("Run Model", type="primary", use_container_width=True):
        with st.spinner("Running 10,000 simulations…"):
            res = CyberCostModel(
                market_cap_m=market_cap,
                sector=sector,
                breach_score=float(score),
                n=N,
            ).run()

        st.markdown("### 💵 Projected Breach Cost")
        eal = res["p50"] * score / 100

        # Histogram first
        fig = _hist(res["scores"], "Breach Cost Distribution", "Cost ($M)",
                    res["p10"], res["p50"], res["p90"])
        st.pyplot(fig)
        plt.close()

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Low estimate (p10)",  f"${res['p10']:,.0f}M")
        c2.metric("Expected (p50)",      f"${res['p50']:,.0f}M")
        c3.metric("Severe (p90)",        f"${res['p90']:,.0f}M")
        c4.metric("Expected Annual Loss", f"${eal:,.0f}M",
                  help="p50 cost × P(breach). Standard cyber insurance pricing input.")

        st.caption(
            "The long right tail represents catastrophic outcomes — regulatory fines, "
            "class-action settlements, extended remediation. p90 is the tail, not the expected outcome."
        )

        st.markdown("---")
        st.markdown(
            "💼 **Want a full BreachIQ assessment?** "
            "[breachiq.io](https://breachiq.io) — deeper analysis, custom reporting."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 📈 DCF VALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 DCF Valuation":
    st.title("📈 DCF Valuation")
    st.markdown(
        "Enterprise value as a range, not a number. "
        "Every assumption gets a pessimistic / expected / optimistic instead of a false point estimate."
    )

    c1, c2 = st.columns(2)
    with c1:
        revenue  = st.number_input(
            "Current annual revenue ($M)", min_value=1.0, value=500.0, step=50.0,
            help="Where to find this: 10-K income statement, Yahoo Finance financials tab.",
        )
        years    = st.select_slider("Forecast horizon", [5, 7, 10, 15], value=10, format_func=lambda x: f"{x} years")
    with c2:
        net_debt = st.number_input("Net debt ($M)  —  negative = net cash position", value=0.0, step=50.0)

    st.markdown("---")

    def _tri(label, lo_d, mid_d, hi_d, key, mn=0, mx=80):
        st.markdown(f"**{label}**")
        a, b, c = st.columns(3)
        lo  = a.slider("Pessimistic",  mn, mx, lo_d,  key=f"{key}lo",  format="%d%%")
        mid = b.slider("Expected",     mn, mx, mid_d, key=f"{key}mid", format="%d%%")
        hi  = c.slider("Optimistic",   mn, mx, hi_d,  key=f"{key}hi",  format="%d%%")
        if not (lo <= mid <= hi):
            st.warning(f"⚠️ {label}: values must be Pessimistic ≤ Expected ≤ Optimistic.")
        return lo / 100, mid / 100, hi / 100

    rev_g  = _tri("Revenue growth (annual %)", 8,  14, 20, "rg", mn=-20)
    margin = _tri("Operating margin (%)",      10, 15, 22, "om")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    dr_mean = c1.slider("Discount rate",              6, 20, 10, format="%d%%", key="dr")
    dr_std  = c2.slider("Discount rate — uncertainty", 0,  8,  2, format="%d%%", key="drstd")
    tg_mean = c3.slider("Terminal growth rate",        1,  5,  2, format="%d%%", key="tg")
    tg_std  = c4.slider("Terminal growth — uncertainty", 0, 2,  1, format="%d%%", key="tgstd")

    ok = rev_g[0] <= rev_g[1] <= rev_g[2] and margin[0] <= margin[1] <= margin[2]

    if st.button("Run Model", type="primary", use_container_width=True, disabled=not ok):
        with st.spinner("Running 10,000 simulations…"):
            res = DCFModel(
                revenue=revenue,
                revenue_growth=rev_g,
                operating_margin=margin,
                discount_rate=(dr_mean / 100, dr_std / 100),
                terminal_growth=(tg_mean / 100, tg_std / 100),
                years=years,
                net_debt=net_debt,
                n=N,
            ).run()

        st.markdown("### 🏢 Enterprise Value")

        # Histogram first
        fig = _hist(res["values"], "Enterprise Value Distribution", "Enterprise Value ($M)",
                    res["p10"], res["p50"], res["p90"])
        st.pyplot(fig)
        plt.close()

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Bear case (p10)", f"${res['p10']:,.0f}M")
        c2.metric("Base case (p50)", f"${res['p50']:,.0f}M")
        c3.metric("Bull case (p90)", f"${res['p90']:,.0f}M")

        if net_debt != 0:
            st.markdown("**Equity value (enterprise value − net debt)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Equity p10", f"${res['equity_p10']:,.0f}M")
            c2.metric("Equity p50", f"${res['equity_p50']:,.0f}M")
            c3.metric("Equity p90", f"${res['equity_p90']:,.0f}M")


# ══════════════════════════════════════════════════════════════════════════════
# 📉 PORTFOLIO VaR
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📉 Portfolio VaR":
    st.title("📉 Portfolio VaR")
    st.markdown(
        "Value at Risk: how much could this portfolio lose in a bad scenario? "
        "Uses correlated Monte Carlo simulation across all assets."
    )

    portfolio_value = st.number_input("Portfolio value ($)", min_value=1_000, value=1_000_000, step=10_000)

    n_assets = st.slider("Number of assets", 1, 5, 3)

    st.markdown("**Asset details** — annual figures, converted to daily internally")
    st.caption("Expected return and volatility are annual percentages.")

    asset_names, returns, vols, weights = [], [], [], []
    cols = st.columns(n_assets)
    for i, col in enumerate(cols):
        with col:
            default_names = ["Equities", "Bonds", "Gold", "Real Estate", "Commodities"]
            name = st.text_input("Asset name", default_names[i], key=f"aname{i}")
            ret  = st.slider("Expected return (%/yr)", -10, 30, [7, 3, 5, 6, 4][i], key=f"aret{i}")
            vol  = st.slider("Volatility (%/yr)",        1, 50, [15, 5, 12, 10, 18][i], key=f"avol{i}")
            wt   = st.slider("Portfolio weight (%)",      1, 99, [40, 40, 10, 5, 5][i], key=f"awt{i}")
            asset_names.append(name)
            returns.append(ret / 100 / 252)      # annual → daily
            vols.append(vol / 100 / (252 ** 0.5)) # annual → daily
            weights.append(wt)

    # Normalize weights
    total_wt = sum(weights)
    weights_norm = [w / total_wt for w in weights]
    if abs(total_wt - 100) > 1:
        st.info(f"Weights sum to {total_wt}% — auto-normalized to 100%.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    correlation = c1.slider("Correlation between assets", -0.5, 0.9, 0.2, 0.05,
                            help="How much do assets move together? 0 = independent, 1 = perfectly correlated.")
    confidence  = c2.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95,
                                   format_func=lambda x: f"{int(x*100)}%")
    horizon     = c3.select_slider("Time horizon", [1, 5, 10, 30], value=1,
                                   format_func=lambda x: f"{x} day{'s' if x > 1 else ''}")

    # Build covariance matrix from individual vols + single correlation
    n = n_assets
    corr_matrix = np.full((n, n), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    # Clamp correlation to ensure positive definite
    min_corr = -1.0 / max(n - 1, 1) + 0.01
    if correlation < min_corr:
        st.warning(f"Correlation {correlation} may cause issues with {n} assets. Clamped to {min_corr:.2f}.")
        corr_matrix = np.full((n, n), min_corr)
        np.fill_diagonal(corr_matrix, 1.0)

    sigma = np.array(vols)
    cov   = np.outer(sigma, sigma) * corr_matrix

    if st.button("Run Model", type="primary", use_container_width=True):
        with st.spinner("Running 50,000 simulations…"):
            res = VaRModel(
                portfolio_value=portfolio_value,
                returns_mean=np.array(returns),
                returns_cov=cov,
                weights=np.array(weights_norm),
                horizon_days=horizon,
                confidence=confidence,
                n=50_000,
            ).run()

        conf_pct = int(confidence * 100)
        st.markdown(f"### 📉 {conf_pct}% {horizon}-day Value at Risk")

        pnl = res["pnl_scores"]

        # Histogram first
        fig = _hist(pnl,
                    f"Portfolio P&L Distribution — {horizon}-day horizon",
                    "Profit / Loss ($)",
                    float(np.percentile(pnl, 10)),
                    float(np.percentile(pnl, 50)),
                    float(np.percentile(pnl, 90)))
        st.pyplot(fig)
        plt.close()

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{conf_pct}% VaR",  f"${res['var']:,.0f}",
                  help="Max expected loss at this confidence level")
        c2.metric("Expected Shortfall", f"${res['cvar']:,.0f}",
                  help="Average loss in the worst scenarios beyond VaR")
        c3.metric("Worst 1% loss",      f"${res['p1_loss']:,.0f}")
        c4.metric("Prob. any loss",     f"{res['prob_loss_pct']}%")

        st.caption(
            f"VaR means: in {100 - conf_pct}% of scenarios, losses exceed "
            f"**${res['var']:,.0f}**. Expected Shortfall is the average of those worst-case scenarios."
        )
