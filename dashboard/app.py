"""Period Audit dashboard. Run: streamlit run dashboard/app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets Streamlit Community Cloud import the package without PYTHONPATH

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from period_audit.grouper import Period, group
from period_audit.scoring import score_claim

OUT = ROOT / "data" / "outputs"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
UNDER, OVER, GRAY = "#d03b3b", "#ec835a", "#b4b2a9"
INK2, GRID = "#52514e", "#e7e6e1"
LABELS = {"ai_draft_unreviewed": "Unreviewed AI-drafted OASIS", "missed_comorbidity": "Missed comorbidity code",
          "rtp_principal": "Unacceptable principal dx (RTP)", "sequencing_error": "Principal/secondary sequencing",
          "unsupported_comorbidity": "Unsupported comorbidity added", "missed_visit_note": "Visit note not filed"}

st.set_page_config(page_title="Period Audit", layout="wide")


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=40, b=8), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                      font=dict(color=INK2, size=13), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data
def load():
    return (json.loads((OUT / "summary.json").read_text()), pd.read_parquet(OUT / "periods.parquet"),
            pd.read_csv(OUT / "isolated_errors.csv"), pd.read_csv(OUT / "qa_capture_curve.csv"),
            pd.read_csv(OUT / "ai_review_sweep.csv"), pd.read_csv(OUT / "by_component.csv"))


S, P, ISO, CURVE, SWEEP, COMP = load()
sm, q = S["summary"], S["qa_model"]

st.title("Period Audit")
st.caption("Where home health revenue leaks between the visit and the claim. **All periods are SYNTHETIC**; "
           "grouping and prices use the real CMS PDGM grouper tables (v07.2.26) and CY 2026 rates.")
tabs = st.tabs(["Leakage", "Where it comes from", "QA prioritization", "AI review", "Try the grouper"])

with tabs[0]:
    c = st.columns(5)
    c[0].metric("Simulated 30-day periods", f"{int(sm['periods']):,}")
    c[1].metric("Revenue supported by the record", f"${sm['revenue_supported']/1e6:,.2f}M")
    c[2].metric("Net under-billed", f"${-sm['net_dollars']/1e3:,.0f}K",
                f"{sm['net_dollars']/sm['revenue_supported']:.2%} of supported", delta_color="off")
    c[3].metric("Paid claims with a $ error", f"{sm['share_periods_with_revenue_error']:.1%}")
    c[4].metric("Returned to provider", f"{sm['rtp_rate']:.1%}", f"${sm['dollars_delayed_by_rtp']/1e6:,.2f}M delayed",
                delta_color="off")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=["Under-coded", "Over-coded"], y=[-sm["under_coded_dollars"], sm["over_coded_dollars"]],
                    marker_color=[UNDER, OVER], text=[f"${-sm['under_coded_dollars']/1e3:,.0f}K",
                                                      f"${sm['over_coded_dollars']/1e3:,.0f}K"],
                    textposition="outside", hovertemplate="%{x}: $%{y:,.0f}<extra></extra>")
        fig.update_layout(title="Dollars billed wrong, by direction", yaxis_tickprefix="$", showlegend=False)
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Over-coding is a compliance exposure, not a gain: it is money an audit can claw back.")
    with right:
        comp = COMP.sort_values("periods")
        fig = go.Figure(go.Bar(y=comp.component_changed, x=comp.periods, orientation="h", marker_color=BLUE,
                               customdata=comp[["net_dollars", "avg_abs_dollars"]],
                               hovertemplate="%{y}<br>%{x} periods<br>net $%{customdata[0]:,.0f}"
                                             "<br>avg |error| $%{customdata[1]:,.0f}<extra></extra>"))
        fig.update_layout(title="Which part of the PDGM payment moved (paid claims with an error)")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("LUPA flips are rare but cost ~$1,750 each, the most expensive error per incident.")

with tabs[1]:
    st.markdown("Each error process switched on **alone** at its scenario rate, same 20,000-period book. "
                "Rates are illustrative assumptions you can change in `Scenario`; dollars come from the real grouper.")
    iso = ISO.assign(label=ISO.error_type.map(LABELS)).sort_values("net_dollars_per_period")
    fig = go.Figure()
    fig.add_bar(y=iso.label, x=-iso.under_dollars_per_period, orientation="h", name="Under-billed $/period",
                marker_color=UNDER)
    fig.add_bar(y=iso.label, x=iso.over_dollars_per_period, orientation="h", name="Over-billed $/period",
                marker_color=OVER)
    fig.update_layout(barmode="group", title="Dollar error per period in the book, by error type", xaxis_tickprefix="$")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.dataframe(iso[["label", "rate", "share_periods_payment_changed", "net_dollars_per_period",
                      "under_dollars_per_period", "over_dollars_per_period", "rtp_rate"]].rename(columns={"label": "error"}),
                 hide_index=True, width="stretch")

with tabs[2]:
    st.markdown(f"QA can't read every chart. Ranking **{q['test_periods']:,} held-out paid claims** by the model "
                f"(selected by cross-validation on training claims only) vs a simple rule and random review.")
    fig = go.Figure()
    for col, name, color in [("model", "Model: expected $ error", BLUE), ("rule_ai_then_boundary",
                             "Rule: AI-drafted first, then nearest level boundary", ORANGE),
                             ("classifier", "Model: P(any error)", AQUA), ("random", "Random review", GRAY)]:
        fig.add_scatter(x=CURVE.share_reviewed, y=CURVE[col], name=name, mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=7),
                        hovertemplate=name + "<br>review %{x:.0%} -> catch %{y:.0%} of $ error<extra></extra>")
    fig.add_vline(x=0.2, line_dash="dot", line_color=GRAY)
    fig.update_layout(title="Share of dollar error caught vs share of charts reviewed",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(style(fig, 460), width="stretch")
    ci = q["capture_at_20pct_ci95"]
    st.markdown(f"At **20% review**, the model catches **{q['capture_at_20pct']['model']:.0%}** of dollar error "
                f"(95% CI {ci['model'][0]:.0%}-{ci['model'][1]:.0%}) vs **{q['capture_at_20pct']['rule']:.0%}** for the rule "
                f"and {q['capture_at_20pct']['random']:.0%} for random review.")

with tabs[3]:
    fig = go.Figure()
    fig.add_scatter(x=SWEEP.ai_draft_unreviewed, y=SWEEP.abs_dollars_per_period, name="Absolute $ error / period",
                    mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8))
    fig.add_scatter(x=SWEEP.ai_draft_unreviewed, y=SWEEP.over_dollars_per_period, name="Over-billed $ / period",
                    mode="lines+markers", line=dict(color=OVER, width=2), marker=dict(size=8))
    fig.update_layout(title="Share of AI-drafted OASIS reaching billing unreviewed", xaxis_tickformat=".0%",
                      yaxis_tickprefix="$")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption("AI draft error patterns are the ones measured for a local 7B model in Second Read (project 1). "
               "A stronger scribe would flatten this curve; the point is that the review step is worth real money.")

with tabs[4]:
    st.markdown("Enter a period and get its CY 2026 HIPPS code, payment and pre-bill QA risk.")
    a, b = st.columns(2)
    with a:
        pdx = st.text_input("Principal diagnosis (ICD-10)", "I5022")
        sec = st.text_input("Secondary diagnoses (comma separated)", "I480, E1142, F32A")
        timing = st.radio("Timing", ["early", "late"], horizontal=True)
        inst = st.checkbox("Institutional referral (hospital/SNF within 14 days)", True)
        ai = st.checkbox("OASIS came from an unreviewed AI draft", True)
        visits = st.slider("Total visits (SN)", 1, 20, 6)
    with b:
        oasis = {k: st.selectbox(k, list(range(n + 1)), index=d) for k, n, d in
                 [("M1800", 3, 2), ("M1810", 3, 2), ("M1820", 3, 2), ("M1830", 6, 3), ("M1840", 4, 1),
                  ("M1850", 5, 2), ("M1860", 6, 3)]}
        oasis["M1033"] = st.multiselect("M1033 risks", [str(i) for i in range(1, 10)], ["1", "3", "7"]) or ["10"]
    p = Period(pdx, [s.strip() for s in sec.split(",") if s.strip()], timing, inst, oasis, {"sn": visits})
    res = score_claim(p, ai)
    g = res["grouping"]
    st.subheader(f"HIPPS {g['hipps']}  ·  {g.get('clinical_group_name') or 'ungroupable'}  ·  "
                 f"{g.get('functional_level') or ''}  ·  comorbidity {g.get('comorbidity') or '-'}")
    if g["return_code"] != "00":
        st.error(res["action"] + " - " + "; ".join(res["reasons"]))
    else:
        c = st.columns(4)
        c[0].metric("Payment (national)", f"${g['payment']:,.2f}")
        c[1].metric("Functional points", g["functional_points"])
        c[2].metric("LUPA", "yes" if g["is_lupa"] else "no", f"threshold {g['lupa_threshold']} visits", delta_color="off")
        c[3].metric("QA risk", f"${res['risk_score']:,.0f}", res["score_meaning"], delta_color="off")
        for r in res["reasons"]:
            st.markdown(f"- {r}")
