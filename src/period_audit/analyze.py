"""End-to-end analysis: simulate -> SQL summaries -> per-error-type isolation -> QA prioritization model.

Outputs in data/outputs/:
  periods.parquet          the simulated book (SYNTHETIC), truth vs billed for every period
  summary.json             SQL book-level summary + isolated error-type impacts + model results
  by_clinical_group.csv, by_component.csv, by_error_tag.csv
  isolated_errors.csv      dollar impact of each error type with all other errors switched off
  qa_capture_curve.csv     share of $ error captured vs share of charts reviewed (model / rule / random)
  qa_scores.parquet        out-of-sample risk scores for the test periods
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .grouper import _groups
from .simulate import Scenario, simulate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"
ERROR_KNOBS = ["ai_draft_unreviewed", "missed_comorbidity", "rtp_principal", "sequencing_error",
               "unsupported_comorbidity", "missed_visit_note"]


def run_sql(df: pd.DataFrame) -> dict:
    con = duckdb.connect()
    con.register("periods_df", df)
    con.execute("create table periods as select * from periods_df")
    con.execute((ROOT / "sql" / "leakage.sql").read_text())
    res = {"summary": con.sql("select * from summary").df().iloc[0].to_dict()}
    for t in ["by_clinical_group", "by_component", "by_error_tag"]:
        d = con.sql(f"select * from {t}").df()
        d.to_csv(OUT / f"{t}.csv", index=False)
        res[t] = d.to_dict("records")
    con.close()
    return res


def isolate_errors(base: Scenario) -> pd.DataFrame:
    """Re-simulate the same book with exactly one error process switched on at its default rate."""
    zero = {k: 0.0 for k in ERROR_KNOBS}
    rows = []
    for knob in ERROR_KNOBS:
        sc = replace(base, **{**zero, knob: getattr(base, knob)})
        d = simulate(sc)
        paid = d[~d.rtp]
        rows.append({
            "error_type": knob, "rate": getattr(base, knob), "periods": len(d),
            "share_periods_payment_changed": float(paid.revenue_error.mean()),
            "net_dollars_per_period": float(d.payment_delta.sum() / len(d)),
            "under_dollars_per_period": float(d.payment_delta.clip(upper=0).sum() / len(d)),
            "over_dollars_per_period": float(d.payment_delta.clip(lower=0).sum() / len(d)),
            "rtp_rate": float(d.rtp.mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- QA prioritization
def features(df: pd.DataFrame) -> pd.DataFrame:
    """Only what a QA team can see on the claim before billing (no truth columns)."""
    g = _groups()
    low = df.clinical_group_billed.map(lambda x: int(g[x]["LOW_POINT"]) if x in g else np.nan)
    high = df.clinical_group_billed.map(lambda x: int(g[x]["HIGH_POINT"]) if x in g else np.nan)
    pts = df.points_billed.astype(float)
    X = pd.DataFrame({
        "points": pts,
        "dist_to_level_boundary": np.minimum((pts - (low + 0.5)).abs(), (pts - (high - 0.5)).abs()),
        "level": df.level_billed.map({"Low": 0, "Medium": 1, "High": 2}),
        "comorbidity": df.comorbidity_billed.map({"none": 0, "low": 1, "high": 2}),
        "n_secondary": df.n_secondary_billed,
        "visits": df.visits_billed,
        "visits_over_lupa": df.visits_billed - df.lupa_threshold_billed,
        "early": (df.timing == "early").astype(int),
        "institutional": df.institutional.astype(int),
        "ai_drafted": df.ai_drafted.astype(int),
        "clinical_group": pd.Categorical(df.clinical_group_billed, categories=list("ABCDEFGHIJKL")),
    })
    for k in ["M1800", "M1810", "M1820", "M1830", "M1840", "M1850", "M1860"]:
        X[k] = df[f"{k}_billed"]
    # Internal-consistency flags a QA nurse would check (cf. Second Read QA rules).
    X["flag_bedfast_mismatch"] = ((df.M1850_billed >= 4) != (df.M1860_billed == 6)).astype(int)
    X["flag_chairfast_tub"] = ((df.M1860_billed >= 4) & (df.M1830_billed <= 3)).astype(int)
    X["flag_bedfast_toilet"] = ((df.M1860_billed == 6) & (df.M1840_billed <= 1)).astype(int)
    return X


def capture_curve(abs_err: np.ndarray, score: np.ndarray, steps=np.linspace(0, 1, 21)) -> list[float]:
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(abs_err[order]) / abs_err.sum()
    n = len(abs_err)
    return [0.0 if s == 0 else float(cum[max(0, int(round(s * n)) - 1)]) for s in steps]


def qa_model(df: pd.DataFrame, seed: int = 7) -> dict:
    paid = df[~df.rtp].reset_index(drop=True)  # returned claims are already stopped by the grouper edit
    X, y = features(paid), paid.revenue_error.astype(int).to_numpy()
    abs_err = paid.payment_delta.abs().to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(paid)), test_size=0.3, random_state=seed, stratify=y)

    def make(kind: str):
        common = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                      enable_categorical=True, tree_method="hist", random_state=seed, n_jobs=4)
        if kind == "classifier":       # P(any payment error)
            return xgb.XGBClassifier(eval_metric="logloss", **common)
        return xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3, **common)  # E[|$ error|]

    def fit_score(kind, tr, te):
        m = make(kind)
        target = y if kind == "classifier" else abs_err
        m.fit(X.iloc[tr], target[tr])
        return m, (m.predict_proba(X.iloc[te])[:, 1] if kind == "classifier" else m.predict(X.iloc[te]))

    # Choose the model type by 5-fold CV on the TRAINING periods only (metric: $ captured at 20% review).
    from sklearn.model_selection import KFold
    cv = {"classifier": [], "dollar_regressor": []}
    for a, b in KFold(5, shuffle=True, random_state=seed).split(idx_tr):
        tr, va = idx_tr[a], idx_tr[b]
        for kind in cv:
            _, s = fit_score(kind, tr, va)
            cv[kind].append(capture_curve(abs_err[va], s, np.array([0.2]))[0])
    chosen = max(cv, key=lambda k: np.mean(cv[k]))
    model, p = fit_score(chosen, idx_tr, idx_te)
    model.save_model(OUT / "qa_model.json")
    (OUT / "qa_model_meta.json").write_text(json.dumps({"kind": chosen, "features": list(X.columns)}))
    _, p_clf = fit_score("classifier", idx_tr, idx_te)
    yt, et = y[idx_te], abs_err[idx_te]
    Xt = X.iloc[idx_te]
    # Transparent rule a QA lead might use today: AI-drafted charts first, then closest to a level boundary.
    rule = Xt.ai_drafted.to_numpy() * 100 - Xt.dist_to_level_boundary.fillna(99).to_numpy()
    rng = np.random.default_rng(seed)
    steps = np.linspace(0, 1, 21)
    curve = pd.DataFrame({"share_reviewed": steps,
                          "model": capture_curve(et, p, steps),
                          "classifier": capture_curve(et, p_clf, steps),
                          "rule_ai_then_boundary": capture_curve(et, rule, steps),
                          "random": capture_curve(et, rng.random(len(et)), steps)})
    curve.to_csv(OUT / "qa_capture_curve.csv", index=False)
    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    scores = paid.iloc[idx_te][["period_id", "clinical_group_billed", "hipps_billed", "payment_billed",
                                "payment_delta", "revenue_error", "errors"]].copy()
    scores["risk_score"] = p.round(4)
    scores.to_parquet(OUT / "qa_scores.parquet", index=False)
    at = lambda col, s: float(curve.loc[np.isclose(curve.share_reviewed, s), col].iloc[0])
    # Bootstrap the test periods (scores fixed) for a CI on $ captured at 20% review.
    boot = {"model": [], "rule": [], "model_minus_rule": []}
    for _ in range(500):
        b = rng.integers(0, len(et), len(et))
        m_ = capture_curve(et[b], p[b], np.array([0.2]))[0]
        r_ = capture_curve(et[b], rule[b], np.array([0.2]))[0]
        boot["model"].append(m_); boot["rule"].append(r_); boot["model_minus_rule"].append(m_ - r_)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    return {
        "train_periods": int(len(idx_tr)), "test_periods": int(len(idx_te)), "test_error_rate": float(yt.mean()),
        "model_selected_by_cv": chosen,
        "cv_capture_at_20pct": {k: float(np.mean(v)) for k, v in cv.items()},
        "roc_auc": float(roc_auc_score(yt, p)), "pr_auc": float(average_precision_score(yt, p)),
        "classifier_roc_auc": float(roc_auc_score(yt, p_clf)),
        "rule_roc_auc": float(roc_auc_score(yt, rule)),
        "capture_at_20pct": {"model": at("model", .2), "classifier": at("classifier", .2),
                             "rule": at("rule_ai_then_boundary", .2), "random": at("random", .2)},
        "capture_at_20pct_ci95": ci,
        "capture_at_30pct": {"model": at("model", .3), "classifier": at("classifier", .3),
                             "rule": at("rule_ai_then_boundary", .3), "random": at("random", .3)},
        "top_features_gain": imp.head(10).round(2).to_dict(),
    }


def ai_review_sweep(base: Scenario, n: int = 10000) -> pd.DataFrame:
    """How leakage moves with the share of AI-drafted OASIS that reaches billing unreviewed."""
    rows = []
    for share in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        d = simulate(replace(base, n_periods=n, ai_draft_unreviewed=share))
        rows.append({"ai_draft_unreviewed": share,
                     "net_dollars_per_period": float(d.payment_delta.mean()),
                     "abs_dollars_per_period": float(d.payment_delta.abs().mean()),
                     "over_dollars_per_period": float(d.payment_delta.clip(lower=0).mean()),
                     "functional_level_mismatch": float((d.level_true != d.level_billed)[~d.rtp].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ai_review_sweep.csv", index=False)
    return out


def main(n_periods: int = 20000) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = Scenario(n_periods=n_periods)
    df = simulate(sc)
    df.to_parquet(OUT / "periods.parquet", index=False)
    res = run_sql(df)
    iso = isolate_errors(sc)
    iso.to_csv(OUT / "isolated_errors.csv", index=False)
    res["isolated_errors"] = iso.to_dict("records")
    res["qa_model"] = qa_model(df)
    res["ai_review_sweep"] = ai_review_sweep(sc).to_dict("records")
    res["scenario"] = asdict(sc)
    res["synthetic"] = True
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    s = res["summary"]
    print(f"periods={s['periods']:,} supported=${s['revenue_supported']:,.0f} net=${s['net_dollars']:,.0f} "
          f"under=${s['under_coded_dollars']:,.0f} over=${s['over_coded_dollars']:,.0f} "
          f"error share={s['share_periods_with_revenue_error']:.3f} rtp={s['rtp_rate']:.3f}")
    print(iso.round(3).to_string(index=False))
    print(json.dumps(res["qa_model"], indent=2))
    print(pd.DataFrame(res["ai_review_sweep"]).round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
