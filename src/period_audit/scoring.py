"""Score one claim for pre-bill QA review using the saved model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from .analyze import features
from .grouper import FI_ITEMS, Period, group

OUT = Path(__file__).resolve().parents[2] / "data" / "outputs"


@lru_cache(maxsize=1)
def _model():
    meta = json.loads((OUT / "qa_model_meta.json").read_text())
    m = xgb.XGBRegressor() if meta["kind"] == "dollar_regressor" else xgb.XGBClassifier()
    m.load_model(OUT / "qa_model.json")
    return m, meta


def score_claim(p: Period, ai_drafted: bool) -> dict:
    g = group(p)
    if g.return_code != "00":
        return {"grouping": g.to_dict(), "action": "STOP: claim will be returned to provider",
                "reasons": [f"grouper return code {g.return_code}, validity flag {g.validity_flag}"]}
    row = {"points_billed": g.functional_points, "clinical_group_billed": g.clinical_group,
           "level_billed": g.functional_level, "comorbidity_billed": g.comorbidity,
           "n_secondary_billed": len(p.secondary_dx), "visits_billed": sum(p.visits.values()),
           "lupa_threshold_billed": g.lupa_threshold, "timing": p.timing, "institutional": p.institutional,
           "ai_drafted": ai_drafted, **{f"{k}_billed": int(p.oasis[k]) for k in FI_ITEMS}}
    X = features(pd.DataFrame([row]))
    model, meta = _model()
    X = X[meta["features"]]
    s = float(model.predict(X)[0]) if meta["kind"] == "dollar_regressor" else float(model.predict_proba(X)[0, 1])
    reasons = []
    if ai_drafted:
        reasons.append("OASIS functional items came from an unreviewed AI draft")
    if X.dist_to_level_boundary.iloc[0] <= 3:
        reasons.append(f"functional score {g.functional_points} is within 3 points of a level boundary")
    if 0 <= X.visits_over_lupa.iloc[0] <= 1:
        reasons.append("visit count is at or 1 above the LUPA threshold")
    for flag, text in [("flag_bedfast_mismatch", "M1850 and M1860 disagree on bedfast status"),
                       ("flag_chairfast_tub", "chairfast/bedfast patient coded as bathing in shower/tub"),
                       ("flag_bedfast_toilet", "bedfast patient coded as getting to the toilet")]:
        if X[flag].iloc[0]:
            reasons.append(text)
    return {"grouping": g.to_dict(), "risk_score": round(s, 2),
            "score_meaning": "expected $ error" if meta["kind"] == "dollar_regressor" else "P(error)",
            "reasons": reasons}
