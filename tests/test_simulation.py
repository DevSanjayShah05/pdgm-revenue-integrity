import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from period_audit.simulate import Scenario, simulate

OUT = Path(__file__).resolve().parents[1] / "data" / "outputs"
ZERO = dict(ai_draft_unreviewed=0, missed_comorbidity=0, rtp_principal=0, sequencing_error=0,
            unsupported_comorbidity=0, missed_visit_note=0)


@pytest.fixture(scope="module")
def clean():
    return simulate(Scenario(n_periods=1500, seed=11, **ZERO))


def test_no_errors_means_billed_equals_truth(clean):
    assert (clean.payment_delta == 0).all()
    assert (clean.hipps_true == clean.hipps_billed).all()
    assert not clean.rtp.any() and clean.errors.eq("").all()


def test_every_true_claim_groups(clean):
    assert clean.hipps_true.str.len().eq(5).all() and not clean.hipps_true.eq("00000").any()


def test_calibration_is_plausible(clean):
    levels = clean.level_true.value_counts(normalize=True)
    assert levels.between(0.25, 0.42).all()          # CMS designs ~1/3 per level
    assert 0.03 < clean.lupa_true.mean() < 0.11      # national LUPA rates are in single digits


def test_simulation_is_deterministic():
    a = simulate(Scenario(n_periods=300, seed=3))
    b = simulate(Scenario(n_periods=300, seed=3))
    pd.testing.assert_frame_equal(a, b)


def test_rtp_codes_are_returned_and_unpaid():
    d = simulate(Scenario(n_periods=600, seed=5, **{**ZERO, "rtp_principal": 1.0}))
    assert d.rtp.all() and (d.payment_billed == 0).all() and (d.payment_delta == 0).all()


def test_missed_comorbidity_lowers_payment_unless_lupa_status_flips():
    # LUPA thresholds differ by HIPPS group, so dropping a comorbidity can move a LUPA period
    # into a group with a lower visit threshold and RAISE payment. That is real PDGM behaviour.
    d = simulate(Scenario(n_periods=1500, seed=5, **{**ZERO, "missed_comorbidity": 1.0}))
    same_lupa = d.lupa_true == d.lupa_billed
    assert (d.loc[same_lupa, "payment_delta"] <= 0).all() and (d.payment_delta < 0).any()
    assert (d.loc[d.payment_delta > 0, "lupa_true"]).all()


def test_unsupported_comorbidity_raises_payment_unless_lupa_status_flips():
    d = simulate(Scenario(n_periods=1500, seed=5, **{**ZERO, "unsupported_comorbidity": 1.0}))
    same_lupa = d.lupa_true == d.lupa_billed
    assert (d.loc[same_lupa, "payment_delta"] >= 0).all() and (d.payment_delta > 0).any()
    assert (d.loc[d.payment_delta < 0, "lupa_billed"]).all()


def test_committed_results_are_internally_consistent():
    s = json.loads((OUT / "summary.json").read_text())
    assert s["synthetic"] is True
    sm = s["summary"]
    assert abs(sm["under_coded_dollars"] + sm["over_coded_dollars"] - sm["net_dollars"]) < 1
    q = s["qa_model"]
    assert q["capture_at_20pct"]["model"] > q["capture_at_20pct"]["random"]
    curve = pd.read_csv(OUT / "qa_capture_curve.csv")
    for col in ["model", "classifier", "rule_ai_then_boundary", "random"]:
        assert np.all(np.diff(curve[col]) >= -1e-9) and curve[col].iloc[-1] == pytest.approx(1.0)
