"""The Python grouper must reproduce every CMS test claim shipped with HHGS v07.2.26."""

from pathlib import Path

import pytest

from period_audit.cms_claims import parse
from period_audit.grouper import Period, group

FIX = Path(__file__).parent / "fixtures"
RECORDS = [parse(l) for f in ("TestDataV07226.txt", "TestDataV07226_GRC.txt")
           for l in open(FIX / f, encoding="latin-1") if l.strip()]


@pytest.mark.parametrize("rec", RECORDS, ids=[f"{i:03d}-{r.claim_id[:20]}" for i, r in enumerate(RECORDS)])
def test_matches_cms_expected_output(rec):
    g = group(rec.period)
    assert (g.hipps, g.validity_flag, g.return_code) == (rec.expected_hipps, rec.expected_validity, rec.expected_rc)


def test_cms_fixture_count():
    assert len(RECORDS) == 141


INDEP = {"M1800": 0, "M1810": 0, "M1820": 0, "M1830": 0, "M1840": 0, "M1850": 0, "M1860": 0, "M1033": ["10"]}


def test_comorbidity_high_beats_low_and_same_subchapter_is_ignored():
    # I5022 heart failure (Heart_11, low) + L89154 sacral pressure ulcer stage 4 (Skin_4): check the
    # interaction table decides high vs low, and that a secondary in the principal's subchapter is ignored.
    base = group(Period("I110", ["I5022"], oasis=INDEP))
    assert base.comorbidity == "low"
    same_sub = group(Period("I5022", ["I5023"], oasis=INDEP))
    assert same_sub.comorbidity == "none"


def test_primary_awarding_promotes_first_secondary():
    g = group(Period("I110", ["Z452"], oasis=INDEP))
    assert g.principal_used == "Z452" and g.clinical_group == "D"


def test_lupa_pricing_uses_per_visit_rates_and_addon():
    p = Period("I110", [], oasis=INDEP, visits={"sn": 2}, initial_period=True)
    g = group(p)
    assert g.is_lupa  # 2 visits is under every CY 2026 threshold
    assert g.payment == round(176.96 * 2 + 176.96 * 0.72, 2)
