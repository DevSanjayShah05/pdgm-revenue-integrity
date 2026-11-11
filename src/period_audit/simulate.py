"""Simulate a home health agency's book of 30-day periods, then inject documentation and coding errors.

SYNTHETIC DATA. Every period is generated twice:
  truth  - what the clinical record supports
  billed - what goes out on the claim after the error processes below
Both are grouped and priced with the CMS-validated grouper, so every dollar of difference
is attributable to a named error type.

Real inputs: CMS ICD-10 codes and flags, PDGM tables, CY 2026 rates, CMS Table 8 OASIS
distributions, and OASIS error patterns measured in Second Read (project 1).
Assumptions (all exposed in `Scenario`): clinical-group mix, diagnosis prevalence, visit
patterns and the six error rates. They are illustrative, not estimates of any agency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .grouper import FI_ITEMS, Period, dx_info, group

REF = Path(__file__).resolve().parents[2] / "data" / "reference"

# Principal diagnosis options by clinical group (all verified groupable in HHGS v07.2.26).
PROFILES = {
    "H": {"share": 0.20, "pdx": ["I5022", "I5032", "I5042", "I110", "I130", "I4891"], "therapy": 0.6},
    "E": {"share": 0.20, "pdx": ["Z4781", "Z4789", "Z471", "S72001D", "S72141D", "M1711", "M1611", "M4856XD"], "therapy": 1.6},
    "B": {"share": 0.10, "pdx": ["I69351", "I69354", "I63512", "G20A1", "G309", "R1310"], "therapy": 1.4},
    "C": {"share": 0.10, "pdx": ["L89154", "L89153", "L97412", "L97512", "T8131XA", "L03115", "E11621", "Z4801"], "therapy": 0.4},
    "L": {"share": 0.08, "pdx": ["J441", "J449", "J9611", "J189"], "therapy": 0.6},
    "I": {"share": 0.07, "pdx": ["E1122", "E1151", "E1165", "E1142"], "therapy": 0.5},
    "A": {"share": 0.07, "pdx": ["E871", "E860", "E43", "G4733"], "therapy": 0.6},
    "J": {"share": 0.05, "pdx": ["N390", "N184", "K5730"], "therapy": 0.5},
    "K": {"share": 0.05, "pdx": ["C189", "D649", "A419", "D509"], "therapy": 0.5},
    "F": {"share": 0.03, "pdx": ["F0390", "F332", "F319"], "therapy": 0.2},
    "G": {"share": 0.03, "pdx": ["Z4889", "Z4802"], "therapy": 0.8},
    "D": {"share": 0.02, "pdx": ["Z432", "Z431", "Z430", "K9423", "Z5111"], "therapy": 0.3},
}
# Common secondary diagnoses and an illustrative prevalence.
SECONDARY_POOL = {
    "I10": .55, "E785": .45, "M1990": .30, "I2510": .20, "I480": .20, "Z7901": .20, "F32A": .18, "E119": .22,
    "K5900": .18, "E669": .15, "J449": .14, "D649": .14, "I5022": .12, "F0390": .11, "Z794": .10, "N184": .08,
    "G629": .08, "I739": .08, "E1142": .07, "I69351": .05, "L89153": .05, "N390": .05, "E440": .04, "Z993": .05,
    "L97412": .03, "C7951": .02,
}
# "Questionable encounter" codes CMS will not accept as principal -> claim returned to provider.
RTP_CODES = ["M6281", "R262", "R531", "Z9181", "R2689", "Z96651", "Z96641", "R627"]
UPCODE_POOL = ["I5022", "I480", "E1142", "L89153", "D649", "E440"]  # low-comorbidity subgroups

# CMS Table 8 (CY 2026 final rule) "percent of periods" by response; merged bins split by assumption.
MARGINALS = {
    "M1800": {0: .120, 1: .112, 2: .620, 3: .148},
    "M1810": {0: .095, 1: .080, 2: .700, 3: .125},
    "M1820": {0: .045, 1: .040, 2: .641, 3: .274},
    "M1830": {0: .008, 1: .014, 2: .094, 3: .330, 4: .157, 5: .300, 6: .097},
    "M1840": {0: .300, 1: .294, 2: .300, 3: .026, 4: .080},
    "M1850": {0: .010, 1: .173, 2: .700, 3: .090, 4: .017, 5: .010},
    "M1860": {0: .010, 1: .018, 2: .131, 3: .662, 4: .090, 5: .061, 6: .028},
}
ORDINAL_MAX = {k: max(v) for k, v in MARGINALS.items()}
SN_VISIT_MEAN = 4.0      # skilled nursing visits in an early period (assumption, tuned for a ~7% LUPA rate)
LATE_VISIT_SCALE = 0.65  # late periods have fewer visits


@dataclass
class Scenario:
    """Error-process rates. Defaults are illustrative assumptions, not measured rates."""
    n_periods: int = 20000
    seed: int = 2026
    ai_draft_unreviewed: float = 0.40   # share of periods whose OASIS ADLs come from an unreviewed AI draft
    missed_comorbidity: float = 0.15    # chance each documented comorbidity code is left off the claim
    rtp_principal: float = 0.03         # principal replaced by a code CMS won't accept (claim returned)
    sequencing_error: float = 0.04      # principal swapped with a secondary from another clinical group
    unsupported_comorbidity: float = 0.02  # an unsupported low-comorbidity code added (over-coding)
    missed_visit_note: float = 0.05     # one visit not documented in time -> billed visit count - 1
    confusion_smoothing: float = 2.0    # pseudo-counts pulling sparse Second Read rows towards "correct"


_CONF_CACHE: dict = {}


def _confusion(smoothing: float) -> dict:
    key = smoothing
    if key in _CONF_CACHE:
        return _CONF_CACHE[key]
    src = json.loads((REF / "second_read_oasis_errors.json").read_text())["confusion"]
    out = {}
    for item in FI_ITEMS:
        rows = {}
        for t in range(ORDINAL_MAX[item] + 1):
            counts = {int(k): v for k, v in src.get(item, {}).get(str(t), {}).items() if k != "blank"}
            counts[t] = counts.get(t, 0) + smoothing                     # prior: mostly correct
            for adj in (t - 1, t + 1):                                     # small prior on off-by-one
                if 0 <= adj <= ORDINAL_MAX[item]:
                    counts[adj] = counts.get(adj, 0) + smoothing * 0.15
            vals, w = zip(*sorted(counts.items()))
            rows[t] = (np.array(vals), np.array(w, dtype=float) / sum(w))
        out[item] = rows
    _CONF_CACHE[key] = out
    return out


def _cut(z: float, marg: dict) -> int:
    u, acc = norm.cdf(z), 0.0
    for k, p in marg.items():
        acc += p
        if u <= acc:
            return k
    return max(marg)


def _true_oasis(rng: np.random.Generator) -> dict:
    mob = rng.standard_normal()
    selfc = 0.7 * mob + np.sqrt(1 - 0.49) * rng.standard_normal()
    a = {}
    for item, (lat, load) in {"M1800": (selfc, .8), "M1810": (selfc, .85), "M1820": (selfc, .85),
                              "M1830": (selfc, .8), "M1840": (mob, .8), "M1850": (mob, .9),
                              "M1860": (mob, .92)}.items():
        a[item] = _cut(load * lat + np.sqrt(1 - load ** 2) * rng.standard_normal(), MARGINALS[item])
    if a["M1860"] == 6 and a["M1850"] < 4:
        a["M1850"] = 4
    if a["M1850"] >= 4 and a["M1860"] != 6:
        a["M1850"] = 3
    frailty = norm.cdf(0.6 * mob + 0.4 * selfc)
    p = [0.30 + 0.25 * frailty, 0.20 + 0.15 * frailty, 0.30 + 0.20 * frailty, 0.25 + 0.15 * frailty,
         0.20, 0.17, 0.70]
    marked = [str(i + 1) for i, pi in enumerate(p) if rng.random() < min(0.97, pi * 1.31)]
    a["M1033"] = marked or ["10"]
    return a


def _visits(rng, therapy: float, early: bool) -> dict:
    scale = 1.0 if early else LATE_VISIT_SCALE
    v = {"sn": rng.poisson(SN_VISIT_MEAN * scale), "pt": rng.poisson(2.6 * therapy * scale),
         "ot": rng.poisson(0.8 * therapy * scale), "slp": rng.poisson(0.15 * therapy),
         "msw": rng.poisson(0.15), "aide": rng.poisson(0.6)}
    if sum(v.values()) == 0:
        v["sn"] = 1
    return {k: int(n) for k, n in v.items()}


def _secondaries(rng, pdx: str) -> list[str]:
    sec = [c for c, p in SECONDARY_POOL.items() if c != pdx and rng.random() < p]
    rng.shuffle(sec)
    return sec[:12]


def simulate(sc: Scenario = Scenario()) -> pd.DataFrame:
    rng = np.random.default_rng(sc.seed)
    conf = _confusion(sc.confusion_smoothing)
    gids, shares = zip(*[(g, p["share"]) for g, p in PROFILES.items()])
    shares = np.array(shares) / sum(shares)
    rows = []
    for i in range(sc.n_periods):
        gid = str(rng.choice(gids, p=shares))
        prof = PROFILES[gid]
        pdx = str(rng.choice(prof["pdx"]))
        early = rng.random() < 0.35
        inst = rng.random() < (0.55 if early else 0.12)
        truth = Period(pdx, _secondaries(rng, pdx), "early" if early else "late", inst, _true_oasis(rng),
                       _visits(rng, prof["therapy"], early), initial_period=early)
        billed = replace(truth, secondary_dx=list(truth.secondary_dx), oasis=dict(truth.oasis),
                         visits=dict(truth.visits))
        errors = []

        ai_drafted = rng.random() < sc.ai_draft_unreviewed
        if ai_drafted:
            changed = False
            for item in FI_ITEMS:
                vals, w = conf[item][truth.oasis[item]]
                new = int(rng.choice(vals, p=w))
                changed |= new != truth.oasis[item]
                billed.oasis[item] = new
            if changed:
                errors.append("oasis_ai_draft")
        kept = []
        for s in billed.secondary_dx:
            info = dx_info(s) or {}
            if info.get("COMORBIDITY_GROUP", "No_group") != "No_group" and rng.random() < sc.missed_comorbidity:
                errors.append("missed_comorbidity")
                continue
            kept.append(s)
        billed.secondary_dx = kept
        if rng.random() < sc.sequencing_error:
            others = [s for s in billed.secondary_dx
                      if (dx_info(s) or {}).get("CLINICAL_GROUP", "NA") not in ("NA", gid)]
            if others:
                s = str(rng.choice(others))
                billed.secondary_dx = [pdx] + [x for x in billed.secondary_dx if x != s]
                billed.principal_dx = s
                errors.append("sequencing_error")
        if rng.random() < sc.unsupported_comorbidity:
            extra = [c for c in UPCODE_POOL if c not in billed.secondary_dx and c != billed.principal_dx]
            if extra:
                billed.secondary_dx.append(str(rng.choice(extra)))
                errors.append("unsupported_comorbidity")
        if rng.random() < sc.missed_visit_note and sum(billed.visits.values()) > 1:
            d = max(billed.visits, key=billed.visits.get)
            billed.visits[d] -= 1
            errors.append("missed_visit_note")
        if rng.random() < sc.rtp_principal:
            billed.principal_dx = str(rng.choice(RTP_CODES))
            errors.append("rtp_principal")

        t, b = group(truth), group(billed)
        rows.append({
            "period_id": f"P{i:05d}", "clinical_group_true": t.clinical_group, "pdx_true": truth.principal_dx,
            "pdx_billed": billed.principal_dx, "timing": truth.timing, "institutional": truth.institutional,
            "n_secondary_true": len(truth.secondary_dx), "n_secondary_billed": len(billed.secondary_dx),
            "secondary_billed": "|".join(billed.secondary_dx),
            "visits_true": sum(truth.visits.values()), "visits_billed": sum(billed.visits.values()),
            **{f"{k}_true": truth.oasis[k] for k in FI_ITEMS}, **{f"{k}_billed": billed.oasis[k] for k in FI_ITEMS},
            "m1033_risks": len([x for x in truth.oasis["M1033"] if x != "10"]),
            "hipps_true": t.hipps, "hipps_billed": b.hipps, "return_code_billed": b.return_code,
            "level_true": t.functional_level, "level_billed": b.functional_level,
            "points_true": t.functional_points, "points_billed": b.functional_points,
            "comorbidity_true": t.comorbidity, "comorbidity_billed": b.comorbidity,
            "clinical_group_billed": b.clinical_group,
            "lupa_true": t.is_lupa, "lupa_billed": b.is_lupa, "lupa_threshold_billed": b.lupa_threshold,
            "payment_true": t.payment, "payment_billed": b.payment if b.return_code == "00" else 0.0,
            "rtp": b.return_code != "00", "ai_drafted": ai_drafted,
            "errors": "|".join(sorted(set(errors))),
        })
    df = pd.DataFrame(rows)
    # A paid claim's dollar error; returned claims are cash-flow delays, tracked separately.
    df["payment_delta"] = np.where(df.rtp, 0.0, (df.payment_billed - df.payment_true).round(2))
    df["revenue_error"] = (~df.rtp) & (df.payment_delta.abs() >= 0.01)
    return df


def scenario_dict(sc: Scenario) -> dict:
    return asdict(sc)
