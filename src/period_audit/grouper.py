"""A Python re-implementation of the CMS Home Health PDGM grouper (HHGS v07.2.26 tables).

Logic follows the CMS "Home Health Grouper Software Installation and User Manual":
  1. period timing (early/late) + referral source (institutional if occurrence code 61/62)
  2. principal diagnosis validation and clinical group (with "primary awarding" promotion)
  3. functional impairment points from M1800-M1860 + M1033, level by clinical-group thresholds
  4. comorbidity adjustment from secondary diagnoses (high interaction > low subgroup > none),
     ignoring secondaries in the same ICD-10 subchapter as the principal diagnosis
  5. HIPPS code -> CY 2026 case-mix weight, LUPA threshold, payment

Validated against all CMS test claims shipped with the grouper (tests/test_grouper.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import pandas as pd

REF = Path(__file__).resolve().parents[2] / "data" / "reference"
FI_ITEMS = ["M1800", "M1810", "M1820", "M1830", "M1840", "M1850", "M1860"]
INSTITUTIONAL_SOURCES = {"61", "62"}
LEVEL_LETTER = {"Low": "A", "Medium": "B", "High": "C"}
UNGROUPABLE_HIPPS = "00000"  # what the CMS grouper writes when no HIPPS can be assigned


@dataclass
class Period:
    principal_dx: str
    secondary_dx: list[str] = field(default_factory=list)
    timing: str = "early"                 # "early" | "late"
    institutional: bool = False           # referral source within 14 days (occurrence code 61/62)
    oasis: dict = field(default_factory=dict)   # M1800..M1860 -> int, "M1033" -> list of marked responses
    visits: dict = field(default_factory=dict)  # discipline -> count, for LUPA pricing
    initial_period: bool = True           # only/first period in a sequence (LUPA add-on eligibility)


@dataclass
class GroupResult:
    hipps: str
    return_code: str
    validity_flag: str
    clinical_group: str | None = None
    clinical_group_name: str | None = None
    principal_used: str | None = None
    functional_points: int | None = None
    functional_level: str | None = None
    comorbidity: str | None = None        # "none" | "low" | "high"
    comorbidity_detail: list[str] = field(default_factory=list)
    weight: float | None = None
    lupa_threshold: int | None = None
    is_lupa: bool | None = None
    payment: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- reference tables
@lru_cache(maxsize=1)
def _dx() -> dict[str, dict]:
    df = pd.read_csv(REF / "diagnosis_codes_v07226.csv.gz", dtype=str, keep_default_na=False)
    return {r["DIAGNOSIS"]: r for r in df.to_dict("records")}


@lru_cache(maxsize=1)
def _groups() -> dict[str, dict]:
    df = pd.read_csv(REF / "clinical_groups_v07226.csv", dtype=str, keep_default_na=False)
    return {r["ID"]: r for r in df.to_dict("records")}


@lru_cache(maxsize=1)
def _low_groups() -> set[str]:
    df = pd.read_csv(REF / "comorbidity_groups_v07226.csv", dtype=str)
    return set(df.loc[df.LOW_COMORBIDITY == "1", "NAME"])


@lru_cache(maxsize=1)
def _interactions() -> set[frozenset]:
    df = pd.read_csv(REF / "comorbidity_interactions_v07226.csv", dtype=str)
    return {frozenset((a, b)) for a, b in zip(df.GROUP1, df.GROUP2)}


@lru_cache(maxsize=1)
def _fi_points() -> dict[tuple[str, int], int]:
    df = pd.read_csv(REF / "fi_responses_v07226.csv", dtype=str)
    return {(r.FIELD, int(r.RESPONSE)): int(r.RESPONSE_CATEGORY_POINT)
            for r in df.itertuples() if r.FIELD != "M1033"}


@lru_cache(maxsize=1)
def _m1033_rule() -> tuple[int, int]:
    df = pd.read_csv(REF / "fi_responses_v07226.csv", dtype=str)
    r = df[df.FIELD == "M1033"].iloc[0]
    return int(r.RESPONSE_LIMIT), int(r.RESPONSE_CATEGORY_POINT)


@lru_cache(maxsize=1)
def _code_first() -> dict[str, list[tuple[str, str | None]]]:
    """Code-first condition id -> list of (prefix, range_end) patterns, e.g. ^O994 or ^M0500 to M058A."""
    df = pd.read_csv(REF / "code_first_conditions_v07226.csv", dtype=str, keep_default_na=False)
    out = {}
    for r in df.itertuples():
        pats = []
        for alt in r.CONDITION.split("|"):
            alt = alt.strip().lstrip("^")
            if " to " in alt:
                lo, hi = [x.strip().lstrip("^") for x in alt.split(" to ")]
                pats.append((lo, hi))
            elif alt:
                pats.append((alt, None))
        out[r.ID] = pats
    return out


def matches_code_first(condition_id: str, code: str) -> bool:
    for lo, hi in _code_first().get(condition_id, []):
        if hi is None and code.startswith(lo):
            return True
        if hi is not None and lo <= code[:len(lo)] <= hi:
            return True
    return False


@lru_cache(maxsize=1)
def _case_mix() -> dict[str, dict]:
    df = pd.read_csv(REF / "case_mix_cy2026.csv", dtype={"hipps": str})
    return {r["hipps"]: r for r in df.to_dict("records")}


@lru_cache(maxsize=1)
def rates() -> dict:
    return json.loads((REF / "rates_cy2026.json").read_text())


def dx_info(code: str) -> dict | None:
    return _dx().get(normalize_dx(code))


def normalize_dx(code: str | None) -> str:
    return (code or "").replace(".", "").strip().upper()


# ----------------------------------------------------------------------------- grouping steps
def validate_principal(pdx: str, secondaries: list[str]) -> tuple[str, str]:
    """Return (return_code, validity_flag) for the principal diagnosis."""
    if not pdx:
        return "03", "07"
    info = dx_info(pdx)
    if info is None:
        return "03", "06"
    if info["UNACCEPTABLE_PDX"] == "1":
        return "03", "01"
    if info["MANIFESTATION_FLAG"] == "1":
        return "03", "02"
    if info["UNSPECIFIED_PDX"] == "1":
        return "03", "03"
    if info["ECOI_FLAG"] == "1":
        return "03", "04"
    # Code-first sequencing is informational (flag 05) and is reported even when the claim
    # cannot be grouped because the code has no clinical group (return code 05).
    cf = info["CODE_FIRST"]
    flag = "05" if cf not in ("", "0") and any(matches_code_first(cf, s) for s in secondaries) else "00"
    if info["CLINICAL_GROUP"] == "NA":
        return "05", flag
    return "00", flag


def functional_points(oasis: dict) -> tuple[int | None, str]:
    """Points from the CMS FI table. Returns (points, return_code)."""
    rc_by_item = {"M1800": "08", "M1810": "09", "M1820": "10", "M1830": "11",
                  "M1840": "12", "M1850": "13", "M1860": "14"}
    pts_table = _fi_points()
    m1033 = oasis.get("M1033")
    if m1033 is None:
        return None, "07"
    marked = {str(x) for x in m1033}
    if not marked <= {str(i) for i in range(1, 11)} or ("10" in marked and len(marked) > 1):
        return None, "07"
    total = 0
    for item in FI_ITEMS:
        v = oasis.get(item)
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None, rc_by_item[item]
        if (item, v) not in pts_table:
            return None, rc_by_item[item]
        total += pts_table[(item, v)]
    limit, pts = _m1033_rule()
    if len(marked & {str(i) for i in range(1, 8)}) >= limit:
        total += pts
    return total, "00"


def functional_level(points: int, group_id: str) -> str:
    g = _groups()[group_id]
    if points < int(g["LOW_POINT"]) + 1:
        return "Low"
    if points >= int(g["HIGH_POINT"]):
        return "High"
    return "Medium"


def comorbidity(principal: str, secondaries: list[str]) -> tuple[str, list[str]]:
    p_sub = (dx_info(principal) or {}).get("SUBCHAPTER")
    groups = []
    for s in secondaries:
        info = dx_info(s)
        if not info or info["COMORBIDITY_GROUP"] in ("", "No_group"):
            continue
        if info["SUBCHAPTER"] == p_sub:  # closely related to the principal: no adjustment
            continue
        groups.append(info["COMORBIDITY_GROUP"])
    uniq = sorted(set(groups))
    inter = _interactions()
    highs = [f"{a}+{b}" for a, b in combinations(uniq, 2) if frozenset((a, b)) in inter]
    if highs:
        return "high", highs
    lows = [g for g in uniq if g in _low_groups()]
    if lows:
        return "low", lows
    return "none", []


def lupa_payment(visits: dict, initial_period: bool) -> float:
    r = rates()
    pay = sum(r["per_visit"][d] * n for d, n in visits.items() if n)
    if initial_period:
        # add-on applies to the first SN, PT, SLP or OT visit (one discipline; SN takes precedence)
        for d in ("sn", "pt", "slp", "ot"):
            if visits.get(d):
                pay += r["per_visit"][d] * (r["lupa_addon_factor"][d] - 1)
                break
    return round(pay, 2)


def group(p: Period) -> GroupResult:
    secondaries = [normalize_dx(s) for s in p.secondary_dx if normalize_dx(s)]
    pdx = normalize_dx(p.principal_dx)
    # Primary awarding: a flagged code in the first secondary position replaces the principal.
    if secondaries and (dx_info(secondaries[0]) or {}).get("PRIMARY_AWARDING_FLAG") == "1":
        pdx, secondaries = secondaries[0], secondaries[1:]
    rc, vf = validate_principal(pdx, secondaries)
    if rc != "00":
        return GroupResult(UNGROUPABLE_HIPPS, rc, vf, principal_used=pdx)
    gid = dx_info(pdx)["CLINICAL_GROUP"]
    pts, frc = functional_points(p.oasis)
    if frc != "00":
        return GroupResult(UNGROUPABLE_HIPPS, frc, vf, clinical_group=gid, principal_used=pdx)
    level = functional_level(pts, gid)
    com, detail = comorbidity(pdx, secondaries)
    pos1 = {("early", False): "1", ("early", True): "2", ("late", False): "3", ("late", True): "4"}[
        (p.timing, p.institutional)]
    hipps = f"{pos1}{gid}{LEVEL_LETTER[level]}{ {'none': 1, 'low': 2, 'high': 3}[com] }1"
    cm = _case_mix()[hipps]
    res = GroupResult(hipps, "00", vf, gid, _groups()[gid]["DESCRIPTION"], pdx, pts, level, com, detail,
                      cm["weight"], int(cm["lupa_threshold"]))
    n_visits = sum(p.visits.values()) if p.visits else None
    if n_visits is None:
        res.payment = round(cm["weight"] * rates()["national_30_day_rate"], 2)
    else:
        res.is_lupa = n_visits < res.lupa_threshold
        res.payment = (lupa_payment(p.visits, p.initial_period) if res.is_lupa
                       else round(cm["weight"] * rates()["national_30_day_rate"], 2))
    return res
