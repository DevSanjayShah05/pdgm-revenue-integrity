"""Build data/reference/ from public CMS files (raw files live in data/raw/cms/, gitignored).

Sources
  * CMS Home Health Grouper Software v07.2.26 (Oct 2026) reference tables (RT/*.txt) and test claims
  * CY 2026 Final HH PDGM case-mix weights and LUPA thresholds (CMS CSV)
  * CY 2026 Final low/high comorbidity adjustment subgroups (CMS CSV), used to cross-check the grouper tables
  * CY 2026 HH PPS final rule (90 FR 55342): national 30-day rate (Table 14), per-visit rates (Table 16),
    LUPA add-on factors (section II.E.3.d). These tables are images in the rule, so values are transcribed.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cms"
GRP = RAW / "grouper" / "HomeHealthGrouperSoftware"
RT = GRP / "tables" / "Versions" / "07226" / "RT"
REF = ROOT / "data" / "reference"
FIX = ROOT / "tests" / "fixtures"

RATES = {
    "source": "CY 2026 HH PPS Final Rule, 90 FR 55342 (FR Doc 2025-21767)",
    "national_30_day_rate": 2038.22,
    "per_visit": {"aide": 80.12, "msw": 283.64, "ot": 194.74, "pt": 193.42, "sn": 176.96, "slp": 210.25},
    # First-visit add-on for LUPA periods that are the only/initial period (CY 2025 update, still in force).
    # OT uses the PT factor (CY 2022 HH PPS final rule).
    "lupa_addon_factor": {"sn": 1.7200, "pt": 1.6225, "slp": 1.6696, "ot": 1.6225},
}


def read_rt(name: str) -> pd.DataFrame:
    return pd.read_csv(RT / f"{name}.txt", sep="\t", dtype=str, encoding="latin-1", keep_default_na=False)


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    FIX.mkdir(parents=True, exist_ok=True)

    dx = read_rt("Diagnosis_Codes")
    keep = ["DIAGNOSIS", "DESCRIPTION", "CLINICAL_GROUP", "SUBCHAPTER", "COMORBIDITY_GROUP",
            "MANIFESTATION_FLAG", "PRIMARY_AWARDING_FLAG", "ECOI_FLAG", "CODE_FIRST",
            "UNACCEPTABLE_PDX", "UNSPECIFIED_PDX", "PREDECESSOR_DX"]
    dx[keep].to_csv(REF / "diagnosis_codes_v07226.csv.gz", index=False, compression="gzip")
    for name in ["Clinical_Groups", "Comorbidity_Groups", "Comorbidity_Interactions", "FI_Responses",
                 "Diagnosis_Subchapters", "Code_First_Conditions"]:
        read_rt(name).to_csv(REF / f"{name.lower()}_v07226.csv", index=False)

    rows = []
    with open(RAW / "pdgm" / "CY 2026 Final HH PDGM Case Mix Weights and LUPA Thresholds.csv",
              encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row and row[0].strip():
                rows.append({"hipps": row[0].strip(), "description": row[1].strip(), "source_timing": row[2].strip(),
                             "comorbidity": int(row[3]), "weight": float(row[4]), "lupa_threshold": int(row[5])})
    assert len(rows) == 432
    pd.DataFrame(rows).to_csv(REF / "case_mix_cy2026.csv", index=False)
    (REF / "rates_cy2026.json").write_text(json.dumps(RATES, indent=2))

    # Cross-check the grouper's low-comorbidity flags against the final-rule subgroup list.
    low_rule = pd.read_csv(next((RAW / "comorb").glob("*Low Comorbidity tab.csv")), skiprows=1)
    rule_names = {s.strip().replace(" ", "_").replace("Neoplasms_", "Neoplasm_") for s in low_rule.iloc[:, 0].dropna()}
    cg = read_rt("Comorbidity_Groups")
    grouper_low = set(cg.loc[cg.LOW_COMORBIDITY == "1", "NAME"])
    assert rule_names == grouper_low, (rule_names ^ grouper_low)

    for f in ["TestDataV07226.txt", "TestDataV07226_GRC.txt"]:
        shutil.copy(GRP / "test" / f, FIX / f)
    print(f"diagnosis codes={len(dx):,} case-mix rows={len(rows)} low comorbidity groups={len(grouper_low)} "
          f"interactions={len(read_rt('Comorbidity_Interactions'))}")


if __name__ == "__main__":
    main()
