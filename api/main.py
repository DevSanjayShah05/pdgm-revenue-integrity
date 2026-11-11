"""Period Audit API: group a 30-day period with CY 2026 PDGM, and score it for pre-bill QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from period_audit.grouper import Period, group
from period_audit.scoring import score_claim

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Period Audit - PDGM revenue integrity API", version="1.0.0",
              description="Research/portfolio tool. National rates, not wage-adjusted. Not for billing.")


class PeriodIn(BaseModel):
    principal_dx: str = Field(..., examples=["I5022"])
    secondary_dx: list[str] = Field(default_factory=list, examples=[["I480", "E1142", "F32A"]])
    timing: Literal["early", "late"] = "early"
    institutional: bool = False
    oasis: dict = Field(..., examples=[{"M1800": 2, "M1810": 2, "M1820": 2, "M1830": 3, "M1840": 1,
                                        "M1850": 2, "M1860": 3, "M1033": ["1", "3", "7"]}])
    visits: dict[str, int] = Field(default_factory=dict, examples=[{"sn": 5, "pt": 4, "ot": 1}])
    initial_period: bool = True
    ai_drafted: bool = False

    def to_period(self) -> Period:
        return Period(self.principal_dx, self.secondary_dx, self.timing, self.institutional,
                      self.oasis, self.visits, self.initial_period)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/group")
def group_period(p: PeriodIn) -> dict:
    return group(p.to_period()).to_dict()


@app.post("/score")
def score(p: PeriodIn) -> dict:
    try:
        return score_claim(p.to_period(), p.ai_drafted)
    except FileNotFoundError as e:
        raise HTTPException(503, "Run `make analyze` first to train the QA model.") from e


@app.get("/summary")
def summary() -> dict:
    return json.loads((ROOT / "data" / "outputs" / "summary.json").read_text())
