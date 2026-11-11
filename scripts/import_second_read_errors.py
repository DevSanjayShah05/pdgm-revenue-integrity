"""Import measured OASIS error patterns from Second Read (project 1) for the PDGM simulation.

Source: github.com/DevSanjayShah05/oasis-scribe-eval, data/results/decisions.csv, system
"qwen2.5:7b+segmented+rag+guard" (60 synthetic visits). Only answerable decisions are used.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "~/projects/oasis-scribe-eval/data/results/decisions.csv").expanduser()
OUT = Path(__file__).resolve().parents[1] / "data" / "reference" / "second_read_oasis_errors.json"
SYSTEM = "qwen2.5:7b+segmented+rag+guard"
ITEMS = ["M1800", "M1810", "M1820", "M1830", "M1840", "M1850", "M1860"]

d = pd.read_csv(SRC, keep_default_na=False)
d = d[(d.system == SYSTEM) & d.outcome.isin(["correct", "wrong", "abstained"])]
conf: dict = {}
for item in ITEMS:
    x = d[d["item"] == item]
    m = defaultdict(Counter)
    for e, p, o in zip(x.expected, x.predicted, x.outcome):
        e = json.loads(e)
        if o == "abstained":
            m[e]["blank"] += 1
        else:
            m[e][json.loads(p)] += 1
    conf[item] = {k: dict(v) for k, v in sorted(m.items())}
x = d[d["item"] == "M1033"]
deltas = Counter()
for e, p in zip(x.expected, x.predicted):
    e, p = json.loads(e), json.loads(p)
    if p is None:
        continue
    count = lambda v: len([r for r in v if r in {"1", "2", "3", "4", "5", "6", "7"}])
    deltas[count(p) - count(e)] += 1
out = {"source": f"oasis-scribe-eval decisions.csv, system={SYSTEM}", "n_visits": int(d.transcript_id.nunique()),
       "confusion": conf, "m1033_count_delta": {str(k): v for k, v in sorted(deltas.items())}}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out["m1033_count_delta"]), {k: sum(sum(c.values()) for c in v.values()) for k, v in conf.items()})
