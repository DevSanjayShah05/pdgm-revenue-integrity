"""Parse CMS HHGS fixed-width claim records (Claim_Layout.txt, v07.2.26)."""

from __future__ import annotations

from dataclasses import dataclass

from .grouper import FI_ITEMS, INSTITUTIONAL_SOURCES, Period


@dataclass
class CmsRecord:
    claim_id: str
    from_date: str
    through_date: str
    period: Period
    expected_hipps: str
    expected_validity: str
    expected_rc: str
    raw_timing: str
    raw_m1033: str


def parse(line: str) -> CmsRecord:
    line = line.rstrip("\r\n").ljust(700)
    timing = line[32:33]
    referral = line[33:35].strip()
    pdx = line[35:43].strip()
    sec = [line[43 + 8 * i: 51 + 8 * i].strip() for i in range(24)]
    flags = line[275:285]
    m1033 = [str(i + 1) for i, c in enumerate(flags) if c == "1"]
    oasis = {"M1033": m1033 if flags.strip() and set(flags) <= {"0", "1"} else None}
    for k, item in enumerate(FI_ITEMS):
        v = line[285 + 2 * k: 287 + 2 * k].strip()
        oasis[item] = int(v) if v.isdigit() else v
    p = Period(principal_dx=pdx, secondary_dx=[s for s in sec if s],
               timing={"1": "early", "2": "late"}.get(timing, "?"),
               institutional=referral in INSTITUTIONAL_SOURCES, oasis=oasis)
    return CmsRecord(line[0:24].strip(), line[24:32], line[349:357], p,
                     line[607:612].strip(), line[612:614].strip(), line[614:616].strip(), timing, flags)
