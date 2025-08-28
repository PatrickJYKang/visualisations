#!/usr/bin/env python3
import re
import sys
from typing import List, Dict, Any

import pandas as pd

RAW_FILE = "big5_sca_gca_2025_raw.txt"
OUT_CSV = "big5_sca_gca_2025.csv"

COLS = [
    "Squad",
    "Comp",
    "# Pl",
    "90s",
    "SCA",
    "SCA90",
    "PassLive",
    "PassDead",
    "TO",
    "Sh",
    "Fld",
    "Def",
    "GCA",
    "GCA90",
    "PassLive_GCA",
    "PassDead_GCA",
    "TO_GCA",
    "Sh_GCA",
    "Fld_GCA",
    "Def_GCA",
]

COMP_KEYWORDS = [
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Ligue 1",
    "Serie A",
]


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def clean_line(s: str) -> str:
    return s.replace("\u00a0", " ").strip()


def parse_rows(lines: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if any(kw in line for kw in COMP_KEYWORDS):
            comp = line
            # walk back to find squad name (skip rank and headers)
            j = i - 1
            while j >= 0:
                prev = lines[j]
                if not prev or is_number(prev) or any(kw in prev for kw in COMP_KEYWORDS) or "RkSquadComp" in prev or "SCASCA" in prev or "Types" in prev:
                    j -= 1
                    continue
                break
            if j < 0:
                i += 1
                continue
            squad = lines[j]

            # collect 18 numeric fields
            vals: List[str] = []
            k = i + 1
            while k < n and len(vals) < 18:
                t = lines[k]
                if is_number(t):
                    vals.append(t)
                k += 1

            if len(vals) == 18:
                # coerce number types sensibly
                nums: List[Any] = []
                for v in vals:
                    try:
                        fv = float(v)
                        if fv.is_integer():
                            nums.append(int(fv))
                        else:
                            nums.append(fv)
                    except Exception:
                        nums.append(v)
                row = {
                    "Squad": squad,
                    "Comp": comp,
                    "# Pl": nums[0],
                    "90s": nums[1],
                    "SCA": nums[2],
                    "SCA90": nums[3],
                    "PassLive": nums[4],
                    "PassDead": nums[5],
                    "TO": nums[6],
                    "Sh": nums[7],
                    "Fld": nums[8],
                    "Def": nums[9],
                    "GCA": nums[10],
                    "GCA90": nums[11],
                    "PassLive_GCA": nums[12],
                    "PassDead_GCA": nums[13],
                    "TO_GCA": nums[14],
                    "Sh_GCA": nums[15],
                    "Fld_GCA": nums[16],
                    "Def_GCA": nums[17],
                }
                rows.append(row)
                i = k
                continue
        i += 1
    return rows


def main() -> None:
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        lines = [clean_line(l) for l in f.read().splitlines()]
    # drop obvious header lines and empties
    lines = [
        l
        for l in lines
        if l
        and not l.startswith("SCASCA")
        and "RkSquadComp" not in l
    ]

    rows = parse_rows(lines)
    if not rows:
        print("No rows parsed. Check input format.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
