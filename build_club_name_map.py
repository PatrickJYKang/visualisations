#!/usr/bin/env python3
"""
Build a cross-source club name mapping seed for the Big-5 leagues.

- Source of truth: Transfermarkt clubs.csv via KaggleHub (dataset: davidcariboo/player-scores).
- Output: CSV with columns [tm_name, fbref_name, understat_name, club_id, club_code, country, domestic_competition_id]
  where fbref_name and understat_name are left blank for manual filling.
- Filter: Only clubs whose domestic_competition_id is in the Big-5 set by default: {GB1, ES1, DE1, IT1, FR1}.

Usage examples:
  python build_club_name_map.py
  python build_club_name_map.py --dataset davidcariboo/player-scores --out ./club_name_map_big5.csv
  python build_club_name_map.py --leagues GB1 ES1 DE1 IT1 FR1 --out ./club_name_map_big5.csv

Notes:
- Requires kagglehub with pandas adapter: pip install kagglehub[pandas-datasets]
"""
from __future__ import annotations

import argparse
import sys
import re
from typing import List

import pandas as pd


def first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns are present: {candidates}")


def load_clubs(dataset: str) -> pd.DataFrame:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as e:
        raise SystemExit(
            "kagglehub is required. Install with: pip install kagglehub[pandas-datasets]"
        ) from e

    clubs = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, dataset, "clubs.csv")
    if not isinstance(clubs, pd.DataFrame) or clubs.empty:
        raise SystemExit("Loaded clubs.csv is empty or invalid.")
    return clubs


def build_map(clubs: pd.DataFrame, leagues: List[str]) -> pd.DataFrame:
    name_col = first_present(clubs, ["name", "club_name"])  # Transfermarkt club name

    # Competition column used to infer FBref country code
    comp_col = None
    for c in ["domestic_competition_id", "domestic_competition"]:
        if c in clubs.columns:
            comp_col = c
            break

    df = clubs.copy()
    if comp_col:
        df = df[df[comp_col].astype(str).isin(leagues)].copy()
    else:
        # If no competition column, keep as-is and warn
        print("[warn] No domestic_competition column found; emitting all clubs (no Big-5 filter).", file=sys.stderr)

    # Map TM domestic competition IDs to FBref country codes for Big-5
    big5_country_map = {
        "GB1": "ENG",  # Premier League
        "ES1": "ESP",  # La Liga
        "DE1": "GER",  # Bundesliga
        "L1": "GER",   # Some TM dumps use 'L1' for Bundesliga
        "IT1": "ITA",  # Serie A
        "FR1": "FRA",  # Ligue 1
    }

    out = pd.DataFrame()
    out["tm_name"] = df[name_col].astype(str)
    out["fbref_name"] = ""
    out["understat_name"] = ""
    if comp_col:
        comp_vals = df[comp_col].astype(str)
        out["fbref_country_code"] = comp_vals.map(big5_country_map).fillna("")
    else:
        out["fbref_country_code"] = ""

    # Drop duplicates by tm_name + fbref_country_code to keep one row per club per country
    out = out.drop_duplicates(subset=["tm_name", "fbref_country_code"])\
             .sort_values(by=["fbref_country_code", "tm_name"]).reset_index(drop=True)

    # Compute a default file_slug from fbref_name if present else tm_name
    def slugify(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    base_for_slug = out["fbref_name"].where(out["fbref_name"].astype(str).str.len() > 0, out["tm_name"])
    out["file_slug"] = base_for_slug.apply(slugify)

    # Ensure column order for manual editing
    out = out[["tm_name", "fbref_name", "fbref_country_code", "understat_name", "file_slug"]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build Big-5 club name mapping (tm_name, fbref_name, fbref_country_code, understat_name)")
    p.add_argument("--dataset", default="davidcariboo/player-scores", help="Kaggle dataset handle for kagglehub (default: davidcariboo/player-scores)")
    p.add_argument("--out", default="./club_name_map_big5.csv", help="Output CSV path (default: ./club_name_map_big5.csv)")
    p.add_argument(
        "--leagues",
        nargs="*",
        default=["GB1", "ES1", "DE1", "IT1", "FR1"],
        help="Domestic competition IDs to include (default: GB1 ES1 DE1 IT1 FR1)",
    )
    args = p.parse_args()

    clubs = load_clubs(args.dataset)
    out_df = build_map(clubs, args.leagues)
    out_df.to_csv(args.out, index=False)
    print(f"Saved mapping: {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
