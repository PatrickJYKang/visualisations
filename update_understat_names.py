#!/usr/bin/env python3
"""
Fill understat_name for ENG and ESP clubs in club_name_map_big5.csv using provided mappings.

- Matches primarily on fbref_name; falls back to tm_name for a few common patterns.
- Only fills blanks (does not overwrite existing understat_name values).
- Keeps column order: [tm_name, fbref_name, fbref_country_code, understat_name].

Usage:
  python update_understat_names.py --file ./club_name_map_big5.csv
"""
from __future__ import annotations

import argparse
import sys
import pandas as pd
from typing import Dict


def load_csv(path: str) -> pd.DataFrame:
    # Prefer utf-8; fall back if necessary to handle special characters
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    except TypeError:
        # Older pandas without encoding_errors
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")


def build_mappings() -> Dict[str, Dict[str, str]]:
    # Understat names by fbref_name for England
    eng_by_fbref = {
        "Arsenal FC": "Arsenal",
        "Aston Villa FC": "Aston Villa",
        "AFC Bournemouth": "Bournemouth",
        "Brentford FC": "Brentford",
        "Brighton & Hove Albion FC": "Brighton",
        "Chelsea FC": "Chelsea",
        "Crystal Palace FC": "Crystal Palace",
        "Everton FC": "Everton",
        "Fulham FC": "Fulham",
        "Ipswich Town FC": "Ipswich",
        "Leicester City FC": "Leicester",
        "Liverpool FC": "Liverpool",
        "Luton Town FC": "Luton",
        "Manchester City FC": "Manchester City",
        "Manchester United FC": "Manchester United",
        "Newcastle United FC": "Newcastle United",
        "Nottingham Forest FC": "Nottingham Forest",
        "Tottenham Hotspur FC": "Tottenham",
        "West Ham United FC": "West Ham",
        "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    }

    # Fallback by tm_name for England (Transfermarkt formal names)
    eng_by_tm = {
        "Arsenal Football Club": "Arsenal",
        "Aston Villa Football Club": "Aston Villa",
        "Association Football Club Bournemouth": "Bournemouth",
        "Brentford Football Club": "Brentford",
        "Brighton and Hove Albion Football Club": "Brighton",
        "Chelsea Football Club": "Chelsea",
        "Crystal Palace Football Club": "Crystal Palace",
        "Everton Football Club": "Everton",
        "Fulham Football Club": "Fulham",
        "Ipswich Town Football Club": "Ipswich",
        "Leicester City Football Club": "Leicester",
        "Liverpool Football Club": "Liverpool",
        "Luton Town": "Luton",
        "Manchester City Football Club": "Manchester City",
        "Manchester United Football Club": "Manchester United",
        "Newcastle United Football Club": "Newcastle United",
        "Nottingham Forest Football Club": "Nottingham Forest",
        "Tottenham Hotspur Football Club": "Tottenham",
        "West Ham United Football Club": "West Ham",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers",
        "Wolverhampton Wanderers Football Club": "Wolverhampton Wanderers",
    }

    # Understat names by fbref_name for Spain
    esp_by_fbref = {
        "FC Barcelona": "Barcelona",
        "Real Madrid CF": "Real Madrid",
        "Atlético Madrid": "Atletico Madrid",
        "Atletico Madrid": "Atletico Madrid",
        "Athletic Club": "Athletic Club",
        "Villarreal CF": "Villarreal",
        "Real Betis Balompié": "Real Betis",
        "Real Betis Balompie": "Real Betis",
        "RC Celta de Vigo": "Celta Vigo",
        "CA Osasuna": "Osasuna",
        "Rayo Vallecano de Madrid": "Rayo Vallecano",
        "RCD Mallorca": "Mallorca",
        "Valencia CF": "Valencia",
        "Real Sociedad de Fútbol": "Real Sociedad",
        "Real Sociedad de Futbol": "Real Sociedad",
        "Getafe CF": "Getafe",
        "Deportivo Alavés": "Alaves",
        "Deportivo Alaves": "Alaves",
        "RCD Espanyol": "Espanyol",
        "Sevilla FC": "Sevilla",
        "Girona FC": "Girona",
        "CD Leganés": "Leganes",
        "CD Leganes": "Leganes",
        "UD Las Palmas": "Las Palmas",
        "Real Valladolid CF": "Real Valladolid",
    }

    # Fallback by tm_name for Spain (Transfermarkt)
    esp_by_tm = {
        "FC Barcelona": "Barcelona",
        "Real Madrid": "Real Madrid",
        "Club Atlético de Madrid": "Atletico Madrid",
        "Club Atletico de Madrid": "Atletico Madrid",
        "Athletic Club": "Athletic Club",
        "Villarreal CF": "Villarreal",
        "Real Betis Balompié": "Real Betis",
        "Real Betis Balompie": "Real Betis",
        "RC Celta de Vigo": "Celta Vigo",
        "CA Osasuna": "Osasuna",
        "Rayo Vallecano": "Rayo Vallecano",
        "RCD Mallorca": "Mallorca",
        "Valencia CF": "Valencia",
        "Real Sociedad": "Real Sociedad",
        "Getafe CF": "Getafe",
        "Deportivo Alavés": "Alaves",
        "Deportivo Alaves": "Alaves",
        "RCD Espanyol": "Espanyol",
        "Sevilla FC": "Sevilla",
        "Girona FC": "Girona",
        "CD Leganés": "Leganes",
        "CD Leganes": "Leganes",
        "UD Las Palmas": "Las Palmas",
        "Real Valladolid": "Real Valladolid",
    }

    return {
        "ENG_fbref": eng_by_fbref,
        "ENG_tm": eng_by_tm,
        "ESP_fbref": esp_by_fbref,
        "ESP_tm": esp_by_tm,
    }


def fill_understat_names(df: pd.DataFrame) -> pd.DataFrame:
    maps = build_mappings()
    # Ensure required columns exist
    for col in ["tm_name", "fbref_name", "fbref_country_code", "understat_name"]:
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column: {col}")

    def maybe_fill(row: pd.Series) -> str:
        current = str(row.get("understat_name") or "").strip()
        if current:
            return current  # keep existing
        country = str(row.get("fbref_country_code") or "").strip()
        fbref = str(row.get("fbref_name") or "").strip()
        tm = str(row.get("tm_name") or "").strip()
        if country == "ENG":
            if fbref in maps["ENG_fbref"]:
                return maps["ENG_fbref"][fbref]
            if tm in maps["ENG_tm"]:
                return maps["ENG_tm"][tm]
        if country == "ESP":
            if fbref in maps["ESP_fbref"]:
                return maps["ESP_fbref"][fbref]
            if tm in maps["ESP_tm"]:
                return maps["ESP_tm"][tm]
        return ""

    df["understat_name"] = df.apply(maybe_fill, axis=1)
    # Keep column order
    return df[["tm_name", "fbref_name", "fbref_country_code", "understat_name"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Populate understat_name for ENG and ESP clubs.")
    ap.add_argument("--file", required=True, help="Path to club_name_map_big5.csv to update in place")
    args = ap.parse_args()

    df = load_csv(args.file)
    before_blank = (df["understat_name"].isna() | (df["understat_name"].astype(str).str.strip() == "")).sum()
    df = fill_understat_names(df)
    after_blank = (df["understat_name"].astype(str).str.strip() == "").sum()

    # Save back preserving utf-8
    df.to_csv(args.file, index=False, encoding="utf-8")
    filled = before_blank - after_blank
    print(f"Updated {args.file}: filled {filled} understat_name values (remaining blanks: {after_blank})")


if __name__ == "__main__":
    main()
