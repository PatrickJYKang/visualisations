#!/usr/bin/env python3
"""
Unified CLI to generate club visuals:
 - SCA and GCA polar percentile charts (FBref Big-5 table)
 - Squad depth chart (Transfermarkt via KaggleHub)
 - Understat shot map

Resolves names from club_name_map_big5.csv (no hardcoded synonyms) and saves
artifacts to ./cache_images by default. Optionally includes SCA/GCA pie charts.
"""

import argparse
import os
import sys
import difflib
import unicodedata
import re
import subprocess
from typing import Optional, Tuple

import pandas as pd

# Reuse plotting and data helpers from the existing module
from plot_sca_gca_pies import (
    resolve_gca_csv_for_club,
    find_gca_csv,
    aggregate_types,
    plot_pie,
    plot_polar_bars,
    big5_type_column_map,
    compute_club_per90,
    percentiles_for_counts,
    percentiles_for_per90,
    _filter_complete_rows,
    LEAGUE_COMP_VALUES,
    slugify,
    parse_season_from_filename,
    SCA_COLS,
    GCA_COLS,
)

DEFAULT_MAPPING = os.path.join('.', 'club_name_map_big5.csv')
DEFAULT_OUTDIR = os.path.join('.', 'cache_images')
DEFAULT_CSVDIR = os.path.join('.', 'cache_csv')


def _normalize(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower()
    try:
        s = unicodedata.normalize('NFKD', s)
        s = s.encode('ascii', 'ignore').decode('ascii')
    except Exception:
        pass
    s = s.replace('&', 'and')
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s


def _best_match_row(user_input: str, df: pd.DataFrame) -> pd.Series:
    """Return the row in mapping CSV that best matches user_input across
    fbref_name, tm_name, and understat_name, using difflib on normalized names."""
    if df.empty:
        raise ValueError('Mapping CSV is empty')
    target = _normalize(user_input)

    def score_row(row) -> float:
        candidates = [row.get('fbref_name', ''), row.get('tm_name', ''), row.get('understat_name', '')]
        scores = []
        for c in candidates:
            if not c:
                continue
            scores.append(difflib.SequenceMatcher(None, target, _normalize(c)).ratio())
        return max(scores) if scores else 0.0

    scores = df.apply(score_row, axis=1)
    best_idx = int(scores.idxmax())
    return df.loc[best_idx]


def _choose_team_name(preferred: str, df: pd.DataFrame) -> str:
    """Pick the best-matching Team string present in the resolved GCA CSV."""
    if 'Team' not in df.columns:
        return preferred
    uniq = [t for t in df['Team'].dropna().astype(str).unique().tolist()]
    if not uniq:
        return preferred
    tgt = _normalize(preferred)

    def score(name: str) -> float:
        return difflib.SequenceMatcher(None, tgt, _normalize(name)).ratio()

    best = max(uniq, key=score)
    return best


def _resolve_big5_table(path_arg: str, season_end: str, fallback_dir: str = '.') -> str:
    if path_arg:
        if os.path.exists(path_arg):
            return path_arg
        raise FileNotFoundError(f"--big5-table not found: {path_arg}")
    # Try season-specific default
    if season_end:
        cand = os.path.join(fallback_dir, f"big5_sca_gca_{season_end}.csv")
        if os.path.exists(cand):
            return cand
    # Last resort: pick any big5_sca_gca_*.csv with largest season year
    pats = [p for p in os.listdir(fallback_dir) if re.match(r"big5_sca_gca_\d{4}\.csv$", p)]
    if pats:
        pats.sort()
        return os.path.join(fallback_dir, pats[-1])
    raise FileNotFoundError("No Big-5 table found. Provide --big5-table or place big5_sca_gca_<year>.csv in repo root.")


def generate_all_charts(
    *,
    user_club: str,
    season_end: str,
    mapping_csv: str,
    country_override: Optional[str],
    tier: Optional[str],
    csvdir: str,
    outdir: str,
    big5_table: Optional[str],
    percentile_metric: str,
    include_pies: bool,
    no_fetch: bool,
    force_fetch: bool,
    r_debug: bool,
    time_pause: int,
    retries: int,
    do_polar: bool = True,
    do_shot: bool = True,
    do_squad: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return paths to (sca_polar, gca_polar, squad_depth, shot_map).

    When a toggle is False, the corresponding chart is not generated and None is returned for its path.
    """
    # Load mapping and select best row
    map_df = pd.read_csv(mapping_csv)
    row = _best_match_row(user_club, map_df)
    fbref_name = str(row.get('fbref_name') or row.get('tm_name') or user_club)
    fbref_country = country_override or str(row.get('fbref_country_code') or '').strip() or None

    # Prefer existing local CSV based on the user's original club input to maximize matches
    # with historical filenames (e.g., "arsenal_2025_match_gca.csv" rather than "arsenal_fc_...").
    gca_csv: Optional[str] = None
    if not force_fetch:
        # Prefer cache directory only (no fallback to repo root)
        try:
            gca_csv = find_gca_csv(user_club, season_end, search_dir=csvdir)
        except Exception:
            gca_csv = None

    # If not found locally, resolve via standard pipeline.
    if not gca_csv:
        # Respect --no-fetch by skipping the Python scraper path entirely (requires country)
        # and erroring out if no local CSV exists.
        country_for_resolve = None if no_fetch else fbref_country
        gca_csv = resolve_gca_csv_for_club(
            club=fbref_name,
            season_end=season_end,
            country=country_for_resolve,
            tier=(tier or None),
            no_fetch=no_fetch,
            force_fetch=force_fetch,
            r_debug=r_debug,
            time_pause=time_pause,
            retries=retries,
            csv_outdir=csvdir,
        )

    # Determine season year for filenames
    season = season_end or parse_season_from_filename(gca_csv) or ''

    # Load CSV and pick Team name present in file
    df = pd.read_csv(gca_csv)
    team_in_csv = _choose_team_name(fbref_name, df)

    # Filter to team and FOR
    if 'Team' in df.columns:
        df = df[df['Team'] == team_in_csv]
    if 'ForAgainst' in df.columns:
        df = df[df['ForAgainst'].astype(str).str.upper() == 'FOR']
    if df.empty:
        raise ValueError(f"No matching rows for '{team_in_csv}' after filters.")

    # Preferred file slug from mapping CSV if available
    file_slug = str(row.get('file_slug') or '').strip()
    preferred_slug = file_slug if file_slug else slugify(team_in_csv)

    # Aggregate totals (only if needed for pies/polar)
    sca_sums = aggregate_types(df, SCA_COLS) if (include_pies or do_polar) else {}
    gca_sums = aggregate_types(df, GCA_COLS) if (include_pies or do_polar) else {}

    # Output naming
    slug = preferred_slug
    suffix = f"_{season}" if season else ''
    os.makedirs(outdir, exist_ok=True)

    # Titles
    title_sca = f"{team_in_csv} SCA Types{(' ' + season) if season else ''}"
    title_gca = f"{team_in_csv} GCA Types{(' ' + season) if season else ''}"

    # Optional pies
    if include_pies:
        out_sca_pie = os.path.join(outdir, f"{slug}{suffix}_sca_pie.png")
        out_gca_pie = os.path.join(outdir, f"{slug}{suffix}_gca_pie.png")
        plot_pie(sca_sums, title_sca, out_sca_pie)
        plot_pie(gca_sums, title_gca, out_gca_pie)

    out_sca_polar: Optional[str] = None
    out_gca_polar: Optional[str] = None
    if do_polar:
        # Polar charts require Big-5 table; restrict lookup to CSV cache directory
        table_path = _resolve_big5_table(big5_table or '', season, fallback_dir=csvdir)
        big5 = pd.read_csv(table_path)
        sca_cols_map = big5_type_column_map(big5, 'sca')
        gca_cols_map = big5_type_column_map(big5, 'gca')

        # League-only, fall back to all rows if none
        if 'Comp' in df.columns:
            league_df = df[df['Comp'].isin(LEAGUE_COMP_VALUES)]
            if league_df.empty:
                league_df = df
        else:
            league_df = df

        sca_df_complete = _filter_complete_rows(league_df, SCA_COLS)
        gca_df_complete = _filter_complete_rows(league_df, GCA_COLS)
        sca_counts = aggregate_types(sca_df_complete, SCA_COLS)
        gca_counts = aggregate_types(gca_df_complete, GCA_COLS)

        if percentile_metric == 'per90':
            sca_per90 = compute_club_per90(df, SCA_COLS)
            gca_per90 = compute_club_per90(df, GCA_COLS)
            sca_percentiles = percentiles_for_per90(big5, sca_cols_map, sca_per90)
            gca_percentiles = percentiles_for_per90(big5, gca_cols_map, gca_per90)
        else:
            sca_percentiles = percentiles_for_counts(big5, sca_cols_map, sca_counts)
            gca_percentiles = percentiles_for_counts(big5, gca_cols_map, gca_counts)

        out_sca_polar = os.path.join(outdir, f"{slug}{suffix}_sca_polar.png")
        out_gca_polar = os.path.join(outdir, f"{slug}{suffix}_gca_polar.png")
        plot_polar_bars(sca_counts, sca_percentiles, title_sca, out_sca_polar)
        plot_polar_bars(gca_counts, gca_percentiles, title_gca, out_gca_polar)

    # Understat shot map via CLI (uses mapping understat_name)
    understat_team = str(row.get('understat_name') or '').strip()
    out_shot: Optional[str] = None
    if do_shot:
        out_shot = os.path.join(outdir, f"{slug}{suffix}_shot_map.png")
        if understat_team:
            cmd = [sys.executable, 'plot_understat_shot_map.py', '--team', understat_team]
            if season:
                cmd += ['--season', str(int(season) - 1)]
            cmd += ['--out', out_shot, '--csvdir', csvdir]
            if no_fetch:
                cmd.append('--no-fetch')
            res = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.', capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Shot map generation failed (understat team='{understat_team}').\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        else:
            raise ValueError("understat_name missing in mapping CSV; cannot generate shot map.")

    # Squad depth via CLI (uses mapping tm_name)
    tm_team = str(row.get('tm_name') or '').strip()
    out_depth: Optional[str] = None
    if do_squad:
        out_depth = os.path.join(outdir, f"{slug}_4231.png")
        if tm_team:
            cmd = [sys.executable, 'generate_squad_depth.py', '--team', tm_team, '--out', out_depth]
            res = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.', capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Squad depth generation failed (tm team='{tm_team}').\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        else:
            raise ValueError("tm_name missing in mapping CSV; cannot generate squad depth chart.")

    return out_sca_polar, out_gca_polar, out_depth, out_shot


def main():
    p = argparse.ArgumentParser(
        description=(
            "Automate club visuals: resolve FBref match GCA CSV once (Python scraper -> local -> R), "
            "then generate SCA/GCA polar percentile charts, squad depth, and Understat shot map."
        )
    )
    p.add_argument('--club', required=True, help='Input club name (free-form). Will be matched to mapping CSV without hardcoded synonyms.')
    p.add_argument('--season', default='', help='Season end year, e.g., 2025. If omitted, parse from CSV filename.')
    p.add_argument('--mapping', default=DEFAULT_MAPPING, help='Path to club name mapping CSV (default: ./club_name_map_big5.csv)')
    p.add_argument('--country', default='', help='Override country code for FBref fetch (e.g., ENG, ITA). Default uses mapping CSV.')
    p.add_argument('--tier', default='', help="Optional league tier (e.g., '1st', '2nd')")
    p.add_argument('--outdir', default=DEFAULT_OUTDIR, help='Output directory for images (default: ./cache_images)')
    p.add_argument('--csvdir', default=DEFAULT_CSVDIR, help='Directory to cache/search CSVs (default: ./cache_csv)')
    p.add_argument('--big5-table', default='', help='Path to Big-5 squads table. Defaults to big5_sca_gca_<season>.csv if present.')
    p.add_argument('--percentile-metric', choices=['counts', 'per90'], default='counts', help='Polar radii basis: totals or per90')
    p.add_argument('--include-pies', action='store_true', help='Also generate SCA/GCA pie charts')
    p.add_argument('--no-fetch', action='store_true', help='Do not call R to fetch if missing; error instead')
    p.add_argument('--force-fetch', action='store_true', help='Force fetch even if a CSV exists locally')
    p.add_argument('--r-debug', action='store_true', help='Print R stdout/stderr when fetching via R')
    p.add_argument('--time-pause', type=int, default=6, help='Seconds to pause between R requests (default: 6)')
    p.add_argument('--retries', type=int, default=5, help='Max retries for R fetch (default: 5)')

    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.csvdir, exist_ok=True)

    try:
        sca_polar, gca_polar, squad_depth, shot_map = generate_all_charts(
            user_club=args.club,
            season_end=args.season,
            mapping_csv=args.mapping,
            country_override=(args.country.strip() or None),
            tier=(args.tier.strip() or None),
            csvdir=args.csvdir,
            outdir=args.outdir,
            big5_table=(args.big5_table.strip() or None),
            percentile_metric=args.percentile_metric,
            include_pies=args.include_pies,
            no_fetch=args.no_fetch,
            force_fetch=args.force_fetch,
            r_debug=args.r_debug,
            time_pause=args.time_pause,
            retries=args.retries,
        )
    except Exception as e:
        raise SystemExit(str(e))

    print('Saved:')
    print(' -', sca_polar)
    print(' -', gca_polar)
    print(' -', squad_depth)
    print(' -', shot_map)


if __name__ == '__main__':
    main()
