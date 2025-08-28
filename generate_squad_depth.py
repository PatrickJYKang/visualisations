#!/usr/bin/env python3
"""
Squad depth visualiser (4-2-3-1) for Transfermarkt Kaggle dataset:
https://www.kaggle.com/datasets/davidcariboo/player-scores

Inputs:
  - players.csv
  - clubs.csv
  - By default, these are loaded via KaggleHub from the dataset above.

Usage:
  Using KaggleHub (default):
    python generate_squad_depth.py --team "Chelsea FC" --out chelsea_4231.png
  Using local CSVs:
    python generate_squad_depth.py --data-dir /path/to/csvs --team "Chelsea FC" --out chelsea_4231.png

Notes:
  - Uses 'sub_position' for placement.
  - Splits CBs and CMs/DMs evenly into left/right.
  - Sorts players by market value where available.
  - Exports a high-resolution PNG.
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Rectangle, Arc, FancyBboxPatch
import numpy as np
import pandas as pd

# Prefer a clean sans-serif font; will fallback if unavailable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

PITCH_W = 90.0  # width in arbitrary units (was 80)
PITCH_H = 120.0  # height in arbitrary units


# ------------------------------ helpers ------------------------------

def first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns are present: {candidates}")

def normalise_team_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def million_eur(x) -> str:
    if pd.isna(x):
        return ""
    try:
        val = float(x) / 1_000_000.0
        if val >= 100:
            return f"€{val:.0f}m"
        if val >= 10:
            return f"€{val:.1f}m"
        if val >= 1:
            return f"€{val:.2f}m"
        return f"€{val*1000:.0f}k"
    except Exception:
        return ""

def format_player(row: pd.Series, name_col: str, age_col: str, mv_col: str) -> str:
    name = str(row.get(name_col, "")).strip()
    age = row.get(age_col, None)
    age_txt = f"{int(age)}" if pd.notna(age) else ""
    # Do not display market value per request
    if age_txt:
        return f"{name} · {age_txt}"
    return name

def safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


# Format a readable club title from a club_code like "st-johnstone-fc" -> "St Johnstone FC"
def prettify_from_club_code(code: str) -> str:
    raw = str(code).strip()
    # Replace hyphens/underscores with spaces
    name = re.sub(r"[-_]+", " ", raw)
    # Tokenize and apply capitalization rules
    tokens = name.split()
    special_map = {
        "vfl": "VfL",
        "vfb": "VfB",
        "psg": "PSG",
        "rb": "RB",
        "bsc": "BSC",
        "tsv": "TSV",
        "ssv": "SSV",
        "rcd": "RCD",
        "ogc": "OGC",
        "om": "OM",
        "krc": "KRC",
        "tsg": "TSG",
    }
    always_upper = {
        "FC", "SV", "CF", "CD", "SC", "AC", "AS", "AFC", "UD",
        "FK", "IF", "BK", "SK", "US", "UC", "RC", "SD"
    }
    pretty = []
    for t in tokens:
        low = t.lower()
        if low in special_map:
            pretty.append(special_map[low])
            continue
        up = t.upper()
        if up in always_upper:
            pretty.append(up)
        else:
            # Standard word capitalization
            pretty.append(t.capitalize())
    return " ".join(pretty)


# ------------------------------ data prep ------------------------------

def load_and_filter(data_dir: Optional[str], team_name: str, dataset: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, str]]:
    players = None
    clubs = None

    # Prefer local CSVs if a data_dir is provided and the files exist
    if data_dir:
        players_path = os.path.join(data_dir, "players.csv")
        clubs_path = os.path.join(data_dir, "clubs.csv")
        if os.path.exists(players_path) and os.path.exists(clubs_path):
            players = pd.read_csv(players_path, low_memory=False)
            clubs = pd.read_csv(clubs_path, low_memory=False)

    # Otherwise, fetch via KaggleHub
    if players is None or clubs is None:
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
        except ImportError as e:
            raise FileNotFoundError(
                "players.csv and clubs.csv not found locally and kagglehub is not installed. "
                "Install with 'pip install kagglehub[pandas-datasets]' or provide --data-dir."
            ) from e
        players = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, dataset, "players.csv")
        clubs = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, dataset, "clubs.csv")

    # Column names vary across dataset versions. Resolve them.
    club_id_col_in_players = None
    for c in ["current_club_id", "club_id", "current_club_domestic_competition_id"]:
        if c in players.columns:
            if c == "current_club_domestic_competition_id":
                continue
            club_id_col_in_players = c
            break
    if club_id_col_in_players is None:
        raise KeyError("Could not find a club id column in players.csv (tried current_club_id, club_id).")

    club_id_col_in_clubs = first_present(clubs, ["club_id", "id"])
    club_name_col = first_present(clubs, ["name", "club_name"])

    # Map club name -> id using exact, case-insensitive match.
    clubs["_norm_name"] = clubs[club_name_col].astype(str).map(normalise_team_name)
    target = normalise_team_name(team_name)
    matched = clubs[clubs["_norm_name"] == target]
    if matched.empty:
        raise ValueError(f'Club "{team_name}" not found in clubs.csv (exact match).')

    club_id = int(matched.iloc[0][club_id_col_in_clubs])
    # Optional short code/slug for title building
    club_code_val = None
    if "club_code" in clubs.columns:
        val = matched.iloc[0]["club_code"]
        if pd.notna(val):
            club_code_val = str(val)

    # Filter players
    team_players = players[players[club_id_col_in_players] == club_id].copy()

    # Resolve common columns in players
    name_col = first_present(team_players, ["name", "player_name"])
    subpos_col = first_present(team_players, ["sub_position", "subpos", "sub_position_name"])
    # Age is optional in some dataset versions; ignore if absent
    age_col = None
    for c in ["age", "player_age"]:
        if c in team_players.columns:
            age_col = c
            break
    mv_col = None
    for c in ["market_value_in_eur", "market_value_eur", "market_value", "current_value"]:
        if c in team_players.columns:
            mv_col = c
            break

    # Optional: restrict to last_season == 2024 if column present
    last_season_col = None
    for c in ["last_season", "season", "last_season_id"]:
        if c in team_players.columns:
            last_season_col = c
            break
    if last_season_col:
        last_vals = pd.to_numeric(team_players[last_season_col], errors="coerce")
        team_players = team_players[last_vals == 2024]

    # Temporarily disabled: filter out players with market value below squad median
    # if mv_col:
    #     mv_vals = pd.to_numeric(team_players[mv_col], errors="coerce")
    #     mv_median = mv_vals.median()
    #     team_players = team_players[mv_vals >= mv_median]

    # Sorting proxy: prefer market value, then age ascending, then name
    sort_keys = []
    if mv_col:
        sort_keys.append((mv_col, False))
    if age_col:
        sort_keys.append((age_col, True))
    sort_keys.append((name_col, True))

    if sort_keys:
        by = [k for k, _ in sort_keys]
        ascending = [asc for _, asc in sort_keys]
        team_players = team_players.sort_values(by=by, ascending=ascending)

    meta_cols = {"name": name_col, "age": age_col, "mv": mv_col, "subpos": subpos_col}
    return team_players, {"club_id": club_id, "club_code": club_code_val}, meta_cols


def split_evenly(rows: List[pd.Series]) -> Tuple[List[pd.Series], List[pd.Series]]:
    left, right = [], []
    for i, r in enumerate(rows):
        (left if i % 2 == 0 else right).append(r)
    return left, right


def bucket_positions(df: pd.DataFrame, meta_cols: Dict[str, str]) -> Dict[str, List[pd.Series]]:
    sp = meta_cols["subpos"]
    buckets: Dict[str, List[pd.Series]] = {
        "GK": [],
        "RB": [],
        "LB": [],
        "CB": [],
        "DM": [],
        "CM": [],
        "AM": [],
        "RW": [],
        "LW": [],
        "CF": [],
    }

    mapping = {
        "Goalkeeper": "GK",
        "Right-Back": "RB",
        "Left-Back": "LB",
        "Centre-Back": "CB",
        "Defensive Midfield": "DM",
        "Central Midfield": "CM",
        "Attacking Midfield": "AM",
        "Right Winger": "RW",
        "Left Winger": "LW",
        "Right Midfield": "RW",
        "Left Midfield": "LW",
        "Second Striker": "CF",
        "Centre-Forward": "CF",
        "Striker": "CF",
    }

    for _, row in df.iterrows():
        raw = str(row.get(sp, "")).strip()
        pos = mapping.get(raw, None)
        if pos:
            buckets[pos].append(row)

    return buckets


# ------------------------------ drawing ------------------------------

def draw_pitch(ax: plt.Axes) -> None:
    # Pitch dimensions in arbitrary units (0..PITCH_W width, 0..PITCH_H height)
    ax.add_patch(Rectangle((0, 0), PITCH_W, PITCH_H, fill=False, lw=2, ec="white"))
    # Centre line (horizontal at halfway) and circle
    ax.plot([0, PITCH_W], [PITCH_H / 2, PITCH_H / 2], color="white", lw=1.5)
    ax.add_patch(Circle((PITCH_W / 2, PITCH_H / 2), 9.15, fill=False, ec="white", lw=1.5))
    # Penalty boxes (bottom)
    ax.add_patch(Rectangle((0.15 * PITCH_W, 0), 0.7 * PITCH_W, 0.225 * PITCH_H, fill=False, ec="white", lw=1.5))
    ax.add_patch(Rectangle((0.3 * PITCH_W, 0), 0.4 * PITCH_W, 0.075 * PITCH_H, fill=False, ec="white", lw=1.5))
    ax.add_patch(Arc((PITCH_W / 2, 0.225 * PITCH_H), 18.3, 18.3, theta1=0, theta2=180, ec="white", lw=1.5))
    # Penalty boxes (top)
    ax.add_patch(Rectangle((0.15 * PITCH_W, PITCH_H - 0.225 * PITCH_H), 0.7 * PITCH_W, 0.225 * PITCH_H, fill=False, ec="white", lw=1.5))
    ax.add_patch(Rectangle((0.3 * PITCH_W, PITCH_H - 0.075 * PITCH_H), 0.4 * PITCH_W, 0.075 * PITCH_H, fill=False, ec="white", lw=1.5))
    ax.add_patch(Arc((PITCH_W / 2, PITCH_H - 0.225 * PITCH_H), 18.3, 18.3, theta1=180, theta2=0, ec="white", lw=1.5))

    ax.set_xlim(0, PITCH_W)
    ax.set_ylim(0, PITCH_H)
    ax.axis("off")

def add_box(ax: plt.Axes, xy: Tuple[float, float], title: str, lines: List[str], max_lines: int = 8) -> None:
    x, y = xy
    width = 21
    height_per_line = 2.8
    header_h = 4
    n_lines = min(len(lines), max_lines)
    height = header_h + n_lines * height_per_line + 1.5

    card = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.4,rounding_size=2.2",
        fc="#111318",
        ec="#3b3f47",
        lw=1.2,
    )
    ax.add_patch(card)

    ax.add_patch(Rectangle((x - width / 2, y + height / 2 - header_h), width, header_h, fc="#222734", ec="#3b3f47", lw=1))

    ax.text(
        x,
        y + height / 2 - header_h / 2,
        title,
        va="center",
        ha="center",
        fontsize=10.5,
        color="#e2e6ef",
        weight="bold",
    )

    y0 = y + height / 2 - header_h - 0.8
    for i in range(n_lines):
        txt = lines[i]
        ax.text(
            x - width / 2 + 1.2,
            y0 - i * height_per_line,
            txt,
            va="top",
            ha="left",
            fontsize=9.8,
            color="#cfd5e1",
            path_effects=[pe.withStroke(linewidth=0.5, foreground="#0a0b0f")],
        )

def add_group(
    ax: plt.Axes,
    xy: Tuple[float, float],
    title: str,
    lines: List[str],
    *,
    max_lines: int = 12,
    title_fs: int = 20,
    line_fs: int = 16,
    line_gap: float = 3.6,
    shadow: bool = True,
) -> None:
    """Draw a centered heading with centered player names underneath, no card background.

    xy is treated as the TOP-CENTER anchor where the heading goes; player lines flow downward.
    """
    x, y_top = xy
    n = min(len(lines), max_lines)
    peff = [pe.withStroke(linewidth=0.6, foreground="#0a0b0f")] if shadow else None
    bbox = dict(facecolor="#000000", alpha=1.0, boxstyle="round,pad=0.25", edgecolor="none")

    # Title
    ax.text(
        x,
        y_top,
        title,
        va="top",
        ha="center",
        fontsize=title_fs,
        color="#e2e6ef",
        weight="bold",
        path_effects=peff,
        bbox=bbox,
    )

    # Player lines
    for i in range(n):
        ax.text(
            x,
            y_top - (i + 1) * line_gap,
            lines[i],
            va="top",
            ha="center",
            fontsize=line_fs,
            color="#cfd5e1",
            path_effects=peff,
            bbox=bbox,
        )

def build_layout_labels(buckets: Dict[str, List[pd.Series]], meta_cols: Dict[str, str]) -> List[Tuple[Tuple[float, float], str, List[str]]]:
    # Merge CBs into one group; keep DM and CM separate
    cb_all = buckets["CB"]
    dm_list = buckets["DM"]
    cm_list = buckets["CM"]

    name_col, age_col, mv_col = meta_cols["name"], meta_cols["age"], meta_cols["mv"]

    def fmt_list(rows: List[pd.Series]) -> List[str]:
        return [format_player(r, name_col, age_col, mv_col) for r in rows]

    # Dynamic vertical offsets to avoid overlap when there are many CBs or CFs
    LINE_GAP = 4.2
    MAX_LINES = 12
    MARGIN = 4.0

    # CF pushes AM down
    cf_top = 0.96 * PITCH_H
    cf_n = min(len(buckets["CF"]), MAX_LINES)
    cf_bottom = cf_top - cf_n * LINE_GAP
    am_base = 0.80 * PITCH_H
    am_top = min(am_base, cf_bottom - MARGIN)

    # CB pushes GK down
    cb_top = 0.40 * PITCH_H
    cb_n = min(len(cb_all), MAX_LINES)
    cb_bottom = cb_top - cb_n * LINE_GAP
    gk_base = 0.20 * PITCH_H
    gk_top = min(gk_base, cb_bottom - MARGIN)

    groups = [
        # Top-center anchors for headings; names flow downward
        ((0.50 * PITCH_W, gk_top), "GK", fmt_list(buckets["GK"])) ,
        ((0.17 * PITCH_W, 0.40 * PITCH_H), "LB", fmt_list(buckets["LB"])) ,
        ((0.50 * PITCH_W, 0.40 * PITCH_H), "CB", fmt_list(cb_all)) ,
        ((0.83 * PITCH_W, 0.40 * PITCH_H), "RB", fmt_list(buckets["RB"])) ,
        ((0.30 * PITCH_W, 0.60 * PITCH_H), "DM", fmt_list(dm_list)) ,
        ((0.70 * PITCH_W, 0.60 * PITCH_H), "CM", fmt_list(cm_list)) ,
        ((0.50 * PITCH_W, am_top), "AM", fmt_list(buckets["AM"])) ,
        ((0.17 * PITCH_W, 0.80 * PITCH_H), "LW", fmt_list(buckets["LW"])) ,
        ((0.83 * PITCH_W, 0.80 * PITCH_H), "RW", fmt_list(buckets["RW"])) ,
        ((0.50 * PITCH_W, cf_top), "CF", fmt_list(buckets["CF"])) ,
    ]
    return groups


# ------------------------------ main ------------------------------

def make_figure(team_name: str, data_dir: Optional[str], out_path: str, dataset: str = "davidcariboo/player-scores") -> None:
    df, _, meta_cols = load_and_filter(data_dir, team_name, dataset)
    buckets = bucket_positions(df, meta_cols)
    groups = build_layout_labels(buckets, meta_cols)

    plt.figure(figsize=(10.5, 12), facecolor="#0a0b0f")
    ax = plt.gca()
    ax.set_facecolor("#0a0b0f")
    ax.set_aspect('equal', adjustable='box')

    draw_pitch(ax)

    title_bbox = dict(facecolor="#000000", alpha=1.0, boxstyle="round,pad=0.25", edgecolor="none")
    # Prefer the shortened club_code for the title if available; fallback to provided team_name
    club_code = _["club_code"] if isinstance(_, dict) and "club_code" in _ else None
    if club_code:
        display_team = prettify_from_club_code(club_code)
    else:
        display_team = team_name.strip()
    max_chars = 35
    if len(display_team) > max_chars:
        display_team = display_team[: max_chars - 3].rstrip() + "..."
    ax.text(0.5, 1.03, f"{display_team} — Squad Depth", transform=ax.transAxes,
            ha="center", va="bottom", color="#e2e6ef", fontsize=24, weight="bold", bbox=title_bbox)
    ax.text(0.5, 1.01, "Data source: Transfermarkt via David Cariboo on Kaggle (https://www.kaggle.com/datasets/davidcariboo/player-scores)",
            transform=ax.transAxes, ha="center", va="bottom", color="#cfd5e1", fontsize=11, bbox=title_bbox)

    for xy, title, lines in groups:
        add_group(ax, xy, title, lines, max_lines=12, title_fs=20, line_fs=16, line_gap=3.6)

    # Footer removed; credits moved under the title

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Squad depth visualiser (4-2-3-1).")
    parser.add_argument("--data-dir", required=False, default=None, help="Folder containing players.csv and clubs.csv. If omitted, KaggleHub will be used to fetch 'davidcariboo/player-scores'.")
    parser.add_argument("--team", required=True, help="Exact club name as in clubs.csv")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--dataset", default="davidcariboo/player-scores", help="Kaggle dataset handle or version for KaggleHub (default: davidcariboo/player-scores)")
    args = parser.parse_args()

    out = args.out or f"{safe(args.team)}_4231.png"
    make_figure(args.team, args.data_dir, out, dataset=args.dataset)

if __name__ == "__main__":
    main()