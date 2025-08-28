#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for servers
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import subprocess
import tempfile
import textwrap
import math

# Global style: dark mode + Arial
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.edgecolor": "#444444",
    "font.family": "Arial",
    "font.size": 14,              # base font size
    "axes.titlesize": 22,         # default title size
    "figure.titlesize": 22,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
})


SCA_COLS = [
    "PassLive_SCA_Types",
    "PassDead_SCA_Types",
    "TO_SCA_Types",
    "Sh_SCA_Types",
    "Fld_SCA_Types",
    "Def_SCA_Types",
]

GCA_COLS = [
    "PassLive_GCA_Types",
    "PassDead_GCA_Types",
    "TO_GCA_Types",
    "Sh_GCA_Types",
    "Fld_GCA_Types",
    "Def_GCA_Types",
]

LABELS = [
    "Pass (live)",
    "Pass (dead)",
    "Take-ons",
    "Shot",
    "Fouled",
    "Defensive",
]

PALETTE = {
    "Pass (live)": "#1f77b4",   # blue
    "Pass (dead)": "#17becf",   # cyan
    "Take-ons": "#ff7f0e",     # orange
    "Shot": "#2ca02c",         # green
    "Fouled": "#d62728",       # red
    "Defensive": "#9467bd",    # purple
}

# Recognized league competition names in match logs for league-only per90 calc
LEAGUE_COMP_VALUES = {
    "Premier League",
    "La Liga",
    "Serie A",             # Italy (note: Brazil uses "Série A")
    "Bundesliga",
    "Ligue 1",
    # Additional domestic leagues beyond Big-5
    "Primeira Liga",       # Portugal
    "Liga Portugal",       # alt naming
    "Eredivisie",          # Netherlands
    "Major League Soccer", # USA
    "MLS",                 # short form sometimes used
    "Série A",             # Brazil (accented)
    "Liga MX",             # Mexico
    "Liga Profesional",    # Argentina
    "Pro League",          # Belgium (new branding)
    "First Division A",    # Belgium (older naming)
    # Prominent second tiers
    "Championship",        # England 2nd tier
    "Serie B",             # Italy 2nd tier
    "2. Bundesliga",       # Germany 2nd tier
    "Ligue 2",             # France 2nd tier
    "Segunda División",    # Spain 2nd tier
}


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.lower()).strip("_")


def parse_season_from_filename(path: str) -> str:
    # Try to extract e.g., _2025_match_gca.csv
    m = re.search(r"_(\d{4})_match_gca\\.csv$", os.path.basename(path))
    return m.group(1) if m else ""


def find_gca_csv(club: str, season: str = "", search_dir: str = ".") -> str:
    slug = slugify(club)
    candidates: List[str] = []
    if season:
        # Prefer exact '<slug>_<season>_match_gca.csv' (Python scraper naming)
        candidates = glob.glob(os.path.join(search_dir, f"{slug}_{season}_match_gca.csv"))
        if not candidates:
            # Next, allow extra tokens between slug and season
            candidates = glob.glob(os.path.join(search_dir, f"{slug}_*_{season}_match_gca.csv"))
        if not candidates:
            # permissive fallback (anywhere in name)
            candidates = glob.glob(os.path.join(search_dir, f"*{slug}*_{season}_match_gca.csv"))
    if not candidates:
        # Prefer explicit '<slug>_..._match_gca.csv'
        candidates = glob.glob(os.path.join(search_dir, f"{slug}_*_match_gca.csv"))
        if not candidates:
            # permissive fallback (anywhere in name)
            candidates = glob.glob(os.path.join(search_dir, f"*{slug}*_match_gca.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No GCA CSV found for club '{club}'. Looked for {slug}_*_match_gca.csv (and seasoned variants) in {search_dir}"
        )
    # Prefer one with a season-like 4-digit year; pick the highest year if multiple
    def extract_year(p: str) -> int:
        m = re.search(r"_(\d{4})_match_gca\\.csv$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    candidates.sort(key=extract_year, reverse=True)
    return candidates[0]


def _season_end_to_label(season: str) -> str:
    """Convert a season end year like '2025' to FBref label '2024-2025'.
    If already in 'YYYY-YYYY' form, return as-is.
    """
    s = str(season).strip()
    if not s:
        raise ValueError("season end year is required")
    if "-" in s:
        return s
    if len(s) == 4 and s.isdigit():
        end_y = int(s)
        return f"{end_y-1}-{end_y}"
    raise ValueError(f"Unrecognized season value: {season}")


def fetch_gca_via_python(
    club: str,
    season_end: str,
    *,
    country: str | None = None,
    outdir: str = "cache_csv",
) -> str | None:
    """Run our stdlib FBref scraper to save a CSV to root with consistent name.

    Returns the saved path or None on failure.
    """
    slug = slugify(club)
    try:
        season_label = _season_end_to_label(season_end)
    except Exception:
        return None
    out_path = os.path.join(outdir, f"{slug}_{season_end}_match_gca.csv")
    # Ensure destination directory exists
    os.makedirs(outdir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(__file__), "fbref_scrape_match_gca.py")
    cmd = [
        sys.executable,
        script_path,
        "--club",
        club,
        "--season",
        season_label,
        "--out",
        out_path,
    ]
    if country:
        cmd += ["--country", country]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if proc.stderr:
            print(proc.stderr.strip())
    except Exception as e:
        # Some environments may return a non-zero exit code even if the CSV was written.
        # If the expected output file exists and is non-empty, proceed with it.
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Python scraper exited non-zero for {club}, but CSV found at {out_path}; proceeding with it.")
            return out_path
        print(f"Python scraper failed for {club}: {e}")
        return None
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    return None


def resolve_gca_csv_for_club(
    *,
    club: str,
    season_end: str,
    country: str | None,
    tier: str | None,
    no_fetch: bool,
    force_fetch: bool,
    r_debug: bool,
    time_pause: int,
    retries: int,
    csv_outdir: str = "cache_csv",
) -> str:
    """Prefer Python scraper, then existing CSV in cache_csv, then R fetch.

    Returns a path to a non-empty CSV.
    """
    # 1) Python scraper (requires country and season)
    if season_end and country:
        # If force_fetch, attempt Python first regardless of existing files
        path_py = fetch_gca_via_python(club, season_end, country=country, outdir=csv_outdir)
        if path_py:
            return path_py
        else:
            print(f"Python scraper did not produce CSV for {club}; trying local CSV...")
    else:
        if not season_end:
            print("--season not provided; skipping Python scraper step.")
        if not country:
            print("--country not provided; skipping Python scraper step (requires country).")

    # 2) Existing CSV in cache_csv only
    try:
        path_local = find_gca_csv(club, season_end, search_dir=csv_outdir)
        if not force_fetch:
            return path_local
        else:
            print("Local CSV found in cache but --force-fetch set; will still attempt a fetch.")
    except FileNotFoundError:
        if no_fetch:
            raise
        path_local = ""

    # 3) Fetch via R
    print(f"Fetching via R for '{club}'...")
    # Ensure cache directory exists for R output
    os.makedirs(csv_outdir, exist_ok=True)
    path_r = fetch_gca_via_r(
        club,
        season_end,
        outdir=csv_outdir,
        country=country,
        tier=tier,
        r_debug=r_debug,
        time_pause=int(time_pause),
        retries=int(retries),
    )
    # If 2025 empty edge-case, retry 2024
    if os.path.exists(path_r) and os.path.getsize(path_r) == 0 and (season_end in ("", "2025")):
        print("Fetched CSV is empty; retrying previous season 2024 via R...")
        path_r = fetch_gca_via_r(
            club,
            "2024",
            outdir=csv_outdir,
            country=country,
            tier=tier,
            r_debug=r_debug,
            time_pause=int(time_pause),
            retries=int(retries),
        )
    return path_r


def generate_charts_for_club(
    *,
    club: str,
    season_end: str,
    country: str | None,
    tier: str | None,
    csvdir: str,
    outdir: str,
    chart_type: str,
    big5_table: str,
    percentile_metric: str,
    no_fetch: bool,
    force_fetch: bool,
    r_debug: bool,
    time_pause: int,
    retries: int,
) -> None:
    # Resolve CSV via Python -> local -> R
    gca_csv = resolve_gca_csv_for_club(
        club=club,
        season_end=season_end,
        country=country,
        tier=tier,
        no_fetch=no_fetch,
        force_fetch=force_fetch,
        r_debug=r_debug,
        time_pause=time_pause,
        retries=retries,
        csv_outdir=csvdir,
    )

    # Determine season string to append to filenames
    season = season_end or parse_season_from_filename(gca_csv) or ""

    # Load CSV
    df = pd.read_csv(gca_csv)

    # Filter to this team (prefer exact). If not found, try slug-equality; else skip team filter.
    if "Team" in df.columns:
        if (df["Team"] == club).any():
            df = df[df["Team"] == club]
        else:
            try:
                uniq = [t for t in df["Team"].dropna().unique().tolist()]
            except Exception:
                uniq = []
            target_slug = slugify(club)
            matches = [t for t in uniq if slugify(str(t)) == target_slug]
            if len(matches) == 1:
                print(f"Exact team name '{club}' not found; using CSV team name '{matches[0]}'")
                df = df[df["Team"] == matches[0]]
            else:
                print("Exact team name not found; proceeding without 'Team' filter.")
    if "ForAgainst" in df.columns:
        df = df[df["ForAgainst"].str.upper() == "FOR"]

    if df.empty:
        raise ValueError(f"No matching rows for {club} after filters.")

    # Aggregate breakdowns
    sca_sums = aggregate_types(df, SCA_COLS)
    gca_sums = aggregate_types(df, GCA_COLS)

    # Output filenames
    slug = slugify(club)
    suffix = f"_{season}" if season else ""
    if chart_type == "pie":
        out_sca = os.path.join(outdir, f"{slug}{suffix}_sca_pie.png")
        out_gca = os.path.join(outdir, f"{slug}{suffix}_gca_pie.png")
    else:
        out_sca = os.path.join(outdir, f"{slug}{suffix}_sca_polar.png")
        out_gca = os.path.join(outdir, f"{slug}{suffix}_gca_polar.png")

    # Titles
    title_sca = f"{club} SCA Types{f' {season}' if season else ''}"
    title_gca = f"{club} GCA Types{f' {season}' if season else ''}"

    # Plot
    if chart_type == "pie":
        plot_pie(sca_sums, title_sca, out_sca)
        plot_pie(gca_sums, title_gca, out_gca)
    else:
        if not big5_table:
            raise ValueError("--big5-table is required for --chart-type polar")
        big5 = pd.read_csv(big5_table)
        sca_cols_map = big5_type_column_map(big5, "sca")
        gca_cols_map = big5_type_column_map(big5, "gca")
        # For polar charts, align to league-only fixtures and exclude incomplete fixtures
        # If no recognized league rows found, fall back to all rows to avoid empty plots for non-Big-5 leagues.
        if "Comp" in df.columns:
            league_df = df[df["Comp"].isin(LEAGUE_COMP_VALUES)]
            if league_df.empty:
                league_df = df
        else:
            league_df = df
        sca_df_complete = _filter_complete_rows(league_df, SCA_COLS)
        gca_df_complete = _filter_complete_rows(league_df, GCA_COLS)
        sca_counts = aggregate_types(sca_df_complete, SCA_COLS)
        gca_counts = aggregate_types(gca_df_complete, GCA_COLS)

        if percentile_metric == "per90":
            sca_per90 = compute_club_per90(df, SCA_COLS)
            gca_per90 = compute_club_per90(df, GCA_COLS)
            sca_percentiles = percentiles_for_per90(big5, sca_cols_map, sca_per90)
            gca_percentiles = percentiles_for_per90(big5, gca_cols_map, gca_per90)
        else:
            sca_percentiles = percentiles_for_counts(big5, sca_cols_map, sca_counts)
            gca_percentiles = percentiles_for_counts(big5, gca_cols_map, gca_counts)
        plot_polar_bars(sca_counts, sca_percentiles, title_sca, out_sca)
        plot_polar_bars(gca_counts, gca_percentiles, title_gca, out_gca)


def _pattern_candidates_for_club(club: str) -> List[str]:
    """Return list of URL substring patterns derived solely from the input club name.
    No hardcoded club-specific synonyms; keeps behavior generic and input-driven.
    """
    base = re.sub(r"[^A-Za-z0-9]+", "-", club.lower()).strip("-")
    hyphen = "-".join(slugify(club).split("_"))
    compact = re.sub(r"[^A-Za-z0-9]+", "", club.lower())  # e.g., "fcporto"
    # If club name starts with a short prefix token (e.g., FC, SC, AC, CF), also try without it
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", club.lower()) if t]
    no_prefix = ""
    if len(tokens) > 1 and len(tokens[0]) <= 3:
        no_prefix = "-".join(tokens[1:])  # e.g., "porto"
    patterns = {base, hyphen, compact}
    if no_prefix:
        patterns.add(no_prefix)
        patterns.add(no_prefix.replace("-", ""))
    return [p for p in patterns if p]


def fetch_gca_via_r(
    club: str,
    season: str = "",
    outdir: str = ".",
    country: str | None = None,
    tier: str | None = None,
    *,
    r_debug: bool = False,
    time_pause: int = 6,
    retries: int = 5,
) -> str:
    """Call R (worldfootballR) to fetch a club's match GCA log and save CSV.
    Returns the saved CSV path. Tries a set of leagues and seasons.
    """
    patterns = _pattern_candidates_for_club(club)
    # R vectors
    r_patterns = ", ".join(f'"{p}"' for p in patterns)
    # Prefer provided country/tier; else search a small set
    if country:
        r_countries = f'"{country}"'
    else:
        r_countries = ", ".join(f'"{c}"' for c in ["ENG", "ITA", "GER", "ESP", "FRA", "NED", "POR", "BEL", "SCO"])
    if tier:
        r_tiers = f'"{tier}"'
    else:
        r_tiers = ", ".join(f'"{t}"' for t in ["1st", "2nd"])  # include 2nd for Sunderland
    club_escaped = club.replace("\\", "\\\\").replace("\"", "\\\"")
    season_num = season.strip()
    r_code = textwrap.dedent(f'''
        suppressPackageStartupMessages(library(worldfootballR))
        club <- "{club_escaped}"
        outdir <- "{outdir}"
        patterns <- c({r_patterns})
        countries <- c({r_countries})
        tiers <- c({r_tiers})
        years <- if ({'TRUE' if season_num else 'FALSE'}) c(as.integer({season_num})) else c(2025, 2024, 2023)
        slug <- gsub("[^A-Za-z0-9]+", "_", tolower(club))
        team_url <- NA_character_
        season_used <- NA_integer_
        find_hit <- function(urls, pats) {{
          lu_low <- tolower(urls)
          for (pat in pats) {{
            hits <- grepl(pat, lu_low, fixed = TRUE)
            if (any(hits)) return(urls[which(hits)[1]])
          }}
          return(NA_character_)
        }}
        for (yr in years) {{
          found <- FALSE
          for (ct in countries) {{
            for (tr in tiers) {{
              lu <- NULL
              for (i in seq_len({retries})) {{
                tmp <- try(fb_league_urls(country = ct, gender = "M", season_end_year = yr, tier = tr), silent = TRUE)
                if (!inherits(tmp, "try-error") && !is.null(tmp) && length(tmp) > 0) {{ lu <- tmp; break }}
                wait <- 8 * i
                message("league_urls try ", i, " for ", ct, "-", tr, " ", yr, ": sleeping ", wait, "s...")
                Sys.sleep(wait)
              }}
              if (is.null(lu) || length(lu) == 0) next
              urls <- NULL
              for (i in seq_len({retries})) {{
                tmp2 <- try(fb_teams_urls(lu), silent = TRUE)
                if (!inherits(tmp2, "try-error") && !is.null(tmp2) && length(tmp2) > 0) {{ urls <- tmp2; break }}
                wait <- 8 * i
                message("teams_urls try ", i, " for ", ct, "-", tr, " ", yr, ": sleeping ", wait, "s...")
                Sys.sleep(wait)
              }}
              if (is.null(urls) || length(urls) == 0) next
              tu <- find_hit(urls, patterns)
              if (!is.na(tu)) {{ team_url <- tu; season_used <- yr; found <- TRUE; break }}
            }}
            if (found) break
          }}
          if (found) break
        }}
        if (is.na(team_url)) stop(paste0("Team URL not found for ", club, " in seasons ", paste(years, collapse=",")))
        retry_gca <- function(u, tries = {retries}) {{
          for (i in seq_len(tries)) {{
            res <- try(fb_team_match_log_stats(team_urls = u, stat_type = "gca", time_pause = {time_pause}), silent = TRUE)
            if (!inherits(res, "try-error")) return(res)
            wait <- 8 * i
            message("Attempt ", i, "/", tries, " failed: ", conditionMessage(attr(res, "condition")), ". Sleeping ", wait, "s...")
            Sys.sleep(wait)
          }}
          stop("Failed to fetch GCA after retries")
        }}
        df <- retry_gca(team_url)
        out_file <- file.path(outdir, sprintf("%s_%d_match_gca.csv", slug, season_used))
        utils::write.csv(df, out_file, row.names = FALSE)
        cat("SAVED_FILE:", out_file, "\n")
    ''')
    # Write to a temp R file and run
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as tf:
        tf.write(r_code)
        r_path = tf.name
    try:
        proc = subprocess.run(["Rscript", r_path], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"R fetch failed for {club}: {e.stderr or e.stdout}")
    finally:
        try:
            os.remove(r_path)
        except Exception:
            pass
    if r_debug:
        # Print R stdout/stderr to help diagnose rate limits or matching issues
        if proc.stdout:
            print("[R stdout]", proc.stdout.strip())
        if proc.stderr:
            print("[R stderr]", proc.stderr.strip())
    saved_path = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SAVED_FILE:"):
            saved_path = line.split("SAVED_FILE:", 1)[1].strip()
            break
    if not saved_path or not os.path.exists(saved_path):
        raise RuntimeError(f"R fetch did not report a saved file for {club}. Output was:\n{proc.stdout}")
    return saved_path


def aggregate_types(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    sums = {}
    for col, label in zip(cols, LABELS):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            sums[label] = float(vals.sum(skipna=True))
        else:
            sums[label] = 0.0
    return sums


def _filter_complete_rows(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return only rows where all requested columns are present and numeric (not NA).

    Rules:
    - If any of the columns in `cols` is missing from df entirely, we treat it as missing for all rows
      and the result will be an empty DataFrame (no fixtures considered complete).
    - Values like "NA", "" or non-numeric strings are coerced to NaN and cause the row to be dropped.
    """
    # Ensure all columns exist
    for c in cols:
        if c not in df.columns:
            return df.iloc[0:0]
    # Build a mask where all cols are valid numbers
    masks = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        masks.append(s.notna())
    if not masks:
        return df
    complete_mask = masks[0]
    for m in masks[1:]:
        complete_mask = complete_mask & m
    return df[complete_mask].copy()


def plot_pie(data: Dict[str, float], title: str, out_path: str) -> None:
    # Filter zero categories
    items = [(k, v) for k, v in data.items() if v and v > 0]
    if not items:
        print(f"Nothing to plot for {title}; all zeros or missing.")
        return
    labels, values = zip(*items)
    colors = [PALETTE.get(lbl, None) for lbl in labels]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=220)
    wedges, _texts = ax.pie(
        values,
        labels=None,  # we'll use legend for labels
        colors=colors,
        autopct=None,  # we'll draw percentage labels outside manually
        startangle=140,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        textprops={"color": "white", "fontsize": 11},
    )
    ax.axis("equal")  # Equal aspect ratio ensures pie is drawn as a circle.

    # Title without box, with subtle outline for contrast
    title_text = ax.set_title(
        title,
        fontsize=22,
        pad=10,
        color="white",
    )
    title_text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    # Percent labels outside with connector lines
    total = sum(values)
    for wedge, val in zip(wedges, values):
        pct = 100.0 * val / total if total else 0.0
        if pct < 0.5:
            # Skip ultra-small slices to reduce clutter
            continue
        ang = 0.5 * (wedge.theta2 + wedge.theta1)
        rad = math.radians(ang)
        # Start point on the outer edge of the pie
        x = math.cos(rad)
        y = math.sin(rad)
        # End point slightly outside the pie radius
        r_label = 1.18
        tx = r_label * math.cos(rad)
        ty = r_label * math.sin(rad)
        ha = "left" if tx >= 0 else "right"
        ax.annotate(
            f"{pct:.1f}%",
            xy=(x, y),
            xytext=(tx, ty),
            ha=ha,
            va="center",
            color="white",
            fontsize=15,
            arrowprops=dict(arrowstyle="-", color="#9aa4b2", lw=1.0),
        )

    # Legend with counts (kept inside figure but to the right)
    counts = [int(round(v)) for v in values]
    legend_labels = [f"{l}: {c}" for l, c in zip(labels, counts)]
    ax.legend(
        wedges,
        legend_labels,
        title="Types",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        prop={"size": 14},
        title_fontsize=14,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _percentile_rank(series: pd.Series, value: float) -> float:
    s = _coerce_numeric(series)
    if len(s) == 0:
        return 0.0
    # Proportion of values <= value
    return float((s.le(value).sum() / s.count()) * 100.0)


def percentiles_for_counts(big5_df: pd.DataFrame, column_map: Dict[str, str], club_counts: Dict[str, float]) -> Dict[str, float]:
    """Compute counts-based percentiles for each label using a Big-5 squads table.

    column_map maps our LABELS to column names in big5_df.
    club_counts are the aggregated counts for the target club (same season).
    Returns {label -> percentile in [0,100]}.
    """
    out: Dict[str, float] = {}
    for label, col in column_map.items():
        if col not in big5_df.columns:
            # Try a loose match (case-insensitive) if exact not found
            candidates = [c for c in big5_df.columns if c.strip().lower() == col.strip().lower()]
            if candidates:
                col = candidates[0]
            else:
                out[label] = 0.0
                continue
        value = float(club_counts.get(label, 0.0) or 0.0)
        out[label] = _percentile_rank(big5_df[col], value)
    return out


def compute_club_per90(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """Compute per90 for each type from the club's match log using league-only rows.

    Falls back to all rows if no recognized league rows are present.
    Assumes each match approximates one 90, and excludes fixtures that lack any required
    column in `cols` (so we also "minus that 90").
    Returns mapping {label -> per90} aligned with LABELS.
    """
    use_df = df
    if "Comp" in df.columns:
        league_df = df[df["Comp"].isin(LEAGUE_COMP_VALUES)]
        if not league_df.empty:
            use_df = league_df
    # Drop fixtures that are missing any of the required type columns
    use_df = _filter_complete_rows(use_df, cols)
    # Number of 90s approximated by number of remaining fixtures
    n_matches = int(len(use_df))
    sums = aggregate_types(use_df, cols)
    per90: Dict[str, float] = {}
    for lbl in LABELS:
        val = float(sums.get(lbl, 0.0) or 0.0)
        per90[lbl] = (val / n_matches) if n_matches > 0 else 0.0
    return per90


def percentiles_for_per90(
    big5_df: pd.DataFrame,
    column_map: Dict[str, str],
    club_per90: Dict[str, float],
) -> Dict[str, float]:
    """Compute per90-based percentiles for each label using the Big-5 squads table.

    For each type column, build a Big-5 per90 series as counts / 90s, then
    percentile-rank the club's per90 value within that series.
    Returns {label -> percentile in [0,100]}.
    """
    # Locate the 90s column (case-insensitive match for "90s")
    n90_col = None
    for c in big5_df.columns:
        if str(c).strip().lower().replace(" ", "") == "90s":
            n90_col = c
            break
    if not n90_col:
        raise ValueError("Big-5 table is missing a '90s' column required for per90 percentiles.")

    n90 = _coerce_numeric(big5_df[n90_col])
    out: Dict[str, float] = {}
    for label, col in column_map.items():
        if col not in big5_df.columns:
            candidates = [c for c in big5_df.columns if c.strip().lower() == col.strip().lower()]
            if candidates:
                col = candidates[0]
            else:
                out[label] = 0.0
                continue
        counts = _coerce_numeric(big5_df[col])
        # Avoid division by zero -> NaN -> fill with 0
        per90_series = counts.div(n90.replace(0, pd.NA)).fillna(0)
        value = float(club_per90.get(label, 0.0) or 0.0)
        out[label] = _percentile_rank(per90_series, value)
    return out


def _find_anchor_index(cols: List[str], anchor: str) -> int:
    anc = anchor.strip().lower()
    for i, c in enumerate(cols):
        if str(c).strip().lower() == anc:
            return i
    # tolerate minor formatting differences
    anc2 = anc.replace(" ", "").replace("_", "")
    for i, c in enumerate(cols):
        cc = str(c).replace(" ", "").replace("_", "").lower()
        if cc == anc2:
            return i
    return -1


def big5_type_column_map(big5_df: pd.DataFrame, kind: str) -> Dict[str, str]:
    """Return a mapping of our LABELS -> column names in big5_df for SCA or GCA per-type counts.

    Tries in order:
    1) Unique column names (e.g., PassLive_GCA)
    2) Positional: columns immediately after the 'SCA90' or 'GCA90' anchors
    """
    cols = list(big5_df.columns)
    if kind == "gca":
        explicit = {
            "Pass (live)": "PassLive_GCA",
            "Pass (dead)": "PassDead_GCA",
            "Take-ons": "TO_GCA",
            "Shot": "Sh_GCA",
            "Fouled": "Fld_GCA",
            "Defensive": "Def_GCA",
        }
        if all(any(str(c).strip().lower() == v.lower() for c in cols) for v in explicit.values()):
            # map to the actual matching column names (case-insensitive)
            resolved: Dict[str, str] = {}
            for lbl, v in explicit.items():
                for c in cols:
                    if str(c).strip().lower() == v.lower():
                        resolved[lbl] = c
                        break
            return resolved
        # fallback: positional after GCA90
        anchor_idx = _find_anchor_index(cols, "GCA90")
        if anchor_idx >= 0 and anchor_idx + 6 < len(cols):
            seg = cols[anchor_idx + 1 : anchor_idx + 7]
            return {lbl: seg[i] for i, lbl in enumerate(LABELS)}
    else:  # kind == 'sca'
        explicit = {
            "Pass (live)": "PassLive",
            "Pass (dead)": "PassDead",
            "Take-ons": "TO",
            "Shot": "Sh",
            "Fouled": "Fld",
            "Defensive": "Def",
        }
        if all(any(str(c).strip().lower() == v.lower() for c in cols) for v in explicit.values()):
            # Prefer the block immediately after SCA90 if duplicates exist
            anchor_idx = _find_anchor_index(cols, "SCA90")
            if anchor_idx >= 0 and anchor_idx + 6 < len(cols):
                seg = cols[anchor_idx + 1 : anchor_idx + 7]
                # Double-check these look like our expected names (loose check)
                return {lbl: seg[i] for i, lbl in enumerate(LABELS)}
            # else fall back to first matches by name
            resolved: Dict[str, str] = {}
            for lbl, v in explicit.items():
                # choose the first appearance
                for c in cols:
                    if str(c).strip().lower() == v.lower():
                        resolved[lbl] = c
                        break
            return resolved
        # ultimate fallback: positional after SCA90
        anchor_idx = _find_anchor_index(cols, "SCA90")
        if anchor_idx >= 0 and anchor_idx + 6 < len(cols):
            seg = cols[anchor_idx + 1 : anchor_idx + 7]
            return {lbl: seg[i] for i, lbl in enumerate(LABELS)}

    # If all else fails, return an empty mapping; caller will handle zeros
    return {}


def plot_polar_bars(
    counts: Dict[str, float],
    percentiles: Dict[str, float],
    title: str,
    out_path: str,
) -> None:
    # Filter out zero-count categories to avoid zero-width bars
    items = [(k, counts.get(k, 0.0), float(percentiles.get(k, 0.0))) for k in LABELS]
    items = [(k, v, p) for (k, v, p) in items if v and v > 0]
    if not items:
        print(f"Nothing to plot for {title}; all zeros or missing.")
        return

    labels = [k for k, _, _ in items]
    values = [v for _, v, _ in items]
    percs = [p for _, _, p in items]  # 0..100
    total = sum(values)
    shares = [v / total for v in values]
    widths = [max(1e-3, s * 2 * math.pi) for s in shares]

    # Compute bar centers (theta) as cumulative positions like a pie
    thetas = []
    acc = 0.0
    for w in widths:
        thetas.append(acc + w / 2.0)
        acc += w

    # Radii scaled to percentiles (0..100)
    radii = percs

    fig = plt.figure(figsize=(7, 7), dpi=220)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#0e1117")

    # Bars
    bars = ax.bar(
        thetas,
        radii,
        width=widths,
        bottom=0.0,
        color=[PALETTE.get(lbl, "#888888") for lbl in labels],
        edgecolor="white",
        linewidth=1.2,
        align="center",
    )

    # Title with subtle outline
    title_text = ax.set_title(title, fontsize=22, pad=12, color="white")
    title_text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    # Radial limits: keep gridlines but hide labels
    ax.set_rlim(0, 100)
    ax.set_rticks([20, 40, 60, 80, 100])
    ax.grid(True, color="#444444", lw=0.8, alpha=0.5)
    ax.tick_params(colors="white", labelcolor=(0, 0, 0, 0))

    # Remove theta ticks/clutter; optionally show 0 at top
    ax.set_thetagrids([])

    # Annotate bars with share percentage near the top of each bar
    for bar, share, perc in zip(bars, shares, percs):
        theta = bar.get_x() + bar.get_width() / 2.0
        r = min(100.0, perc + 6.0)
        pct_text = f"{share * 100:.0f}%"
        txt = ax.text(theta, r, pct_text, color="white", ha="center", va="bottom", fontsize=13)
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])

    # Legend with counts (volume)
    legend_labels = [f"{lbl}: {int(round(val))}" for lbl, val in zip(labels, values)]
    ax.legend(
        bars,
        legend_labels,
        title="Types",
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        frameon=False,
        prop={"size": 14},
        title_fontsize=14,
    )

    # Data source text below the chart (figure coords) to avoid title overlap
    fig.text(
        0.5,
        0.03,
        "Data source: FBRef",
        ha="center",
        va="bottom",
        color="#9aa4b2",
        fontsize=10,
    )

    # Leave extra bottom space for the credit line
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SCA/GCA charts (pie or polar-percentile) for a club from FBref match logs. Python scraper first, fallback to local CSV, then R.")
    parser.add_argument("--club", default="", help="Club name (exact match in CSV 'Team'; also used to find file by slug). Not required when using --smoke.")
    parser.add_argument("--season", default="", help="Optional season end year (e.g., 2025). If omitted, the latest available file is used.")
    parser.add_argument(
        "--csv",
        default="",
        help=(
            "Optional direct path to a *_match_gca.csv file (override). "
            "By default, the script derives the filename from --club (e.g., 'nottingham_forest_'), "
            "searches locally, and if not found it will fetch and save via R unless --no-fetch is set."
        ),
    )
    parser.add_argument("--outdir", default="./cache_images", help="Directory to save the output images (default: ./cache_images)")
    parser.add_argument("--csvdir", default="./cache_csv", help="Directory to search for and save FBref GCA CSVs (default: ./cache_csv)")
    parser.add_argument("--no-fetch", action="store_true", help="Do not call R to fetch CSV if it is missing; error instead")
    parser.add_argument("--r-debug", action="store_true", help="Print R stdout/stderr when fetching via R")
    parser.add_argument("--time-pause", type=int, default=6, help="Seconds to pause between requests in worldfootballR (default: 6)")
    parser.add_argument("--retries", type=int, default=5, help="Max retries for R fetch (default: 5)")
    parser.add_argument("--country", default="", help="Optional FBref/worldfootballR country code (e.g., ENG, ITA, GER) to narrow R search")
    parser.add_argument("--tier", default="", help="Optional league tier (e.g., '1st', '2nd') to narrow R search")
    parser.add_argument("--force-fetch", action="store_true", help="Force fetching via R even if a CSV exists")
    parser.add_argument("--chart-type", choices=["pie", "polar"], default="pie", help="Chart type: 'pie' (default) or 'polar' (bar width=share, radius=percentile)")
    parser.add_argument("--big5-table", default="", help="Path to Big-5 squads CSV (required for --chart-type polar). Expected wide format: SCA, SCA90, then 6 SCA type columns; later GCA, GCA90, then 6 GCA type columns. *_GCA suffix optional.")
    parser.add_argument(
        "--percentile-metric",
        choices=["counts", "per90"],
        default="counts",
        help="Percentile basis for polar radii: 'counts' (totals) or 'per90' (league-only per90).",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a quick batch across multiple clubs/leagues saving to ./cache_images (uses Python scraper if country + season available)")
    args = parser.parse_args()
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    country = args.country.strip() or None
    tier = args.tier.strip() or None

    if args.smoke:
        samples = [
            ("Tottenham Hotspur", "ENG"),
            ("Bayern Munich", "GER"),
            ("Ajax", "NED"),
            ("Barcelona", "ESP"),
            ("Benfica", "POR"),
            ("Paris Saint-Germain", "FRA"),
        ]
        for club, ct in samples:
            try:
                generate_charts_for_club(
                    club=club,
                    season_end=args.season,
                    country=ct,
                    tier=tier,
                    csvdir=args.csvdir,
                    outdir=args.outdir,
                    chart_type=args.chart_type,
                    big5_table=args.big5_table,
                    percentile_metric=args.percentile_metric,
                    no_fetch=args.no_fetch,
                    force_fetch=args.force_fetch,
                    r_debug=args.r_debug,
                    time_pause=args.time_pause,
                    retries=args.retries,
                )
            except Exception as e:
                print(f"[smoke] {club}: {e}")
        return

    if not (args.club or args.csv):
        raise SystemExit("--club is required unless using --smoke or providing --csv")

    # If --csv provided, bypass fetch logic inside generator by passing no_fetch and direct path handling.
    if args.csv:
        # We'll temporarily point find_gca_csv to the provided path by loading directly within generator logic
        # Simplest: set club to provided club and rely on resolve to prefer local path; but to honor --csv, we temporarily run plotting from this path by
        # reading it here and constructing charts. To keep code simple, just set club and let generator resolve; but generator doesn't take csv path.
        # Instead, for explicit --csv, run a one-off lightweight flow: read and plot from the provided CSV.
        # Reuse generator pieces by mocking country/tier=None and skipping resolve.
        df = pd.read_csv(args.csv)
        club = args.club or (df["Team"].iloc[0] if "Team" in df.columns and not df.empty else "Club")
        # Fit minimal path via temp function
        season = args.season or parse_season_from_filename(args.csv) or ""
        if "Team" in df.columns and (df["Team"] == club).any():
            df = df[df["Team"] == club]
        if "ForAgainst" in df.columns:
            df = df[df["ForAgainst"].str.upper() == "FOR"]
        if df.empty:
            raise ValueError("--csv provided but no matching rows after filters.")
        sca_sums = aggregate_types(df, SCA_COLS)
        gca_sums = aggregate_types(df, GCA_COLS)
        slug = slugify(club)
        suffix = f"_{season}" if season else ""
        if args.chart_type == "pie":
            out_sca = os.path.join(args.outdir, f"{slug}{suffix}_sca_pie.png")
            out_gca = os.path.join(args.outdir, f"{slug}{suffix}_gca_pie.png")
            plot_pie(sca_sums, f"{club} SCA Types{f' {season}' if season else ''}", out_sca)
            plot_pie(gca_sums, f"{club} GCA Types{f' {season}' if season else ''}", out_gca)
        else:
            if not args.big5_table:
                raise ValueError("--big5-table is required for --chart-type polar")
            big5 = pd.read_csv(args.big5_table)
            sca_cols_map = big5_type_column_map(big5, "sca")
            gca_cols_map = big5_type_column_map(big5, "gca")

            # Align to league-only fixtures and exclude incomplete fixtures
            # If no recognized league rows found, fall back to all rows to avoid empty plots for non-Big-5 leagues.
            if "Comp" in df.columns:
                league_df = df[df["Comp"].isin(LEAGUE_COMP_VALUES)]
                if league_df.empty:
                    league_df = df
            else:
                league_df = df
            sca_df_complete = _filter_complete_rows(league_df, SCA_COLS)
            gca_df_complete = _filter_complete_rows(league_df, GCA_COLS)
            sca_counts = aggregate_types(sca_df_complete, SCA_COLS)
            gca_counts = aggregate_types(gca_df_complete, GCA_COLS)

            if args.percentile_metric == "per90":
                sca_per90 = compute_club_per90(df, SCA_COLS)
                gca_per90 = compute_club_per90(df, GCA_COLS)
                sca_percentiles = percentiles_for_per90(big5, sca_cols_map, sca_per90)
                gca_percentiles = percentiles_for_per90(big5, gca_cols_map, gca_per90)
            else:
                sca_percentiles = percentiles_for_counts(big5, sca_cols_map, sca_counts)
                gca_percentiles = percentiles_for_counts(big5, gca_cols_map, gca_counts)

            out_sca = os.path.join(args.outdir, f"{slug}{suffix}_sca_polar.png")
            out_gca = os.path.join(args.outdir, f"{slug}{suffix}_gca_polar.png")
            plot_polar_bars(sca_counts, sca_percentiles, f"{club} SCA Types{f' {season}' if season else ''}", out_sca)
            plot_polar_bars(gca_counts, gca_percentiles, f"{club} GCA Types{f' {season}' if season else ''}", out_gca)
        return

    # Normal single-club flow using Python->local->R resolution
    generate_charts_for_club(
        club=args.club,
        season_end=args.season,
        country=country,
        tier=tier,
        csvdir=args.csvdir,
        outdir=args.outdir,
        chart_type=args.chart_type,
        big5_table=args.big5_table,
        percentile_metric=args.percentile_metric,
        no_fetch=args.no_fetch,
        force_fetch=args.force_fetch,
        r_debug=args.r_debug,
        time_pause=args.time_pause,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
