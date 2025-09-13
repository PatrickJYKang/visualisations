#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use headless backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Arc
import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

DEF_OUTDIR = os.path.join('.', 'cache_images')
DEF_CSVDIR = os.path.join('.', 'cache_csv')


def slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def ensure_csv(team: str,
               season: Optional[int],
               explicit_csv: Optional[str],
               allow_fetch: bool = True,
               csvdir: str = DEF_CSVDIR,
               rscript_bin: str = 'Rscript',
               r_exporter: str = 'understat_team_shots_to_csv.R') -> str:
    """Locate or generate the Understat shots CSV for a team/season.

    Order of resolution:
    1) explicit_csv if provided and exists
    2) default path in csv cache dir: <csvdir>/<slug>_<season>_understat_shots.csv (if season given)
    3) If no season: try to find latest matching <csvdir>/<slug>_*_understat_shots.csv
    4) If not found and allow_fetch: call the R exporter to create it (saving into <csvdir>)
    """
    if explicit_csv:
        if os.path.exists(explicit_csv):
            return explicit_csv
        raise FileNotFoundError(f"CSV not found at --csv {explicit_csv}")

    slug = slugify(team)
    os.makedirs(csvdir, exist_ok=True)

    # Helper to choose latest season if multiple CSVs exist
    def find_latest() -> Optional[str]:
        candidates = []
        for fname in os.listdir(csvdir):
            m = re.match(rf"{re.escape(slug)}_(\d{{4}})_understat_shots\.csv$", fname)
            if m:
                candidates.append((int(m.group(1)), os.path.join(csvdir, fname)))
        if not candidates:
            return None
        candidates.sort()
        return candidates[-1][1]

    if season is not None:
        default_path = os.path.join(csvdir, f"{slug}_{season}_understat_shots.csv")
        if os.path.exists(default_path):
            return default_path
    else:
        latest = find_latest()
        if latest and os.path.exists(latest):
            return latest

    if not allow_fetch:
        raise FileNotFoundError("Shots CSV not found and fetching is disabled (--no-fetch)")

    # Call R exporter to create the CSV
    cmd = [rscript_bin, r_exporter, '--team', team]
    if season is not None:
        cmd += ['--season', str(season)]
        out_path = os.path.join(csvdir, f"{slug}_{season}_understat_shots.csv")
    else:
        # If season unknown, still direct output into cache dir without year in filename
        out_path = os.path.join(csvdir, f"{slug}_understat_shots.csv")
    cmd += ['--out', out_path]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Try to parse reported output path
        reported_path = None
        for line in (proc.stdout + "\n" + proc.stderr).splitlines():
            line = line.strip()
            if line.startswith('Saved shots CSV:'):
                reported_path = line.split(':', 1)[1].strip()
                break
        # Prefer the reported path if it exists; otherwise, fall back to our intended out_path
        if reported_path and os.path.exists(reported_path):
            return reported_path
        if out_path and os.path.exists(out_path):
            return out_path
    except subprocess.CalledProcessError as e:
        # Fall through to check default path
        sys.stderr.write(f"R exporter failed: {e}\n{e.stdout or ''}{e.stderr or ''}\n")

    # Fallback: check default path(s) again
    if season is not None:
        default_path = os.path.join(csvdir, f"{slug}_{season}_understat_shots.csv")
        if os.path.exists(default_path):
            return default_path
    latest = find_latest()
    if latest and os.path.exists(latest):
        return latest

    raise FileNotFoundError("Unable to locate or generate Understat shots CSV")


def draw_pitch(ax, length: float = PITCH_LENGTH, width: float = PITCH_WIDTH):
    """
    Draw only the attacking half, rotated 90° clockwise.
    Rotated frame:
      x' = y (0..width), y' = length - x (0..length/2)
    """
    ax.set_facecolor('#000000')

    half_len = length / 2.0

    # Outer boundary (goal line at y'=0, touchlines x'=0 and x'=width, centre line at y'=half_len)
    ax.plot([0, width], [0, 0], color='white', linewidth=1)            # goal line
    ax.plot([0, width], [half_len, half_len], color='white', linewidth=1)  # centre line (top boundary)
    ax.plot([0, 0], [0, half_len], color='white', linewidth=1)         # left touch
    ax.plot([width, width], [0, half_len], color='white', linewidth=1)  # right touch

    # Centre circle (will be clipped to half)
    centre = (width / 2.0, half_len)
    ax.add_patch(Circle(centre, 9.15, fill=False, color='white', linewidth=1))

    # Penalty area (depth 16.5 from goal line)
    pen_top_y = 16.5
    pen_left_x = (width - 40.32) / 2.0
    pen_right_x = (width + 40.32) / 2.0
    ax.plot([pen_left_x, pen_right_x], [pen_top_y, pen_top_y], color='white', linewidth=1)
    ax.plot([pen_left_x, pen_left_x], [0, pen_top_y], color='white', linewidth=1)
    ax.plot([pen_right_x, pen_right_x], [0, pen_top_y], color='white', linewidth=1)

    # Six-yard box (depth 5.5 from goal line)
    six_top_y = 5.5
    six_left_x = (width - 18.32) / 2.0
    six_right_x = (width + 18.32) / 2.0
    ax.plot([six_left_x, six_right_x], [six_top_y, six_top_y], color='white', linewidth=1)
    ax.plot([six_left_x, six_left_x], [0, six_top_y], color='white', linewidth=1)
    ax.plot([six_right_x, six_right_x], [0, six_top_y], color='white', linewidth=1)

    # The 'D' (penalty arc) outside the penalty area
    # Arc centered at penalty spot (11m from goal), radius 9.15m, only the part outside the box
    d_center = (width / 2.0, 11.0)
    d_radius = 9.15
    theta = 36.87  # degrees; arcsin((16.5-11)/9.15) ≈ 36.87
    ax.add_patch(Arc(d_center, 2*d_radius, 2*d_radius, angle=0, theta1=theta, theta2=180-theta, color='white', linewidth=1))

    ax.set_xlim(0, width)
    ax.set_ylim(0, half_len)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_shot_map(csv_path: str,
                  out_path: str,
                  for_against: str = 'for',
                  title: Optional[str] = None) -> None:
    df = pd.read_csv(csv_path)

    # Filter by for/against if available; otherwise, infer by team columns if needed
    if 'for_against' in df.columns:
        key = for_against.lower()
        if key not in ('for', 'against'):
            key = 'for'
        dfp = df[df['for_against'].str.lower() == key].copy()
    else:
        dfp = df.copy()

    # Figure size that matches rotated half-pitch ratio (width : half-length)
    fig_w = 6.0
    fig_h = fig_w * (0.5 * PITCH_LENGTH) / PITCH_WIDTH
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    if not dfp.empty:
        # Understat normalized coordinates X in [0,1] of pitch length, Y in [0,1] of pitch width
        x_m = dfp['X'] * PITCH_LENGTH
        y_m = dfp['Y'] * PITCH_WIDTH

        # Keep only attacking half (x >= 52.5m) and rotate 90° clockwise
        half_mask = x_m >= (PITCH_LENGTH / 2.0)
        xr = y_m[half_mask]
        yr = (PITCH_LENGTH - x_m)[half_mask]

        # Weights by xG
        weights = pd.to_numeric(dfp.get('xG', dfp.get('xg', 0)), errors='coerce').fillna(0.0)[half_mask].to_numpy()

        # Build xG-weighted Gaussian heatmap on a grid
        grid_w = 300
        half_len = PITCH_LENGTH / 2.0
        grid_h = int(round(grid_w * (half_len / PITCH_WIDTH)))

        # Convert sigma from meters to pixels separately for x and y
        sigma_m = 0.75
        sigma_px_x = sigma_m * (grid_w - 1) / PITCH_WIDTH
        sigma_px_y = sigma_m * (grid_h - 1) / half_len
        rx = int(np.ceil(3 * sigma_px_x))
        ry = int(np.ceil(3 * sigma_px_y))

        ax_idx = np.arange(-rx, rx + 1)
        ay_idx = np.arange(-ry, ry + 1)
        gx = np.exp(-0.5 * (ax_idx / max(sigma_px_x, 1e-6))**2)
        gy = np.exp(-0.5 * (ay_idx / max(sigma_px_y, 1e-6))**2)
        kernel = np.outer(gy, gx)

        heat = np.zeros((grid_h, grid_w), dtype=float)

        # Map shot coords to grid indices
        ix = np.clip(np.round(xr.to_numpy() / PITCH_WIDTH * (grid_w - 1)).astype(int), 0, grid_w - 1)
        iy = np.clip(np.round(yr.to_numpy() / half_len * (grid_h - 1)).astype(int), 0, grid_h - 1)

        for k in range(len(ix)):
            cx, cy, w = ix[k], iy[k], weights[k]
            if w <= 0:
                continue
            x0 = max(cx - rx, 0)
            x1 = min(cx + rx, grid_w - 1)
            y0 = max(cy - ry, 0)
            y1 = min(cy + ry, grid_h - 1)

            kx0 = x0 - (cx - rx)
            kx1 = kx0 + (x1 - x0)
            ky0 = y0 - (cy - ry)
            ky1 = ky0 + (y1 - y0)

            heat[y0:y1 + 1, x0:x1 + 1] += w * kernel[ky0:ky1 + 1, kx0:kx1 + 1]

        # Dynamic range compression (cap at 99th percentile)
        if heat.max() > 0:
            vmax = float(np.percentile(heat[heat > 0], 99))
        else:
            vmax = None

        im_kwargs = dict(cmap='magma', origin='lower', extent=(0, PITCH_WIDTH, 0, half_len),
                         interpolation='bilinear', alpha=0.9)
        if vmax is not None and vmax > 0:
            im_kwargs['vmin'] = 0
            im_kwargs['vmax'] = vmax
        ax.imshow(heat, **im_kwargs)

        # Draw pitch lines then overlay goals as dots
        draw_pitch(ax)

        if 'result' in dfp.columns:
            goals_mask = dfp['result'].astype(str).str.lower().eq('goal') & (x_m >= (PITCH_LENGTH / 2.0))
            if goals_mask.any():
                xrg = (dfp.loc[goals_mask, 'Y'] * PITCH_WIDTH).to_numpy()
                yrg = (PITCH_LENGTH - dfp.loc[goals_mask, 'X'] * PITCH_LENGTH).to_numpy()
                ax.scatter(xrg, yrg, s=16, linewidths=0.6, edgecolors='black', facecolors='white', alpha=1.0)
    else:
        # No data: still draw pitch frame
        draw_pitch(ax)

    # Title and data source footer
    if title:
        plt.suptitle(title, color='white', fontsize=18, fontname='Arial')
    # Align data source with the left edge of the map (axes), just below the plot
    ax.text(0.0, -0.06, 'Data source: Understat via worldfootballR',
            transform=ax.transAxes, ha='left', va='top',
            color='#9aa4b2', fontsize=10, fontname='Arial', clip_on=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='#000000')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot a minimal Understat team shot map (points only).')
    parser.add_argument('--team', required=True, help='Understat team name, e.g., "Real Sociedad"')
    parser.add_argument('--season', type=int, default=None, help='Season start year, e.g., 2024. Defaults to latest available if CSV exists or R fetch is used.')
    parser.add_argument('--csv', default=None, help='Path to an existing Understat shots CSV to use.')
    parser.add_argument('--out', default=None, help='Output image path. Defaults to ./cache_images/<team_slug>_<season>_understat_shot_map.png')
    parser.add_argument('--for-against', default='for', help="'for' (team shots) or 'against' (conceded). Default: for")
    parser.add_argument('--no-fetch', action='store_true', help='Do not call R exporter if CSV is missing.')
    parser.add_argument('--rscript', default='Rscript', help='Rscript binary (if fetching). Default: Rscript')
    parser.add_argument('--r-exporter', default='understat_team_shots_to_csv.R', help='Path to R exporter script (if fetching).')
    parser.add_argument('--csvdir', default=DEF_CSVDIR, help='Directory to search for and cache Understat CSVs (default: ./cache_csv)')

    args = parser.parse_args()

    # Locate or generate CSV
    csv_path = ensure_csv(
        team=args.team,
        season=args.season,
        explicit_csv=args.csv,
        allow_fetch=(not args.no_fetch),
        csvdir=args.csvdir,
        rscript_bin=args.rscript,
        r_exporter=args.r_exporter,
    )

    # Resolve season for naming if not given
    season = args.season
    if season is None:
        try:
            df_tmp = pd.read_csv(csv_path, nrows=1)
            if 'season' in df_tmp.columns:
                season = int(df_tmp['season'].iloc[0])
        except Exception:
            pass

    slug = slugify(args.team)
    if args.out:
        out_path = args.out
    else:
        out_dir = DEF_OUTDIR
        os.makedirs(out_dir, exist_ok=True)
        if season is not None:
            out_path = os.path.join(out_dir, f"{slug}_{season}_understat_shot_map.png")
        else:
            out_path = os.path.join(out_dir, f"{slug}_understat_shot_map.png")

    # Build a descriptive title
    fa = str(args.for_against).lower()
    fa_label = 'For' if fa == 'for' else 'Against'
    if season is not None:
        title = f"{args.team} Shot Map ({fa_label}) {season}-{season + 1}"
    else:
        title = f"{args.team} Shot Map ({fa_label})"

    plot_shot_map(csv_path=csv_path, out_path=out_path, for_against=args.for_against, title=title)
    print(f"Saved shot map: {out_path}")


if __name__ == '__main__':
    main()
