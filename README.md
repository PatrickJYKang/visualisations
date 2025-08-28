# Club Visuals CLI

Generate four visuals for a club in one command:

- SCA polar percentile chart (Big-5)
- GCA polar percentile chart (Big-5)
- Squad depth chart (Transfermarkt via KaggleHub)
- Understat shot map

The CLI reads `club_name_map_big5.csv` for all name mappings (no hardcoded synonyms) and saves images to `./cache_images` by default.

## Quick start

- Arsenal 2025 (no network fetch; uses local files only):

```bash
python3 generate_club_charts.py --club "Arsenal" --season 2025 --no-fetch
```

- Include pies as well (optional):

```bash
python3 generate_club_charts.py --club "Arsenal" --season 2025 --include-pies
```

- Force re-fetch (may call R/remote sources; requires Rscript installed):

```bash
python3 generate_club_charts.py --club "Arsenal" --season 2025 --force-fetch
```

## Output files

Written to `./cache_images` by default (override with `--outdir`):

- `<slug>_<season>_sca_polar.png`
- `<slug>_<season>_gca_polar.png`
- `<slug>_4231.png` (squad depth)
- `<slug>_<season>_shot_map.png`
- Optional pies when `--include-pies`:
  - `<slug>_<season>_sca_pie.png`
  - `<slug>_<season>_gca_pie.png`

Where `<slug>` is derived from the resolved FBref team name found in the GCA CSV.

## Name mapping CSV

`club_name_map_big5.csv` is the single source of truth. Required columns:

- `tm_name`: Transfermarkt club name (used for squad depth)
- `fbref_name`: FBref team name (used to resolve GCA CSV)
- `fbref_country_code`: e.g., `ENG`, `ESP`, `GER`, `ITA`, `FRA`
- `understat_name`: ASCII Understat team name (used for shot map)

Keep `understat_name` ASCII only (no accents), as Understat expects that.

## Command reference

```bash
python3 generate_club_charts.py \
  --club "Team Name" \
  [--season 2025] \
  [--mapping ./club_name_map_big5.csv] \
  [--country ENG] \
  [--tier "1st"] \
  [--outdir ./cache_images] \
  [--big5-table ./big5_sca_gca_2025.csv] \
  [--percentile-metric counts|per90] \
  [--include-pies] \
  [--no-fetch] [--force-fetch] [--r-debug] \
  [--time-pause 6] [--retries 5]
```

- `--club`: Free-form input. We fuzzy-match against mapping CSV (`fbref_name`, `tm_name`, `understat_name`).
- `--season`: Season end year used for FBref/Understat files (e.g., 2025).
- `--country`: Override FBref country (else uses mapping `fbref_country_code`).
- `--big5-table`: Big-5 squads table CSV for percentile computation. Defaults to `big5_sca_gca_<season>.csv` if found.
- `--percentile-metric`: `counts` (totals; default) or `per90` when drawing polar radii.
- `--include-pies`: Also generate SCA/GCA pie charts.
- `--no-fetch`: Do not fetch missing data; only use local.
- `--force-fetch`: Fetch even if local files exist.
- `--r-debug`: Show R output when fetching via R.

## Web app (FastAPI)

This repo also includes a small web app for comparing two clubs side-by-side.

- Install deps:

```bash
pip install -r requirements.txt
```

- Run the server:

```bash
uvicorn app.main:app --reload
```

- Open: http://127.0.0.1:8000/

- Endpoints:
  - `GET /api/search?q=<query>` — fuzzy matches clubs from `club_name_map_big5.csv` (no hardcoded synonyms). Returns `{ name, file_slug, score }` suggestions.
  - `POST /api/generate` — body example:

```json
{
  "team_a": "Arsenal",
  "team_b": "Chelsea",
  "season": 2025,
  "include_pies": false,
  "percentile_metric": "counts",
  "no_fetch": true,
  "force_fetch": false
}
```

- Output static:
  - Generated images are served under `/images` from `./cache_images`.
  - App assets (CSS/JS) are served under `/static`.
  - Simple caching: if expected output files already exist, generation is skipped.

## Data sources and requirements

- FBref GCA match-by-match CSVs
  - Resolution: prefer local file; else Python scrape; else R fetch.
  - Controlled by `--no-fetch` and `--force-fetch`.
- Understat shot maps
  - Uses `plot_understat_shot_map.py`; if local `<understat_name>_<season>_understat_shots.csv` is missing and `--no-fetch` is not set, it uses an R exporter via `Rscript`.
  - Requires R installed when fetching.
- Squad depth
  - Uses `generate_squad_depth.py` which loads Kaggle dataset `davidcariboo/player-scores` via KaggleHub by default.
  - Install: `pip install kagglehub[pandas-datasets]`.
  - Alternatively pass local CSVs directly to `generate_squad_depth.py` with `--data-dir` (not exposed by the wrapper). You can also run that script directly if needed.

## Troubleshooting

- Missing Big-5 table: place `big5_sca_gca_<season>.csv` in repo root or pass `--big5-table`.
- Understat fetch blocked: use `--no-fetch` and ensure a local `<understat_name>_<season>_understat_shots.csv` exists.
- KaggleHub not installed: install via pip or run `generate_squad_depth.py --data-dir /path/to/csvs` separately.

## Related scripts

- `plot_sca_gca_pies.py` — data resolution, aggregation, pies, and polar plotting helpers.
- `plot_understat_shot_map.py` — Understat CSV resolution/fetch and shot map rendering.
- `generate_squad_depth.py` — Transfermarkt dataset loader and 4-2-3-1 squad depth visual.
- `build_club_name_map.py` — builds starter `club_name_map_big5.csv` from Kaggle Transfermarkt clubs.
