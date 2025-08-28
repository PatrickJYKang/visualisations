#!/usr/bin/env python3
from __future__ import annotations

import os
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

# Local imports from the existing codebase
try:
    from plot_sca_gca_pies import slugify  # filename slug fallback
    import generate_club_charts as gcc
except Exception as e:
    raise RuntimeError(f"Failed to import project modules: {e}")

ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = ROOT / "club_name_map_big5.csv"
STATIC_OUT_DIR = ROOT / "cache_images"
STATIC_ASSETS_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"
CSV_CACHE_DIR = ROOT / "cache_csv"

app = FastAPI(title="Club Compare Web App", version="0.1.0")

# Static mounts
STATIC_OUT_DIR.mkdir(exist_ok=True)
STATIC_ASSETS_DIR.mkdir(exist_ok=True)
CSV_CACHE_DIR.mkdir(exist_ok=True)
# App assets (CSS/JS)
app.mount("/static", StaticFiles(directory=str(STATIC_ASSETS_DIR)), name="static")
# Generated images from ./cache_images
app.mount("/images", StaticFiles(directory=str(STATIC_OUT_DIR)), name="images")

# Templates
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------- Mapping Loader + Fuzzy Match ----------------------
_mapping_df: Optional[pd.DataFrame] = None
_mapping_mtime: float = 0.0

def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    try:
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    s = s.replace("&", "and")
    # keep only alnum to stabilize matching
    import re
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def load_mapping(force: bool = False) -> pd.DataFrame:
    global _mapping_df, _mapping_mtime
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping CSV not found at {MAPPING_PATH}")
    mtime = MAPPING_PATH.stat().st_mtime
    if _mapping_df is None or force or (mtime != _mapping_mtime):
        _mapping_df = pd.read_csv(MAPPING_PATH)
        _mapping_mtime = mtime
    return _mapping_df

def score_row(q_norm: str, row: pd.Series) -> float:
    candidates = [row.get("fbref_name", ""), row.get("tm_name", ""), row.get("understat_name", "")]
    best = 0.0
    for c in candidates:
        if not c:
            continue
        cand_norm = _normalize(c)
        sc = difflib.SequenceMatcher(None, q_norm, cand_norm).ratio()
        if sc > best:
            best = sc
    return best

def best_match_row(user_input: str) -> pd.Series:
    df = load_mapping()
    if df.empty:
        raise HTTPException(status_code=500, detail="Mapping CSV is empty.")
    q_norm = _normalize(user_input)
    scores = df.apply(lambda r: score_row(q_norm, r), axis=1)
    idx = int(scores.idxmax())
    return df.loc[idx]

# ------------------------------ Schemas ------------------------------
class GeneratePayload(BaseModel):
    team_a: str
    team_b: Optional[str] = None
    season: Optional[int] = 2025
    include_pies: bool = False
    percentile_metric: str = "per90"  # fixed by UI/backend
    no_fetch: bool = False
    force_fetch: bool = False
    country: Optional[str] = None
    tier: Optional[str] = None
    # New generation toggles (default enabled)
    gen_polar: bool = True
    gen_shot: bool = True
    gen_squad: bool = True

# ------------------------------ Helpers ------------------------------

def _resolve_big5_local(season: Optional[int]) -> Optional[Path]:
    """Find Big-5 table CSV. Prefer cache_csv/<season>, else any in cache_csv,
    else <repo root>/<season>, else any in repo root. Return best match or None."""
    # Exact season match in cache dir or repo root
    if season:
        for base in (CSV_CACHE_DIR, ROOT):
            p = base / f"big5_sca_gca_{season}.csv"
            if p.exists():
                return p
    # Any matching file in cache dir or repo root (pick latest season by filename)
    candidates: List[Path] = []
    for base in (CSV_CACHE_DIR, ROOT):
        candidates.extend(sorted(base.glob("big5_sca_gca_*.csv")))
    if candidates:
        def _year_key(p: Path) -> int:
            name = p.stem  # e.g., big5_sca_gca_2025
            parts = name.split("_")
            try:
                return int(parts[-1])
            except Exception:
                return -1
        candidates.sort(key=_year_key, reverse=True)
        return candidates[0]
    return None

def _expected_paths(team_name: str, season: Optional[int]) -> Dict[str, Path]:
    row = best_match_row(team_name)
    base_name = (
        (str(row.get("file_slug")).strip() if str(row.get("file_slug", "")).strip() else None)
        or str(row.get("fbref_name") or row.get("tm_name") or team_name)
    )
    s = slugify(base_name)
    suff = f"_{season}" if season else ""
    return {
        "sca_polar": STATIC_OUT_DIR / f"{s}{suff}_sca_polar.png",
        "gca_polar": STATIC_OUT_DIR / f"{s}{suff}_gca_polar.png",
        "shot_map": STATIC_OUT_DIR / f"{s}{suff}_shot_map.png",
        "squad_depth": STATIC_OUT_DIR / f"{s}_4231.png",
        "sca_pie": STATIC_OUT_DIR / f"{s}{suff}_sca_pie.png",
        "gca_pie": STATIC_OUT_DIR / f"{s}{suff}_gca_pie.png",
    }


def _all_exist(paths: List[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _to_url(p: Path) -> str:
    return f"/images/{p.name}"


def generate_for_team(
    team_name: str,
    season: Optional[int],
    *,
    include_pies: bool,
    percentile_metric: str,
    no_fetch: bool,
    force_fetch: bool,
    country: Optional[str],
    tier: Optional[str],
    gen_polar: bool,
    gen_shot: bool,
    gen_squad: bool,
) -> Dict[str, Any]:
    expected = _expected_paths(team_name, season)
    required: List[Path] = []
    if gen_polar:
        required += [expected["sca_polar"], expected["gca_polar"]]
    if gen_shot:
        required.append(expected["shot_map"])
    if gen_squad:
        required.append(expected["squad_depth"])

    # If nothing requested, return minimal payload
    if not (gen_polar or gen_shot or gen_squad):
        return {"team": team_name}

    # Simple cache: if every requested output already exists, skip regeneration
    if season and required and _all_exist(required):
        result: Dict[str, Any] = {"team": team_name}
        if gen_polar:
            result.update({
                "sca_polar_url": _to_url(expected["sca_polar"]),
                "gca_polar_url": _to_url(expected["gca_polar"]),
            })
        if gen_shot:
            result["shot_map_url"] = _to_url(expected["shot_map"])
        if gen_squad:
            result["squad_depth_url"] = _to_url(expected["squad_depth"])
        if include_pies and _all_exist([expected["sca_pie"], expected["gca_pie"]]):
            result.update({
                "sca_pie_url": _to_url(expected["sca_pie"]),
                "gca_pie_url": _to_url(expected["gca_pie"]),
            })
        return result

    # Otherwise, call the orchestrator with toggles
    big5_path = _resolve_big5_local(season)
    if gen_polar and not big5_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Big-5 table not found. Please place 'big5_sca_gca_" + str(season or "<season>") + 
                ".csv' in './cache_csv' or the repo root."
            ),
        )
    try:
        sca_polar, gca_polar, squad_depth, shot_map = gcc.generate_all_charts(
            user_club=team_name,
            season_end=str(season or ""),
            mapping_csv=str(MAPPING_PATH),
            country_override=country,
            tier=tier,
            csvdir=str(CSV_CACHE_DIR),
            outdir=str(STATIC_OUT_DIR),
            big5_table=(str(big5_path) if big5_path else None),
            percentile_metric=percentile_metric,
            include_pies=include_pies,
            no_fetch=no_fetch,
            force_fetch=force_fetch,
            r_debug=False,
            time_pause=6,
            retries=5,
            do_polar=gen_polar,
            do_shot=gen_shot,
            do_squad=gen_squad,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Generation failed for '{team_name}': {e}")

    result: Dict[str, Any] = {"team": team_name}
    if sca_polar:
        result["sca_polar_url"] = _to_url(Path(sca_polar))
    if gca_polar:
        result["gca_polar_url"] = _to_url(Path(gca_polar))
    if shot_map:
        result["shot_map_url"] = _to_url(Path(shot_map))
    if squad_depth:
        result["squad_depth_url"] = _to_url(Path(squad_depth))
    # If pies requested, compute expected paths and include if present
    if include_pies:
        exp = _expected_paths(team_name, season)
        if exp["sca_pie"].exists():
            result["sca_pie_url"] = _to_url(exp["sca_pie"])
        if exp["gca_pie"].exists():
            result["gca_pie_url"] = _to_url(exp["gca_pie"])
    return result

# ------------------------------ Routes ------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
def api_search(q: str = Query(..., min_length=2), limit: int = Query(8, ge=1, le=25)):
    df = load_mapping()
    qn = _normalize(q)
    scored: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        s = score_row(qn, row)
        if s <= 0:
            continue
        display = (row.get("fbref_name") or row.get("tm_name") or row.get("understat_name") or "").strip()
        slug = str(row.get("file_slug") or "").strip() or slugify(display)
        scored.append({
            "name": display,
            "file_slug": slug,
            "score": float(s),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]

@app.post("/api/generate")
def api_generate(payload: GeneratePayload):
    if not payload.team_a and not payload.team_b:
        raise HTTPException(status_code=400, detail="Provide at least one team.")
    # Force fixed defaults regardless of client input
    season = 2025
    metric = "per90"
    out: Dict[str, Any] = {"season": season}

    if payload.team_a:
        out["team_a"] = generate_for_team(
            payload.team_a,
            season,
            include_pies=payload.include_pies,
            percentile_metric=metric,
            no_fetch=payload.no_fetch,
            force_fetch=payload.force_fetch,
            country=payload.country,
            tier=payload.tier,
            gen_polar=payload.gen_polar,
            gen_shot=payload.gen_shot,
            gen_squad=payload.gen_squad,
        )
    if payload.team_b:
        out["team_b"] = generate_for_team(
            payload.team_b,
            season,
            include_pies=payload.include_pies,
            percentile_metric=metric,
            no_fetch=payload.no_fetch,
            force_fetch=payload.force_fetch,
            country=payload.country,
            tier=payload.tier,
            gen_polar=payload.gen_polar,
            gen_shot=payload.gen_shot,
            gen_squad=payload.gen_squad,
        )
    return JSONResponse(out)
