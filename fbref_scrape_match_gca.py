#!/usr/bin/env python3
"""
fbref_scrape_match_gca.py

Fetch the FBref Goal & Shot Creation match logs table for a given team hex and season,
parse the table with id "matchlogs_for" (including when wrapped in HTML comments), and
emit rows matching our processing needs.

- Standard library only (urllib + html.parser + csv + regex)
- 1–2 requests per run (depending on slug attempt)
- No club-specific mappings

CLI usage examples:
  # Using a known hex (Tottenham Hotspur) with an FC slug attempt
  python3 fbref_scrape_match_gca.py --hex 361ca564 --season 2024-2025 \
      --slug Tottenham-Hotspur-FC-Match-Logs-All-Competitions \
      --out tottenham_hotspur_2025_match_gca_fbref.csv

  # Resolve hex from club + country, then fetch
  python3 fbref_scrape_match_gca.py --club "Tottenham Hotspur FC" --country ENG --season 2024-2025 \
      --out tottenham_hotspur_2025_match_gca_fbref.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

import fbref_team_hex as teamhex

FBREF_BASE = "https://fbref.com"
MATCHLOG_GCA_PATH = "/en/squads/{hex}/{season}/matchlogs/all_comps/gca/"
TABLE_ID = "matchlogs_for"


def fetch_html(url: str, *, timeout: int = 20) -> Tuple[str, int]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = "utf-8"
            content_type = resp.headers.get("Content-Type")
            if content_type:
                m = re.search(r"charset=([A-Za-z0-9_-]+)", content_type)
                if m:
                    charset = m.group(1)
            html = raw.decode(charset, errors="replace")
            code = resp.getcode() or 200
            return html, code
    except urllib.error.HTTPError as e:
        if e.fp:
            raw = e.fp.read()
            html = raw.decode("utf-8", errors="replace")
        else:
            html = ""
        return html, e.code


def extract_table_html(html: str, table_id: str = TABLE_ID) -> str:
    """Return only the <table id="matchlogs_for">...</table> HTML, not the whole page.

    Handles both direct table presence and comment-wrapped blocks inside
    a div with id="all_<table_id>".
    """

    def slice_table(src: str) -> Optional[str]:
        pos_id = src.find(f'id="{table_id}"')
        if pos_id == -1:
            return None
        start = src.rfind("<table", 0, pos_id)
        if start == -1:
            return None
        end = src.find("</table>", pos_id)
        if end == -1:
            return None
        return src[start : end + len("</table>")]

    # Try direct table presence first
    table = slice_table(html)
    if table:
        return table

    # FBref often wraps tables in comments inside a div with id="all_<table_id>"
    wrapper_id = f'id="all_{table_id}"'
    pos = html.find(wrapper_id)
    if pos != -1:
        start_c = html.find("<!--", pos)
        end_c = html.find("-->", start_c + 4) if start_c != -1 else -1
        if start_c != -1 and end_c != -1:
            block = html[start_c + 4 : end_c]
            table_in_block = slice_table(block) or (block if f'id="{table_id}"' in block else None)
            if table_in_block:
                return table_in_block

    raise ValueError(f"Could not locate table '{table_id}' in page")


class MatchLogsParser(HTMLParser):
    def __init__(self, target_table_id: str = TABLE_ID) -> None:
        super().__init__()
        self.target_table_id = target_table_id
        self.in_table = False
        self.in_tbody = False
        self.in_tr = False
        self.current_datastat: Optional[str] = None
        self.current_text_parts: List[str] = []
        self.current_row: Dict[str, str] = {}
        self.rows: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attrs_d = dict(attrs)
        if tag == "table" and attrs_d.get("id") == self.target_table_id:
            self.in_table = True
            return
        if not self.in_table:
            return
        if tag == "tbody":
            self.in_tbody = True
            return
        if self.in_tbody and tag == "tr":
            self.in_tr = True
            self.current_row = {}
            return
        if self.in_tr and tag in ("td", "th"):
            self.current_datastat = attrs_d.get("data-stat")
            self.current_text_parts = []

    def handle_endtag(self, tag: str):
        if self.in_tr and tag in ("td", "th") and self.current_datastat is not None:
            text = "".join(self.current_text_parts).strip()
            # Collapse internal whitespace
            text = re.sub(r"\s+", " ", text)
            self.current_row[self.current_datastat] = text
            self.current_datastat = None
            self.current_text_parts = []
            return
        if self.in_tr and tag == "tr":
            # Filter out header/separator rows that lack a date or opponent
            if self.current_row and (self.current_row.get("date") or self.current_row.get("opponent")):
                self.rows.append(self.current_row)
            self.in_tr = False
            self.current_row = {}
            return
        if self.in_tbody and tag == "tbody":
            self.in_tbody = False
            return
        if self.in_table and tag == "table":
            self.in_table = False
            return

    def handle_data(self, data: str):
        if self.current_datastat is not None:
            self.current_text_parts.append(data)


def parse_matchlogs_table(html_or_table: str) -> List[Dict[str, str]]:
    parser = MatchLogsParser(TABLE_ID)
    parser.feed(html_or_table)
    return parser.rows


def build_matchlogs_url(team_hex: str, season: str, slug: Optional[str]) -> str:
    base = FBREF_BASE + MATCHLOG_GCA_PATH.format(hex=team_hex, season=season)
    if slug:
        if not base.endswith("/"):
            base += "/"
        return base + slug
    return base


def map_row_to_output(row: Dict[str, str], *, team_url: str, team_name: str) -> Dict[str, str]:
    # Map FBref data-stat keys to our CSV headers
    def na_if_empty(v: str) -> str:
        return v if v not in (None, "") else "NA"

    out = {
        "Team_Url": team_url,
        "Team": team_name,
        "ForAgainst": "For",
        "Date": na_if_empty(row.get("date", "")),
        "Time": na_if_empty(row.get("start_time", "")),
        "Comp": na_if_empty(row.get("comp", "")),
        "Round": na_if_empty(row.get("round", "")),
        "Day": na_if_empty(row.get("dayofweek", "")),
        "Venue": na_if_empty(row.get("venue", "")),
        "Result": na_if_empty(row.get("result", "")),
        "GF": na_if_empty(row.get("goals_for", "")),
        "GA": na_if_empty(row.get("goals_against", "")),
        "Opponent": na_if_empty(row.get("opponent", "")),
        # SCA
        "SCA_SCA_Types": na_if_empty(row.get("sca", "")),
        "PassLive_SCA_Types": na_if_empty(row.get("sca_passes_live", "")),
        "PassDead_SCA_Types": na_if_empty(row.get("sca_passes_dead", "")),
        "TO_SCA_Types": na_if_empty(row.get("sca_take_ons", "")),
        "Sh_SCA_Types": na_if_empty(row.get("sca_shots", "")),
        "Fld_SCA_Types": na_if_empty(row.get("sca_fouled", "")),
        "Def_SCA_Types": na_if_empty(row.get("sca_defense", "")),
        # GCA
        "GCA_GCA_Types": na_if_empty(row.get("gca", "")),
        "PassLive_GCA_Types": na_if_empty(row.get("gca_passes_live", "")),
        "PassDead_GCA_Types": na_if_empty(row.get("gca_passes_dead", "")),
        "TO_GCA_Types": na_if_empty(row.get("gca_take_ons", "")),
        "Sh_GCA_Types": na_if_empty(row.get("gca_shots", "")),
        "Fld_GCA_Types": na_if_empty(row.get("gca_fouled", "")),
        "Def_GCA_Types": na_if_empty(row.get("gca_defense", "")),
    }
    return out


def extract_team_name_from_title(html: str) -> Optional[str]:
    # Example <title>: "2024-2025 Tottenham Hotspur Match Logs (Goal and Shot Creation), All Competitions | FBref.com"
    m = re.search(r"\b20\d{2}-20\d{2}\s+(.*?)\s+Match Logs", html, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape FBref GCA match logs table for a team")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hex", help="8-char FBref team hex, e.g., 361ca564")
    g.add_argument("--club", help="Club name to resolve to hex, e.g., 'Tottenham Hotspur FC'")
    p.add_argument("--country", help="3-letter country code if resolving by club, e.g., ENG")
    p.add_argument("--season", default="2024-2025", help="Season label, e.g., 2024-2025")
    p.add_argument("--slug", help="Optional trailing slug to attempt, e.g., 'Tottenham-Hotspur-FC-Match-Logs-All-Competitions'")
    p.add_argument("--no-fallback", action="store_true", help="If set and slug URL fails, do not retry without slug; exit with error.")
    p.add_argument("--out", help="Optional output CSV path; if omitted, prints CSV to stdout")
    p.add_argument("--dump-table-html", action="store_true", help="Print raw HTML of the matchlogs_for table and exit")

    args = p.parse_args()

    team_hex = args.hex
    if not team_hex:
        if not args.country:
            print("--country is required when using --club", file=sys.stderr)
            sys.exit(2)
        team_hex = teamhex.get_fbref_team_hex(args.club, args.country)

    # Build URL (attempt slug if provided)
    url = build_matchlogs_url(team_hex, args.season, args.slug)

    html, code = fetch_html(url)
    if code >= 400 and args.slug:
        if args.no_fallback:
            print(f"Slug URL failed with status {code}. Stopping due to --no-fallback.", file=sys.stderr)
            sys.exit(1)
        # retry without slug
        fallback = build_matchlogs_url(team_hex, args.season, None)
        html, code = fetch_html(fallback)
        if code >= 400:
            print(f"Failed to load both slug URL and fallback URL (status {code})", file=sys.stderr)
            sys.exit(1)
    elif code >= 400:
        print(f"Failed to load URL (status {code})", file=sys.stderr)
        sys.exit(1)

    # Try to get a team name for output metadata
    team_name = args.club or extract_team_name_from_title(html) or ""
    team_url = f"{FBREF_BASE}/en/squads/{team_hex}/{args.season}/"

    # Extract and parse table
    table_html = extract_table_html(html, TABLE_ID)
    if args.dump_table_html:
        print(table_html)
        return
    rows = parse_matchlogs_table(table_html)

    # Map to output schema
    mapped = [map_row_to_output(r, team_url=team_url, team_name=team_name) for r in rows]

    # Output
    fieldnames = [
        "Team_Url","Team","ForAgainst","Date","Time","Comp","Round","Day","Venue","Result","GF","GA","Opponent",
        "SCA_SCA_Types","PassLive_SCA_Types","PassDead_SCA_Types","TO_SCA_Types","Sh_SCA_Types","Fld_SCA_Types","Def_SCA_Types",
        "GCA_GCA_Types","PassLive_GCA_Types","PassDead_GCA_Types","TO_GCA_Types","Sh_GCA_Types","Fld_GCA_Types","Def_GCA_Types",
    ]

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in mapped:
                writer.writerow(r)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for r in mapped:
            writer.writerow(r)


if __name__ == "__main__":
    main()
