#!/usr/bin/env python3
"""
fbref_team_hex.py

Given a club name (and optionally a country code), fetch the FBref team hex ID
from the country clubs page (e.g., https://fbref.com/en/country/clubs/ENG/).

- Requires only 1 request if country_code is provided.
- No club-specific mappings; uses generic normalization to match names.
- Respects FBref limits: keep requests minimal (1 per lookup when country is known).

CLI usage:
  python3 fbref_team_hex.py --club "Tottenham Hotspur" --country ENG

Outputs the 8-character hex to stdout.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Dict
import urllib.request
import urllib.error


FBREF_BASE = "https://fbref.com"
COUNTRY_CLUBS_PATH = "/en/country/clubs/{code}/"
TEAM_ID_RE = re.compile(r"/en/squads/([0-9a-f]{8})/")


def _normalize_name(name: str) -> str:
    """Generic normalization: lowercase, strip non-alnum.
    Example: "FC Porto" -> "fcporto", "Tottenham Hotspur" -> "tottenhamhotspur".
    """
    return re.sub(r"[^a-z0-9]+", "", name.lower())


@dataclass
class ClubEntry:
    name: str
    href: str
    team_id: str


def _fetch_country_clubs(country_code: str, timeout: int = 20) -> Dict[str, ClubEntry]:
    """Return a mapping of normalized club name -> ClubEntry for a country.

    country_code: 3-letter code like ENG, ESP, ITA, POR, etc.
    """
    url = FBREF_BASE + COUNTRY_CLUBS_PATH.format(code=country_code.strip().upper())
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        # Determine charset from Content-Type; default to utf-8
        charset = "utf-8"
        content_type = resp.headers.get("Content-Type")
        if content_type:
            m_ct = re.search(r"charset=([A-Za-z0-9_-]+)", content_type)
            if m_ct:
                charset = m_ct.group(1)
        html = raw.decode(charset, errors="replace")

    mapping: Dict[str, ClubEntry] = {}

    # Regex to find anchor tags linking to /en/squads/<hex>/... and extract inner text
    # Captures: full href, team_id, and inner HTML for the link text
    pattern_a = re.compile(
        r'<a\s+[^>]*href="([^\"]*?/en/squads/([0-9a-f]{8})/[^\"]*)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for href, team_id, inner in pattern_a.findall(html):
        # Strip any nested tags from inner HTML to get plain text
        text = re.sub(r"<[^>]+>", "", inner)
        text = (text or "").strip()
        if not text:
            continue
        key = _normalize_name(text)
        # First occurrence wins; avoid clobbering if duplicates
        if key not in mapping:
            mapping[key] = ClubEntry(name=text, href=href, team_id=team_id)

    return mapping


def get_fbref_team_hex(club: str, country_code: str, *, timeout: int = 20) -> str:
    """Resolve FBref 8-char team hex for a club within a given country.

    This performs a single request to the country's clubs page and matches
    by normalized club name.

    Raises ValueError if not found.
    """
    if not club or not country_code:
        raise ValueError("Both 'club' and 'country_code' are required")

    mapping = _fetch_country_clubs(country_code, timeout=timeout)

    # Try normalized exact match
    key = _normalize_name(club)
    if key in mapping:
        return mapping[key].team_id

    # Try a few generic variants without introducing club-specific synonyms
    # - Collapse multiple spaces/hyphens
    # - Remove common punctuation variants (already handled by normalization)
    # If still not found, raise with available suggestions
    suggestions = []
    for k, entry in mapping.items():
        if key in k or k in key:
            suggestions.append(entry.name)

    hint = f". Did you mean one of: {', '.join(suggestions[:5])}" if suggestions else ""
    raise ValueError(f"Club '{club}' not found under country '{country_code}'{hint}")


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve FBref team hex from country clubs page.")
    p.add_argument("--club", required=True, help="Club name, e.g., 'Tottenham Hotspur'")
    p.add_argument("--country", required=True, help="Country code, e.g., ENG, ESP, ITA, GER, FRA, POR")
    args = p.parse_args()

    try:
        team_id = get_fbref_team_hex(args.club, args.country)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(team_id)


if __name__ == "__main__":
    main()
