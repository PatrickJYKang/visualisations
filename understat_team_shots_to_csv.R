#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(worldfootballR)
  library(dplyr)
  library(stringr)
  library(readr)
  library(optparse)
})

# CLI options
option_list <- list(
  make_option(c("--team"), type = "character", help = "Understat team name, e.g., 'Real Sociedad'", metavar = "character"),
  make_option(c("--season"), type = "integer", default = NA, help = "Season start year (e.g., 2024). Defaults to latest available.", metavar = "integer"),
  make_option(c("--out"), type = "character", default = NA, help = "Output CSV path. Defaults to ./<team_slug>_<season>_understat_shots.csv", metavar = "character")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$team) || is.na(opt$team) || nchar(opt$team) == 0) {
  stop("Please provide --team <Understat team name>")
}

team_input <- opt$team

# Resolve team meta (season URLs)
meta <- tryCatch(understat_team_meta(team_names = team_input), error = function(e) data.frame())
if (nrow(meta) == 0) {
  stop(paste0("Team not found on Understat or league unsupported: ", team_input,
              "\nUnderstat currently supports: EPL, La liga, Bundesliga, Serie A, Ligue 1, RFPL"))
}

# Choose season (start year). If not supplied, pick latest available
if (is.na(opt$season)) {
  year <- max(meta$year, na.rm = TRUE)
} else {
  if (!(opt$season %in% meta$year)) {
    stop(paste0("Season start year ", opt$season, " not available for ", team_input, ". Available: ", paste(sort(unique(meta$year)), collapse = ", "))) }
  year <- opt$season
}

url <- meta$url[meta$year == year][1]
if (is.na(url) || !nzchar(url)) {
  stop("Could not resolve Understat team season URL")
}

# Fetch team season shots (includes both team and opponent shots for all matches)
shots <- tryCatch(understat_team_season_shots(team_url = url), error = function(e) data.frame())
if (nrow(shots) == 0) {
  stop("No shots returned from Understat for this team/season.")
}

# Add team identity and for/against flag relative to the input team
team_lower <- tolower(team_input)
shots <- shots %>%
  mutate(
    shot_team = if_else(home_away == "h", home_team, away_team),
    for_against = if_else(tolower(shot_team) == team_lower, "for", "against"),
    team = team_input,
    season = year
  ) %>%
  select(
    id, minute, result, X, Y, xG, player, home_away, shot_team,
    home_team, away_team, home_goals, away_goals, date, match_id,
    player_assisted, lastAction, for_against, team, season
  )

# Build default output path in repo root
# Lowercase first, then replace non-alphanumerics to ensure uppercase letters are preserved correctly
slug <- gsub("[^a-z0-9]+", "_", tolower(team_input))
if (is.na(opt$out) || !nzchar(opt$out)) {
  out_path <- file.path(".", sprintf("%s_%s_understat_shots.csv", slug, year))
} else {
  out_path <- opt$out
  out_dir <- dirname(out_path)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

# Write CSV
readr::write_csv(shots, out_path)
cat(sprintf("Saved shots CSV: %s\n", out_path))
