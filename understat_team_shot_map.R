#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(worldfootballR)
  library(ggplot2)
  library(dplyr)
  library(stringr)
  library(optparse)
})

# Parse CLI options
option_list <- list(
  make_option(c("--team"), type = "character", help = "Understat team name, e.g., 'Real Sociedad'", metavar = "character"),
  make_option(c("--season"), type = "integer", default = NA, help = "Season start year (e.g., 2024). Defaults to latest available.", metavar = "integer"),
  make_option(c("--out"), type = "character", default = NA, help = "Output image path. Defaults to ./cache_images/<team>_<season>_understat_shot_map.png", metavar = "character"),
  make_option(c("--for-against"), type = "character", default = "for", help = "Plot 'for' (team shots) or 'against' (opponent shots). Default: for", metavar = "character")
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
  warning("No shots returned from Understat for this team/season. Producing empty pitch.")
}

# Determine which shots belong to the team vs opponents
shots <- shots %>%
  mutate(shot_team = if_else(home_away == "h", home_team, away_team))

is_for <- tolower(opt$`for-against`) == "for"
team_lower <- tolower(team_input)

# Title and credit strings
title_text <- sprintf("%s Shots %s (%d)", team_input, ifelse(is_for, "FOR", "AGAINST"), year)
credit_text <- "Data source: Understat"

shots_team <- shots %>%
  filter(
    if (is_for) {
      tolower(shot_team) == team_lower
    } else {
      (tolower(home_team) == team_lower | tolower(away_team) == team_lower) & tolower(shot_team) != team_lower
    }
  )

# Build default output path
slug <- tolower(gsub("[^a-z0-9]+", "_", team_input))
if (is.na(opt$out) || !nzchar(opt$out)) {
  out_dir <- file.path(".", "cache_images")
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(out_dir, sprintf("%s_%s_understat_shot_map.png", slug, year))
} else {
  out_path <- opt$out
  out_dir <- dirname(out_path)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

# Pitch dimensions (normalized Understat 0..1 coordinates). We'll draw a minimal attacking-half context.
# Using full-pitch outline for simplicity with key areas near the opponent goal (right side).
penalty_x <- 1 - 16.5/105
penalty_ymin <- 0.5 - (40.32/2)/68
penalty_ymax <- 0.5 + (40.32/2)/68
six_x <- 1 - 5.5/105
six_ymin <- 0.5 - (18.32/2)/68
six_ymax <- 0.5 + (18.32/2)/68

p <- ggplot() +
  # Pitch base
  annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1, fill = "#0b5e34", color = NA) +
  # Outer lines
  annotate("segment", x = 0, xend = 1, y = 0, yend = 0, color = "white", linewidth = 0.6) +
  annotate("segment", x = 0, xend = 1, y = 1, yend = 1, color = "white", linewidth = 0.6) +
  annotate("segment", x = 0, xend = 0, y = 0, yend = 1, color = "white", linewidth = 0.6) +
  annotate("segment", x = 1, xend = 1, y = 0, yend = 1, color = "white", linewidth = 0.6) +
  # Penalty area (opponent)
  annotate("rect", xmin = penalty_x, xmax = 1, ymin = penalty_ymin, ymax = penalty_ymax, fill = NA, color = "white", linewidth = 0.6) +
  # Six-yard box (opponent)
  annotate("rect", xmin = six_x, xmax = 1, ymin = six_ymin, ymax = six_ymax, fill = NA, color = "white", linewidth = 0.6)

if (nrow(shots_team) > 0) {
  p <- p +
    geom_point(
      data = shots_team,
      aes(x = X, y = Y),
      shape = 21, stroke = 0.3, size = 1.9, color = "black", fill = "white", alpha = 0.85
    )
}

p <- p +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  theme_void() +
  theme(
    text = element_text(family = "Arial"),
    plot.margin = margin(44, 16, 12, 16),
    plot.background = element_rect(fill = "#0b5e34", color = NA),
    panel.background = element_rect(fill = "#0b5e34", color = NA)
  ) +
  # Title and credit centered above the pitch, with black highlight behind text
  annotate(
    "label", x = 0.5, y = 1.08, label = title_text,
    hjust = 0.5, vjust = 1, size = 4.6, label.size = 0,
    fill = "#000000", color = "#ffffff", family = "Arial"
  ) +
  annotate(
    "label", x = 0.5, y = 1.04, label = credit_text,
    hjust = 0.5, vjust = 1, size = 3.7, label.size = 0,
    fill = "#000000", color = "#d0d5dd", family = "Arial"
  )

# Save
invisible(ggsave(filename = out_path, plot = p, width = 6.5, height = 4.5, dpi = 300))

cat(sprintf("Saved shot map: %s\n", out_path))
