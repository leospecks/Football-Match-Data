"""Create a Quarto-ready Markdown table from downloaded match metadata.

Run: python create_data_overview.py
Then add this to your .qmd page: {{< include _data_overview.md >}}
Uses only Python's standard library; scans data recursively for *_match.json.
"""

import argparse
import html
import json
import re
from datetime import date, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = PROJECT_DIR / "_data_overview.md"

# Optional display-name adjustments to match your preferred table labels.
TEAM_NAME_OVERRIDES = {"Newcastle United Jets FC": "Newcastle United Jets"}


def _season_label(season):
    start, end = season.get("start_year"), season.get("end_year")
    if start is not None and end is not None:
        start, end = int(start), int(end)
        return f"{start % 100:02d}/{end % 100:02d}" if start != end else f"{start % 100:02d}"
    return re.sub(r"\b(?:19|20)(\d{2})\b", r"\1", str(season.get("name") or "—"))


def _cell(value):
    """Escape table separators/HTML and collapse embedded newlines."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return html.escape(text, quote=False).replace("|", r"\|")


def _match_row(match):
    edition = match.get("competition_edition") or {}
    competition = edition.get("competition") or {}
    season = edition.get("season") or {}
    round_info = match.get("competition_round") or {}
    matchday = round_info.get("round_number")
    if matchday is None:
        matchday = round_info.get("name") or "—"
        number = re.fullmatch(r"(?:Round|Matchday)\s+(\d+)", str(matchday), re.I)
        if number:
            matchday = number.group(1)

    def team_name(side):
        name = (match.get(f"{side}_team") or {}).get("name") or "—"
        return TEAM_NAME_OVERRIDES.get(name, name)

    raw_date = match.get("date_time")
    # Use the calendar date stored in the JSON, without timezone conversion.
    played_on = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date() if raw_date else None
    match_id = match.get("id")
    if match_id is None:
        raise ValueError("Missing match ID.")
    row = (
        competition.get("name") or "—",
        _season_label(season),
        matchday,
        f"{team_name('home')} vs. {team_name('away')}",
        played_on.strftime("%d.%m.%y") if played_on else "—",
        str(match_id),
    )
    return played_on or date.max, row


def create_data_overview(data_dir=DATA_DIR, output=OUTPUT_FILE):
    """Return a Markdown table and optionally write it to output.

    Recursively reads *_match.json, including data/<match_id>/... layouts.
    Columns: League, Season, Matchday, Game, Date, Match-ID.
    Rows are sorted chronologically. Duplicate match IDs with identical table
    information are included once. Conflicting duplicates and malformed files
    raise a filename-specific error rather than producing an incomplete table.
    Set output=None to return the Markdown without writing a file.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*_match.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_match.json files found in {data_dir.resolve()}. Run fetch_data.py first."
        )
    records = {}
    for path in files:
        try:
            with path.open(encoding="utf-8-sig") as source:
                match = json.load(source)
            played_on, row = _match_row(match)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"Cannot read match metadata in {path}: {error}") from error
        match_id = row[-1]
        record = (played_on, row)
        if match_id in records and records[match_id] != record:
            raise ValueError(f"Conflicting metadata for match {match_id} in {path}.")
        records[match_id] = record

    lines = [
        "| League | Season | Matchday | Game | Date | Match-ID |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for _played_on, row in sorted(records.values(), key=lambda item: (item[0], item[1][-1])):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    markdown = "\n".join(lines) + "\n"
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    return markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Match metadata folder (default: data next to this script).")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Markdown output (default: _data_overview.md next to this script).")
    args = parser.parse_args(argv)
    markdown = create_data_overview(args.data_dir, args.output)
    print(markdown)
    print(f"Wrote {len(markdown.splitlines()) - 2} matches to {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
