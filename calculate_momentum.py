"""Calculate EPV-based momentum for every downloaded SkillCorner match.

Run: python calculate_momentum.py
Reads data/**/<id>_dynamic_events.csv and the matching <id>_match.json.
Writes momentum.csv and _momentum.md next to this script. No extra packages.

Definition (our own metric, not Opta's proprietary model):
* Use player_possession events only.
* Assign possession_epv_for_start/end to their respective timestamps.
* In each time bin, take the highest EPV snapshot for home and away.
* Smooth home-minus-away using a trailing window, with weights 1..window
  from oldest to newest. At the start of a half use available history.
* Multiply by 100: positive momentum points favor home; negative favor away.

The default is 60-second bins and a three-bin window. Periods are kept
separate, so first-half stoppage time never overlaps second-half minutes.
Blank cells mean unavailable data, not zero momentum. A bin without any EPV
snapshots is unavailable. Missing EPV snapshots also invalidate their bin;
any smoothing window containing an unavailable bin stays blank.
An observed bin with no snapshots for one team assigns that team a peak of 0.
The CSV stores interval START/END labels; it is not a continuous EPV timeline.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_CSV = PROJECT_DIR / "momentum.csv"
OUTPUT_MARKDOWN = PROJECT_DIR / "_momentum.md"
BIN_SECONDS = 60
WINDOW_BINS = 3
SCALE = 100.0

PERIOD_STARTS = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
PERIOD_LABELS = {1: "1H", 2: "2H", 3: "ET1", 4: "ET2"}
REQUIRED_COLUMNS = {
    "event_type", "period", "team_id", "time_start", "time_end",
    "possession_epv_for_start", "possession_epv_for_end",
}
MISSING_VALUES = {"", "none", "null", "nan", "na", "n/a"}


def _seconds(value):
    total = 0.0
    for component in str(value).strip().split(":"):
        total = total * 60 + float(component)
    if not math.isfinite(total) or total < 0:
        raise ValueError(f"Invalid event timestamp: {value!r}")
    return total


def _epv(value):
    text = "" if value is None else str(value).strip()
    if text.lower() in MISSING_VALUES:
        return None
    number = float(text)
    if not math.isfinite(number):
        return None
    if not 0 <= number <= 1:
        raise ValueError(f"Expected EPV probabilities between 0 and 1, got {value!r}.")
    return number


def _bin(time, period, bin_seconds, ending=False):
    local_time = time - PERIOD_STARTS[period]
    if local_time < -0.001:
        raise ValueError(
            f"Timestamp precedes period {period}. Expected match-clock timestamps "
            "(second half starts at 00:45:00), as specified by SkillCorner."
        )
    local_time = max(0.0, local_time)
    # A possession ending exactly on a boundary belongs to its ending interval.
    if ending and local_time > 0:
        local_time = math.nextafter(local_time, -math.inf)
    return int(local_time // bin_seconds)


def _read_match(events_file, match_file, bin_seconds, window_bins):
    with match_file.open(encoding="utf-8-sig") as source:
        match = json.load(source)
    match_id = str(match["id"])
    expected_id = events_file.name.removesuffix("_dynamic_events.csv")
    if match_id != expected_id:
        raise ValueError("Match ID in metadata does not match the event filename.")
    home_id, away_id = str(match["home_team"]["id"]), str(match["away_team"]["id"])
    teams = {home_id: 0, away_id: 1}
    peaks = defaultdict(lambda: [0.0, 0.0])
    observed, missing = set(), set()
    last_bin = {}

    # Metadata lets us retain unavailable periods/minutes as blanks rather than
    # silently shortening the match when event coverage ends early.
    for half in match.get("match_periods", []):
        period = int(half["period"])
        duration = half.get("duration_minutes")
        if period in PERIOD_STARTS and duration is not None:
            duration = float(duration) * 60
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid period duration in match metadata.")
            if duration > 0:
                last_bin[period] = max(0, math.ceil(duration / bin_seconds) - 1)

    qualifying_rows = valid_snapshots = 0
    with events_file.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        reader.fieldnames = [field.strip() for field in (reader.fieldnames or [])]
        absent = REQUIRED_COLUMNS - set(reader.fieldnames)
        if absent:
            raise ValueError(
                "Missing required columns: " + ", ".join(sorted(absent))
                + ". This event file cannot support the specified EPV method."
            )
        for row_number, row in enumerate(reader, 2):
            if (row.get("event_type") or "").strip().lower() != "player_possession":
                continue
            try:
                period = int(row["period"])
                if period not in PERIOD_STARTS:
                    continue  # Penalty shootouts have no attacking momentum.
                team_id = row["team_id"].strip()
                if team_id not in teams:
                    raise ValueError(f"Unknown team_id {team_id!r}.")
                start, end = _seconds(row["time_start"]), _seconds(row["time_end"])
                if end < start:
                    raise ValueError("Event ends before it starts.")
                start_bin = _bin(start, period, bin_seconds)
                end_bin = max(start_bin, _bin(end, period, bin_seconds, ending=end > start))
                last_bin[period] = max(last_bin.get(period, -1), end_bin)
                qualifying_rows += 1
                for index, column in ((start_bin, "possession_epv_for_start"),
                                      (end_bin, "possession_epv_for_end")):
                    key = (period, index)
                    value = _epv(row[column])
                    if value is None:
                        missing.add(key)
                    else:
                        observed.add(key)
                        valid_snapshots += 1
                        peaks[key][teams[team_id]] = max(peaks[key][teams[team_id]], value)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise ValueError(f"CSV row {row_number}: {error}") from error
    if qualifying_rows == 0 or valid_snapshots == 0:
        raise ValueError("No populated EPV values on player_possession rows.")

    raw, momentum = {}, {}
    for period, final_index in sorted(last_bin.items()):
        for index in range(final_index + 1):
            key = (period, index)
            raw[key] = (peaks[key][0] - peaks[key][1]
                        if key in observed and key not in missing else None)
        for index in range(final_index + 1):
            history = [raw[period, previous]
                       for previous in range(max(0, index - window_bins + 1), index + 1)]
            if any(value is None for value in history):
                momentum[period, index] = None
            else:
                weights = range(window_bins - len(history) + 1, window_bins + 1)
                momentum[period, index] = SCALE * sum(
                    weight * value for weight, value in zip(weights, history)
                ) / sum(weights)
    return match_id, momentum


def _clock(seconds):
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def calculate_momentum(data_dir=DATA_DIR, output_csv=OUTPUT_CSV,
                       output_markdown=OUTPUT_MARKDOWN,
                       bin_seconds=BIN_SECONDS, window_bins=WINDOW_BINS,
                       skip_invalid=False):
    """Write wide tables and return rows as dictionaries.

    Each match ID is one column; Period and Time identify the interval.
    By default every event file must be usable; errors leave existing output
    files untouched. Set skip_invalid=True to omit unsupported matches with
    warnings. Files missing their companion metadata are also unsupported.
    Pass None for either output path to disable that output.
    """
    if type(bin_seconds) is not int or bin_seconds <= 0:
        raise ValueError("bin_seconds must be a positive integer.")
    if type(window_bins) is not int or window_bins <= 0:
        raise ValueError("window_bins must be a positive integer.")
    files = sorted(Path(data_dir).rglob("*_dynamic_events.csv"))
    if not files:
        raise FileNotFoundError(f"No *_dynamic_events.csv files found under {Path(data_dir).resolve()}.")
    games, errors = {}, []
    for events_file in files:
        expected_id = events_file.name.removesuffix("_dynamic_events.csv")
        match_file = events_file.with_name(f"{expected_id}_match.json")
        try:
            match_id, values = _read_match(events_file, match_file, bin_seconds, window_bins)
            if match_id in games:
                raise ValueError(f"Duplicate event files for match {match_id}; keep one copy.")
            games[match_id] = values
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            errors.append(f"{events_file}: {error}")
    if errors and not skip_invalid:
        raise ValueError("Could not calculate all matches; no output written:\n" + "\n".join(errors)
                         + "\nUse --skip-invalid to explicitly omit unsupported files.")
    for error in errors:
        print(f"WARNING: Skipped {error}", file=sys.stderr)
    if not games:
        raise ValueError("No usable matches with populated EPV data were found.")

    match_ids = sorted(games, key=lambda value: (len(value), value))
    periods = sorted({key for values in games.values() for key in values})
    columns = ["Period", "Time", *match_ids]
    rows = []
    for period, index in periods:
        start = PERIOD_STARTS[period] + index * bin_seconds
        row = {"Period": PERIOD_LABELS[period],
               "Time": f"{_clock(start)}–{_clock(start + bin_seconds)}"}
        for match_id in match_ids:
            value = games[match_id].get((period, index))
            row[match_id] = "" if value is None else f"{value:.4f}"
        rows.append(row)

    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    if output_markdown is not None:
        path = Path(output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["| " + " | ".join(columns) + " |",
                 "| " + " | ".join(["---", "---", *["---:" for _ in match_ids]]) + " |"]
        lines.extend("| " + " | ".join(row[column] for column in columns) + " |" for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--csv-output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--markdown-output", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument("--bin-seconds", type=int, default=BIN_SECONDS)
    parser.add_argument("--window-bins", type=int, default=WINDOW_BINS)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args(argv)
    rows = calculate_momentum(args.data_dir, args.csv_output, args.markdown_output,
                              args.bin_seconds, args.window_bins, args.skip_invalid)
    print(f"Calculated {len(rows)} time intervals for {len(rows[0]) - 2} matches.")
    print(f"CSV: {args.csv_output.resolve()}\nMarkdown: {args.markdown_output.resolve()}")
    print("Positive = home; negative = away. Units: momentum points (EPV difference × 100).")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
