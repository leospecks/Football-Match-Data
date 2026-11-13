"""Composite attacking momentum from SkillCorner match folders.

Run: python calculate_momentum.py
Uses only the standard library. Reads each tracking JSONL once, incrementally.
Outputs momentum.csv, _momentum.md, momentum_report.json, and a reusable
momentum_normalization.json. An optional --components-output CSV exposes
team scores and individual feature contributions for inspection.

Defaults: 10-second bins, 180-second trailing window, 60-second half-life.
Team score = 100 * weighted average of normalized rolling features.
Momentum = home score - away score; these are index points, not probabilities.

Features and weights:
  territory .30: possession-weighted ball depth beyond halfway
  box       .20: time possessing the ball inside the opposition penalty area
  entries   .15: controlled final-third entries per minute
  shots     .20: player possessions ending with a shot, per minute
  recoveries.05: recoveries/interceptions in the opposition half, per minute
  epv       .10: available possession_epv_for_start/end snapshots

Rolling features are divided by pooled 95th-percentile reference levels and
clipped to [0, 1]. References are saved on the first run and reused; use
--recalibrate to replace them deliberately. A zero percentile uses the 95th
percentile of positive observations; features with no positive observations
are disabled. Both teams always use the same active features and weights.

Coverage assumptions:
* Phase intervals identify in-play time; gaps in that file mean stoppages.
  Without that file, frames without known possession remain missing.
* An event CSV is assumed complete between its first and last possession
  events within each half. Unclassified event endpoints are flagged missing.
* EPV missingness is expected outside player_possession rows. Sparse valid
  snapshots are averaged, never duplicated across tracking frames.
* A tracking interval is held for at most MAX_TRACKING_GAP seconds; larger
  gaps are not interpolated. Each rolling window requires sufficient coverage.
* Unknown factors are removed symmetrically. Too little usable base weight
  produces a blank score rather than a potentially misleading reduced model.
"""

import argparse
import bisect
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_CSV = PROJECT_DIR / "momentum.csv"
OUTPUT_MARKDOWN = PROJECT_DIR / "_momentum.md"
OUTPUT_REPORT = PROJECT_DIR / "momentum_report.json"
NORMALIZATION_FILE = PROJECT_DIR / "momentum_normalization.json"

BIN_SECONDS = 10
WINDOW_SECONDS = 180
HALF_LIFE_SECONDS = 60
WEIGHTS = {"territory": .30, "box": .20, "entries": .15,
           "shots": .20, "recoveries": .05, "epv": .10}
MIN_COVERAGE = .80
MIN_ACTIVE_WEIGHT = .50
MAX_TRACKING_GAP = .5
TRACKING_FRAME_SECONDS = .1
ENTRY_HYSTERESIS_METRES = 1.0
CONTROL_DISTANCE_METRES = 3.0
MAX_CONTROL_HEIGHT_METRES = 2.0
DETECTED_ONLY = False

FEATURES = tuple(WEIGHTS)
STARTS = {1: 0, 2: 2700, 3: 5400, 4: 6300}
LABELS = {1: "1H", 2: "2H", 3: "ET1", 4: "ET2"}
MISSING = {"", "none", "null", "nan", "na", "n/a"}
MODEL_VERSION = "composite-momentum-v2"


def _pair():
    return [0.0, 0.0]


@dataclass
class Bin:
    coverage: float = 0.0
    dead: float = 0.0
    territory: list = field(default_factory=_pair)
    box: list = field(default_factory=_pair)
    possession: list = field(default_factory=_pair)
    entries: list = field(default_factory=_pair)
    shots: list = field(default_factory=_pair)
    recoveries: list = field(default_factory=_pair)
    epv_sum: list = field(default_factory=_pair)
    epv_count: list = field(default_factory=_pair)
    bad: set = field(default_factory=set)


@dataclass
class Game:
    match_id: str
    match: dict
    bins: dict = field(default_factory=lambda: defaultdict(Bin))
    ends: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=lambda: {k: False for k in FEATURES})
    event_spans: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    rolled: dict = field(default_factory=dict)
    durations: dict = field(default_factory=dict)


def _number(value):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _seconds(value):
    if value is None or str(value).strip().lower() in MISSING:
        raise ValueError("Missing timestamp.")
    total = 0.0
    for part in str(value).strip().split(":"):
        total = total * 60 + float(part)
    if not math.isfinite(total) or total < 0:
        raise ValueError(f"Invalid timestamp {value!r}.")
    return total


def _local(value, period):
    time = _seconds(value) - STARTS[period]
    if time < -.001:
        raise ValueError("Expected match-clock timestamps: second half starts at 00:45:00.")
    return max(0.0, time)


def _index(time, size, ending=False):
    if ending and time > 0:
        time = math.nextafter(time, -math.inf)
    return int(time // size)


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        yield reader.fieldnames
        yield from reader


def _load_phases(path, team_ids):
    if not path.exists():
        return None
    reader = _read_csv(path)
    headers = next(reader)
    required = {"frame_start", "frame_end", "team_in_possession_id"}
    if not required.issubset(headers):
        raise ValueError(f"{path.name}: missing phase columns {sorted(required - set(headers))}.")
    intervals = []
    for row in reader:
        start, end = float(row["frame_start"]), float(row["frame_end"])
        if not all(math.isfinite(v) for v in (start, end)) or end <= start:
            raise ValueError(f"{path.name}: invalid phase interval.")
        owner = str(row["team_in_possession_id"]).strip().removesuffix(".0")
        if owner not in team_ids:
            raise ValueError(f"{path.name}: unknown possession team {owner}.")
        intervals.append((start, end, team_ids[owner]))
    intervals.sort()
    if not intervals:
        raise ValueError(f"{path.name}: empty phase file cannot establish in-play coverage.")
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError(f"{path.name}: overlapping phase intervals.")
    return [row[0] for row in intervals], intervals


def _load_events(game, path, size, team_ids):
    if not path.exists():
        game.notes.append("Event CSV missing: shots, recoveries and EPV unavailable.")
        return
    reader = _read_csv(path)
    headers = set(next(reader))
    required = {"event_type", "period", "team_id", "time_start", "time_end"}
    if not required.issubset(headers):
        raise ValueError(f"{path.name}: missing event columns {sorted(required - headers)}.")
    locations = [name for name in ("x_start", "player_in_possession_x_start") if name in headers]
    game.capabilities["shots"] = "end_type" in headers
    game.capabilities["recoveries"] = "start_type" in headers and bool(locations)
    game.capabilities["epv"] = {"possession_epv_for_start", "possession_epv_for_end"}.issubset(headers)
    unavailable = [name for name in ("shots", "recoveries", "epv") if not game.capabilities[name]]
    if unavailable:
        game.notes.append("Missing source columns for: " + ", ".join(unavailable) + ".")
    seen = set()
    for row_number, row in enumerate(reader, 2):
        if (row.get("event_type") or "").strip().lower() != "player_possession":
            continue
        try:
            period = int(row["period"])
            if period not in STARTS:
                continue
            event_id = row.get("event_id")
            if event_id:
                if event_id in seen:
                    raise ValueError(f"Duplicate player-possession event ID {event_id}.")
                seen.add(event_id)
            team_id = row["team_id"].strip().removesuffix(".0")
            if team_id not in team_ids:
                raise ValueError(f"Unknown team {team_id}.")
            team = team_ids[team_id]
            start, end = _local(row["time_start"], period), _local(row["time_end"], period)
            if end < start:
                raise ValueError("Event ends before it starts.")
            first = _index(start, size)
            last = max(first, _index(end, size, ending=end > start))
            game.ends[period] = max(game.ends.get(period, 0), end)
            old = game.event_spans.get(period, (start, end))
            game.event_spans[period] = (min(old[0], start), max(old[1], end))
            game.stats["possession_events"] += 1
            if game.capabilities["shots"]:
                outcome = (row.get("end_type") or "").strip().lower()
                if outcome in MISSING or outcome == "unknown":
                    game.bins[period, last].bad.add("shots")
                elif outcome == "shot":
                    game.bins[period, last].shots[team] += 1
            if game.capabilities["recoveries"]:
                origin = (row.get("start_type") or "").strip().lower()
                if origin in MISSING or origin == "unknown":
                    game.bins[period, first].bad.add("recoveries")
                elif origin in ("recovery", "pass_interception"):
                    x = next((value for name in locations
                              for value in [_number(row.get(name))] if value is not None), None)
                    if x is None:
                        game.bins[period, first].bad.add("recoveries")
                    elif x > 0:
                        # Dynamic-event possession coordinates already attack
                        # left-to-right. Do NOT apply the tracking half flip here.
                        game.bins[period, first].recoveries[team] += 1
            if game.capabilities["epv"]:
                for index, name in ((first, "possession_epv_for_start"),
                                    (last, "possession_epv_for_end")):
                    value = _number(row.get(name))
                    if value is None:
                        game.stats["missing_epv_snapshots"] += 1
                        continue
                    if not 0 <= value <= 1:
                        raise ValueError("EPV must be a probability between 0 and 1.")
                    cell = game.bins[period, index]
                    cell.epv_sum[team] += value
                    cell.epv_count[team] += 1
                    game.stats["valid_epv_snapshots"] += 1
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError(f"{path.name}, row {row_number}: {error}") from error
    game.notes.append("Event counts assume complete possession events between each half's first and last event.")


def _controlled(frame, owner, roster):
    ball = frame.get("ball_data") or {}
    height = _number(ball.get("z"))
    if height is not None and height > MAX_CONTROL_HEIGHT_METRES:
        return False
    coordinates = []
    for player in frame.get("player_data", []):
        if roster.get(str(player["player_id"])) != owner:
            continue
        x, y = _number(player.get("x")), _number(player.get("y"))
        if x is not None and y is not None:
            coordinates.append((x, y))
    if not coordinates:
        return None
    return any(math.hypot(x - ball["x"], y - ball["y"]) <= CONTROL_DISTANCE_METRES
               for x, y in coordinates)


def _tracking_state(game, frame, phases, roster, length, width):
    period = frame.get("period")
    if period not in STARTS or frame.get("timestamp") is None:
        return None
    time = _local(frame["timestamp"], period)
    possession = frame.get("possession") or {}
    if phases is not None:
        starts, intervals = phases
        index = bisect.bisect_right(starts, float(frame["frame"])) - 1
        active = intervals[index] if index >= 0 and frame["frame"] < intervals[index][1] else None
        if active is None:
            return (period, time, None, 0.0, False, True, False)
        owner = active[2]
    else:
        group = str(possession.get("group") or "").strip().lower()
        owner = (0 if group in ("home", "home team", "home_team") else
                 1 if group in ("away", "away team", "away_team") else
                 roster.get(str(possession.get("player_id"))))
    ball = frame.get("ball_data") or {}
    x, y = _number(ball.get("x")), _number(ball.get("y"))
    valid = (owner in (0, 1) and x is not None and y is not None
             and abs(x) <= length / 2 and abs(y) <= width / 2
             and (not DETECTED_ONLY or ball.get("is_detected") is True))
    if not valid:
        game.stats["unusable_ball_or_possession_frames"] += 1
        return (period, time, owner, 0.0, True, False, False)
    sides = game.match.get("home_team_side") or []
    direction = sides[period - 1] if len(sides) >= period else None
    if direction not in ("left_to_right", "right_to_left"):
        raise ValueError(f"Missing attacking direction for period {period}.")
    sign = 1 if direction == "left_to_right" else -1
    if owner == 1:
        sign *= -1
    forward = sign * x
    depth = (forward + length / 2) / length
    in_box = forward >= length / 2 - 16.5 and abs(y) <= 20.16
    if ball.get("is_detected") is False:
        game.stats["estimated_ball_frames_used"] += 1
    return (period, time, owner, depth, True, True, in_box)


def _integrate(game, state, end, size):
    period, start, owner, depth, live, valid, in_box = state
    if end <= start:
        return
    game.ends[period] = max(game.ends.get(period, 0), end)
    for index in range(_index(start, size), _index(end, size, ending=True) + 1):
        seconds = min(end, (index + 1) * size) - max(start, index * size)
        cell = game.bins[period, index]
        if valid:
            cell.coverage += seconds
            if live:
                cell.possession[owner] += seconds
                cell.territory[owner] += seconds * max(0.0, 2 * depth - 1)
                cell.box[owner] += seconds * in_box
            else:
                cell.dead += seconds


def _load_tracking(game, path, size, phases, team_ids):
    length, width = float(game.match["pitch_length"]), float(game.match["pitch_width"])
    if not all(math.isfinite(v) and v > 0 for v in (length, width)):
        raise ValueError("Invalid pitch dimensions.")
    roster = {str(player["id"]): team_ids[str(player["team_id"])]
              for player in game.match["players"] if str(player["team_id"]) in team_ids}
    game.capabilities.update({"territory": True, "box": True, "entries": True})
    previous = None
    entry_owner, armed = None, False
    lower = 2 / 3 - ENTRY_HYSTERESIS_METRES / length
    upper = 2 / 3 + ENTRY_HYSTERESIS_METRES / length
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
                state = _tracking_state(game, frame, phases, roster, length, width)
                if state is None:
                    continue
                game.stats["tracking_frames"] += 1
                period, time, owner, depth, live, valid, in_box = state
                if previous is not None and previous[0] == period:
                    gap = time - previous[1]
                    if gap < -1e-6:
                        raise ValueError("Tracking timestamps are not chronological.")
                    if gap <= 1e-9:
                        game.stats["duplicate_timestamps_ignored"] += 1
                        continue
                    _integrate(game, previous, previous[1] + min(gap, MAX_TRACKING_GAP), size)
                    if gap > MAX_TRACKING_GAP:
                        entry_owner, armed = None, False
                        game.stats["tracking_gaps"] += 1
                        for index in range(_index(previous[1], size), _index(time, size) + 1):
                            game.bins[period, index].bad.add("entries")
                elif previous is not None:
                    if period < previous[0]:
                        raise ValueError("Tracking periods are not chronological.")
                    _integrate(game, previous, previous[1] + TRACKING_FRAME_SECONDS, size)
                    entry_owner, armed = None, False
                if valid and live:
                    if entry_owner != owner:
                        entry_owner, armed = owner, depth <= lower
                    elif depth <= lower:
                        armed = True
                    if armed and depth >= upper:
                        controlled = _controlled(frame, owner, roster)
                        cell = game.bins[period, _index(time, size)]
                        if controlled is True:
                            cell.entries[owner] += 1
                            game.stats["controlled_entries"] += 1
                            armed = False
                        elif controlled is None:
                            cell.bad.add("entries")
                else:
                    entry_owner, armed = None, False
                previous = state
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise ValueError(f"{path.name}, line {line_number}: {error}") from error
    if previous is None:
        raise ValueError("No usable tracking timestamps found.")
    _integrate(game, previous, previous[1] + TRACKING_FRAME_SECONDS, size)


def _rolling_features(game, size, window, half_life):
    raw = {}
    for period, end in sorted(game.ends.items()):
        count = math.ceil(end / size - 1e-9)
        for index in range(count):
            key = (period, index)
            duration = min(size, end - index * size)
            game.durations[key] = duration
            cell = game.bins[key]
            # Coverage can exceed duration by a tiny floating-point error.
            good = cell.coverage >= MIN_COVERAGE * duration
            values = {feature: [None, None] for feature in FEATURES}
            if good and cell.coverage > 0:
                values["territory"] = [v / cell.coverage for v in cell.territory]
                values["box"] = [v / cell.coverage for v in cell.box]
                if "entries" not in cell.bad:
                    values["entries"] = [60 * v / duration for v in cell.entries]
            span = game.event_spans.get(period)
            covered_event_bin = bool(span and index * size < span[1]
                                     and (index + 1) * size > span[0])
            known_dead = cell.dead >= MIN_COVERAGE * duration
            for feature in ("shots", "recoveries"):
                if (game.capabilities[feature] and feature not in cell.bad
                        and (covered_event_bin or known_dead)):
                    values[feature] = [60 * v / duration for v in getattr(cell, feature)]
            if game.capabilities["epv"]:
                for team in (0, 1):
                    if cell.epv_count[team]:
                        values["epv"][team] = cell.epv_sum[team] / cell.epv_count[team]
                    elif good and cell.possession[team] <= 1e-9:
                        values["epv"][team] = 0.0  # Known absence of own possession.
            raw[key] = values

        for index in range(count):
            key = (period, index)
            history = range(max(0, index - window // size + 1), index + 1)
            weighted = [(previous, 2 ** (-((index - previous) * size) / half_life)
                         * game.durations[period, previous]) for previous in history]
            total = sum(weight for _, weight in weighted)
            result = {feature: [None, None] for feature in FEATURES}
            for feature in FEATURES:
                for team in (0, 1):
                    valid = [(raw[period, previous][feature][team], weight)
                             for previous, weight in weighted
                             if raw[period, previous][feature][team] is not None]
                    available = sum(weight for _, weight in valid)
                    enough = available > 0 and (feature == "epv" or available / total >= MIN_COVERAGE)
                    if enough:
                        result[feature][team] = sum(value * weight for value, weight in valid) / available
            game.rolled[key] = result


def _read_game(match_file, size, window, half_life):
    with match_file.open(encoding="utf-8-sig") as source:
        match = json.load(source)
    match_id = str(match["id"])
    if match_file.name != f"{match_id}_match.json":
        raise ValueError("Metadata ID does not match its filename.")
    game = Game(match_id, match)
    team_ids = {str(match["home_team"]["id"]): 0, str(match["away_team"]["id"]): 1}
    if len(team_ids) != 2:
        raise ValueError("Home and away team IDs must differ.")
    for half in match.get("match_periods", []):
        period = int(half["period"])
        duration = _number(half.get("duration_minutes"))
        if period in STARTS and duration is not None and duration > 0:
            game.ends[period] = duration * 60
    folder = match_file.parent
    tracking = folder / f"{match_id}_tracking_extrapolated.jsonl"
    if not tracking.exists():
        raise FileNotFoundError(f"Missing {tracking.name} beside the metadata.")
    phases = _load_phases(folder / f"{match_id}_phases_of_play.csv", team_ids)
    game.notes.append("In-play state from phase intervals." if phases is not None else
                      "No phases CSV: unknown tracking possession is missing, not assumed out of play.")
    _load_events(game, folder / f"{match_id}_dynamic_events.csv", size, team_ids)
    _load_tracking(game, tracking, size, phases, team_ids)
    _rolling_features(game, size, window, half_life)
    return game


def _percentile(values, fraction=.95):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _references(games, path, settings, recalibrate):
    if path is not None and path.exists() and not recalibrate:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("version") != MODEL_VERSION or saved.get("settings") != settings:
            raise ValueError("Normalization settings differ. Use --recalibrate or a separate --normalization-file.")
        scales = saved.get("scales", {})
        if set(scales) != set(FEATURES):
            raise ValueError("Normalization file has unexpected features; use --recalibrate.")
        for feature, value in scales.items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid saved normalization for {feature}.")
        return saved, False
    scales, counts, methods = {}, {}, {}
    for feature in FEATURES:
        values = [pair[team] for game in games for row in game.rolled.values()
                  for pair in [row[feature]] for team in (0, 1) if pair[team] is not None]
        counts[feature] = len(values)
        scale = _percentile(values)
        methods[feature] = "pooled_95th_percentile"
        if scale is None or scale <= 0:
            scale = _percentile([value for value in values if value > 0])
            methods[feature] = "positive_95th_percentile" if scale else "disabled_no_positive_values"
        scales[feature] = scale
    return {"version": MODEL_VERSION, "settings": settings, "scales": scales,
            "methods": methods, "sample_counts": counts,
            "calibration_match_ids": [game.match_id for game in games]}, True


def _clock(seconds):
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _label(period, index, size):
    start = STARTS[period] + index * size
    return {"Period": LABELS[period], "Time": f"{_clock(start)}–{_clock(start + size)}"}


def _write_json(path, value):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, columns):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def calculate_momentum(data_dir=DATA_DIR, output_csv=OUTPUT_CSV,
                       output_markdown=OUTPUT_MARKDOWN, *,
                       bin_seconds=BIN_SECONDS, window_seconds=WINDOW_SECONDS,
                       half_life_seconds=HALF_LIFE_SECONDS, weights=None,
                       normalization_file=NORMALIZATION_FILE, recalibrate=False,
                       output_report=OUTPUT_REPORT, components_output=None,
                       skip_invalid=False):
    """Calculate all games; write the same wide-table layout as the EPV script.

    Returns table rows as dictionaries, with one column per match ID. Values
    are formatted strings; unavailable scores are empty strings. Set any
    output path to None to suppress it. normalization_file=None calibrates
    in memory on this run only. Both teams use identical effective weights.
    """
    if type(bin_seconds) is not int or bin_seconds <= 0:
        raise ValueError("bin_seconds must be a positive integer.")
    if type(window_seconds) is not int or window_seconds < bin_seconds or window_seconds % bin_seconds:
        raise ValueError("window_seconds must be a positive multiple of bin_seconds.")
    if not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be finite and positive.")
    weights = dict(WEIGHTS if weights is None else weights)
    if set(weights) != set(FEATURES) or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError(f"Supply finite nonnegative weights for {FEATURES}.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one feature needs a positive weight.")
    weights = {feature: value / total for feature, value in weights.items()}
    candidates = defaultdict(list)
    for path in sorted(Path(data_dir).rglob("*_match.json")):
        candidates[path.name].append(path)
    if not candidates:
        raise FileNotFoundError(f"No *_match.json files found under {Path(data_dir).resolve()}.")
    games, errors, ignored_duplicates = [], [], []
    for filename, paths in candidates.items():
        match_id = filename.removesuffix("_match.json")
        paths.sort(key=lambda path: (path.parent.name != match_id, str(path)))
        ignored_duplicates.extend(str(path) for path in paths[1:])
        try:
            print(f"Reading match {match_id}…", flush=True)
            games.append(_read_game(paths[0], bin_seconds, window_seconds, half_life_seconds))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error) as error:
            errors.append(f"{paths[0]}: {error}")
    if errors and not skip_invalid:
        raise ValueError("No outputs written; unsupported match files:\n" + "\n".join(errors)
                         + "\nUse --skip-invalid to explicitly omit these matches.")
    if not games:
        raise ValueError("No usable matches.\n" + "\n".join(errors))
    for error in errors:
        print(f"WARNING: Skipped {error}", file=sys.stderr)
    games.sort(key=lambda game: (len(game.match_id), game.match_id))
    settings = {"bin_seconds": bin_seconds, "window_seconds": window_seconds,
                "half_life_seconds": half_life_seconds, "minimum_coverage": MIN_COVERAGE,
                "max_tracking_gap": MAX_TRACKING_GAP, "detected_only": DETECTED_ONLY,
                "entry_hysteresis_metres": ENTRY_HYSTERESIS_METRES,
                "control_distance_metres": CONTROL_DISTANCE_METRES,
                "max_control_height_metres": MAX_CONTROL_HEIGHT_METRES}
    ref_path = Path(normalization_file) if normalization_file is not None else None
    reference, new_reference = _references(games, ref_path, settings, recalibrate)
    scales = reference["scales"]
    results, details, summaries = {}, [], {}
    for game in games:
        values, active_counts = {}, Counter()
        for (period, index), rolling in sorted(game.rolled.items()):
            active = [feature for feature in FEATURES if weights[feature] > 0
                      and scales[feature] is not None
                      and all(value is not None for value in rolling[feature])]
            usable_weight = sum(weights[feature] for feature in active)
            usable = usable_weight >= MIN_ACTIVE_WEIGHT - 1e-9
            score = [None, None]
            normalized = {feature: [None, None] for feature in FEATURES}
            for feature in active:
                active_counts[feature] += 1
                normalized[feature] = [min(1.0, max(0.0, value / scales[feature]))
                                       for value in rolling[feature]]
            if usable:
                score = [100 * sum(weights[feature] * normalized[feature][team] for feature in active)
                         / usable_weight for team in (0, 1)]
            momentum = score[0] - score[1] if usable else None
            values[period, index] = momentum
            if components_output is not None:
                detail = {"Match-ID": game.match_id, **_label(period, index, bin_seconds),
                          "Home score": score[0], "Away score": score[1], "Momentum": momentum,
                          "Available base weight": usable_weight, "Active factors": ";".join(active)}
                for feature in FEATURES:
                    for team, side in enumerate(("Home", "Away")):
                        detail[f"{side} {feature} rolling"] = rolling[feature][team]
                        detail[f"{side} {feature} normalized"] = normalized[feature][team]
                details.append(detail)
        results[game.match_id] = values
        summaries[game.match_id] = {
            "home": game.match["home_team"]["name"], "away": game.match["away_team"]["name"],
            "bins": len(values), "bins_with_momentum": sum(v is not None for v in values.values()),
            "feature_active_bins": dict(active_counts), "source_capabilities": game.capabilities,
            "statistics": dict(game.stats), "notes": game.notes}
    columns = ["Period", "Time", *results]
    keys = sorted({key for values in results.values() for key in values})
    rows = []
    for period, index in keys:
        row = _label(period, index, bin_seconds)
        for match_id, values in results.items():
            value = values.get((period, index))
            row[match_id] = "" if value is None else f"{value:.4f}"
        rows.append(row)
    report = {"version": MODEL_VERSION, "settings": settings, "weights": weights,
              "minimum_active_base_weight": MIN_ACTIVE_WEIGHT, "normalization": reference,
              "skipped_matches": errors, "ignored_duplicate_metadata": ignored_duplicates,
              "games": summaries,
              "interpretation": "Positive favors home; negative favors away. Composite index points, not scoring probabilities.",
              "caution": "Effective weights vary with data availability. Compare games with different active factors cautiously."}
    if not any(value is not None for values in results.values() for value in values.values()):
        raise ValueError("No window has sufficient calibrated common features. No outputs written. "
                         "Check tracking/phase coverage, event schemas and normalization settings.")
    _write_csv(output_csv, rows, columns)
    if output_markdown is not None:
        path = Path(output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["| " + " | ".join(columns) + " |",
                 "| " + " | ".join(["---", "---", *["---:" for _ in results]]) + " |"]
        lines.extend("| " + " | ".join(row[column] for column in columns) + " |" for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if new_reference:
        _write_json(ref_path, reference)
    _write_json(output_report, report)
    if components_output is not None:
        _write_csv(components_output, details, list(details[0]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--csv-output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--markdown-output", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument("--report-output", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--normalization-file", type=Path, default=NORMALIZATION_FILE)
    parser.add_argument("--components-output", type=Path, default=None)
    parser.add_argument("--bin-seconds", type=int, default=BIN_SECONDS)
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS)
    parser.add_argument("--half-life-seconds", type=float, default=HALF_LIFE_SECONDS)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args(argv)
    rows = calculate_momentum(
        args.data_dir, args.csv_output, args.markdown_output,
        bin_seconds=args.bin_seconds, window_seconds=args.window_seconds,
        half_life_seconds=args.half_life_seconds, normalization_file=args.normalization_file,
        recalibrate=args.recalibrate, output_report=args.report_output,
        components_output=args.components_output, skip_invalid=args.skip_invalid)
    print(f"Wrote {len(rows)} time intervals for {len(rows[0]) - 2} games.")
    print(f"CSV: {args.csv_output.resolve()}\nMarkdown: {args.markdown_output.resolve()}")
    print(f"Report: {args.report_output.resolve()}\nNormalization: {args.normalization_file.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
