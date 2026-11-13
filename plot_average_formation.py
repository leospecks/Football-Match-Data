"""Plot a team's mean player positions from SkillCorner raw files.

Install: python -m pip install matplotlib
Edit SETTINGS below, then run: python plot_average_formation.py
Standalone: does not import plot_match.py or plot_heatmap.py.
"""

import csv
import json
from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.patches import Circle, Rectangle


# ---------------- SETTINGS: edit these variables ----------------
PROJECT_DIR = Path(__file__).resolve().parent
MATCH_ID = "2007721"
MATCH_FOLDER = PROJECT_DIR / "data" / MATCH_ID
TRACKING_FILE = MATCH_FOLDER / f"{MATCH_ID}_tracking_extrapolated.jsonl"
MATCH_FILE = MATCH_FOLDER / f"{MATCH_ID}_match.json"

TEAM = "home"                 # "home" or "away"
PLAYERS = None                 # None = everyone observed; or shirt numbers [6, 28]
PERIOD = None                  # None = all periods; 1 or 2 = one half
START_TIME = None              # Optional raw timestamp, e.g. "00:15:00"
END_TIME = None                # Optional raw timestamp, e.g. "00:30:00"
NORMALIZE_DIRECTION = True     # Align all halves to attack toward the right
DETECTED_ONLY = False          # True excludes extrapolated player positions
SHOW_NAMES = True
OUTPUT_FILE = None             # Optional image path, e.g. "assets/average_formation.png"
OUTPUT_CSV = None              # Optional table path, e.g. "average_positions.csv"
# ---------------------------------------------------------------


def _seconds(value):
    total = 0.0
    for part in str(value).strip().split(":"):
        total = total * 60 + float(part)
    if not isfinite(total) or total < 0:
        raise ValueError("Times must be finite and nonnegative.")
    return total


def _draw_pitch(ax, length, width):
    ax.set_facecolor("#287a42")

    def box(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor="white", linewidth=1.2, zorder=1))

    box(-length / 2, -width / 2, length, width)
    ax.plot([0, 0], [-width / 2, width / 2], color="white", linewidth=1.2, zorder=1)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor="white", zorder=1))
    ax.plot(0, 0, "w.", zorder=1)
    for side in (-1, 1):
        goal = side * length / 2
        for depth, span in ((16.5, 40.32), (5.5, 18.32)):
            box(goal if side < 0 else goal - depth, -span / 2, depth, span)
        ax.plot(goal - side * 11, 0, "w.", zorder=1)
        box(goal if side > 0 else goal - 2, -3.66, 2, 7.32)
    ax.set(xlim=(-length / 2 - 4, length / 2 + 4),
           ylim=(-width / 2 - 4, width / 2 + 4),
           aspect="equal", xticks=[], yticks=[])


def plot_average_formation(tracking_file, match_file, team="home", *,
                           players=None, period=None, start=None, end=None,
                           normalize_direction=True, detected_only=False,
                           show_names=True, output=None, output_csv=None):
    """Return (fig, ax, averages) for one team's mean player coordinates.

    team: "home" or "away". players: optional integer shirt-number list.
    period: one positive integer, or None for all periods. start/end follow
    the raw timestamp clock, without halftime offsets. A time filter applies
    to each selected period's stored clock, so specify period when necessary.

    Each player's x/y is the arithmetic mean of their valid tracking samples.
    This is suitable for SkillCorner's regular 10 Hz data; irregularly spaced
    samples are not time-weighted. Missing coordinates are omitted, never
    replaced by zero. Nonfinite or out-of-pitch coordinates are also omitted.
    By default extrapolated positions and all valid game states are included,
    including stoppages. No in-play filter is applied. The JSONL is read one
    line at a time and only per-player sums are retained.

    normalize_direction rotates right-to-left periods by 180 degrees so that
    both x and y remain consistent relative to the team's attacking direction.
    It needs home_team_side in the metadata. False uses stadium coordinates.

    averages is a list of dictionaries with player_id, number, name, x, y,
    sample_count and estimated_sample_count. Substitutes have their own mean;
    this can produce more than 11 plotted players over a whole match. Players
    with no valid samples are omitted. Labels may overlap for similar means;
    restrict players/time or turn show_names off to simplify the plot.
    """
    if team not in ("home", "away"):
        raise ValueError("team must be 'home' or 'away'.")
    if period is not None and (type(period) is not int or period < 1):
        raise ValueError("period must be a positive integer or None.")
    start = _seconds(start) if start is not None else None
    end = _seconds(end) if end is not None else None
    if start is not None and end is not None and end <= start:
        raise ValueError("end must be later than start.")
    with Path(match_file).open(encoding="utf-8-sig") as source:
        match = json.load(source)
    selected_team = match[f"{team}_team"]
    roster = {str(p["id"]): p for p in match["players"]
              if str(p["team_id"]) == str(selected_team["id"])}
    if players is not None:
        if not isinstance(players, (list, tuple, set)) or any(type(n) is not int for n in players):
            raise ValueError("players must be integer shirt numbers, e.g. [6, 28], or None.")
        unknown = set(players) - {p["number"] for p in roster.values()}
        if unknown:
            raise ValueError(f"Unknown {team} shirt numbers: {sorted(unknown)}")
        roster = {pid: p for pid, p in roster.items() if p["number"] in players}
    if not roster:
        raise ValueError("No players selected for this team.")
    length, width = float(match["pitch_length"]), float(match["pitch_width"])
    if not all(isfinite(value) and value > 0 for value in (length, width)):
        raise ValueError("Pitch dimensions must be finite and positive.")
    sides = match.get("home_team_side") or []
    sums = {pid: [0.0, 0.0, 0, 0] for pid in roster}
    with Path(tracking_file).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if line.startswith("version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Tracking file is a Git LFS pointer. Re-download with fetch_data.py --overwrite.")
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{tracking_file}, line {line_number}: {error}") from error
            frame_period = frame.get("period")
            if frame_period is None or frame.get("timestamp") is None:
                continue
            if period is not None and frame_period != period:
                continue
            time = _seconds(frame["timestamp"])
            if (start is not None and time < start) or (end is not None and time > end):
                continue
            sign = 1
            if normalize_direction:
                direction = sides[frame_period - 1] if 1 <= frame_period <= len(sides) else None
                if direction not in ("left_to_right", "right_to_left"):
                    raise ValueError("Missing home_team_side for this period. Set normalize_direction=False.")
                sign = 1 if direction == "left_to_right" else -1
                if team == "away":
                    sign *= -1
            for p in frame.get("player_data", []):
                pid = str(p["player_id"])
                if pid not in sums or (detected_only and p.get("is_detected") is not True):
                    continue
                x, y = p.get("x"), p.get("y")
                if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and isfinite(value) for value in (x, y)):
                    continue
                if abs(x) > length / 2 or abs(y) > width / 2:
                    continue
                sums[pid][0] += sign * x
                sums[pid][1] += sign * y
                sums[pid][2] += 1
                sums[pid][3] += p.get("is_detected") is False

    averages = []
    for pid, (sum_x, sum_y, count, estimated) in sums.items():
        if count == 0:
            continue
        p = roster[pid]
        name = p.get("short_name") or f"{p.get('first_name') or ''} {p.get('last_name') or ''}"
        averages.append({"player_id": p["id"], "number": p["number"],
                         "name": " ".join(name.split()),
                         "x": sum_x / count, "y": sum_y / count,
                         "sample_count": count, "estimated_sample_count": estimated})
    averages.sort(key=lambda row: (row["number"], str(row["player_id"])))
    if not averages:
        raise ValueError("No valid positions for this team with the selected filters.")

    kit = match.get(f"{team}_team_kit") or {}
    shirt_color = kit.get("jersey_color") or ("#ef4444" if team == "home" else "#2563eb")
    number_color = kit.get("number_color") or "#ffffff"
    fig, ax = plt.subplots(figsize=(11, 7.5))
    _draw_pitch(ax, length, width)
    ax.scatter([p["x"] for p in averages], [p["y"] for p in averages],
               s=500, color=shirt_color, edgecolors=number_color,
               linewidths=1.8, zorder=3)
    for p in averages:
        ax.text(p["x"], p["y"], str(p["number"]), color=number_color,
                ha="center", va="center", weight="bold", fontsize=10, zorder=4)
        if show_names and p["name"]:
            label = ax.annotate(p["name"], (p["x"], p["y"]), xytext=(0, -19),
                                textcoords="offset points", ha="center", va="top",
                                color="white", fontsize=8.5, zorder=4)
            label.set_path_effects([effects.withStroke(linewidth=2, foreground="#18552c")])
    if normalize_direction:
        ax.annotate("Attack", xy=(length * .35, width / 2 + 2),
                    xytext=(length * .10, width / 2 + 2), color="white",
                    ha="left", va="center", fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "white"})
    period_label = "All periods" if period is None else f"Period {period}"
    if start is not None or end is not None:
        period_label += f" | Raw clock: {start if start is not None else 'start'}–{end if end is not None else 'end'} s"
    game_label = f"{match['home_team']['name']} vs. {match['away_team']['name']}"
    fig.suptitle(f"{selected_team['name']} — average positions", fontsize=15, weight="bold")
    ax.set_title(f"{game_label}\n{period_label} | {len(averages)} players", fontsize=10, pad=12)
    source_label = "Detected positions only" if detected_only else "Includes extrapolated positions"
    fig.text(.5, .025, f"Means of valid tracking samples | {source_label} | Data: SkillCorner",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(averages[0]))
            writer.writeheader()
            writer.writerows(averages)
    return fig, ax, averages


if __name__ == "__main__":
    fig, ax, averages = plot_average_formation(
        TRACKING_FILE, MATCH_FILE, team=TEAM, players=PLAYERS, period=PERIOD,
        start=START_TIME, end=END_TIME, normalize_direction=NORMALIZE_DIRECTION,
        detected_only=DETECTED_ONLY, show_names=SHOW_NAMES,
        output=OUTPUT_FILE, output_csv=OUTPUT_CSV,
    )
    plt.show()
