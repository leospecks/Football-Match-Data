"""Heatmaps from SkillCorner match JSON and raw tracking JSONL.

Install: python -m pip install matplotlib numpy
Edit the SETTINGS below, then run: python plot_heatmap.py
In Jupyter, import plot_heatmap() and call it as shown at the bottom.
"""

import json
from math import ceil, isfinite

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle


# ---------------- SETTINGS: edit these variables ----------------
TRACKING_FILE = "1886347_tracking_extrapolated.jsonl"
MATCH_FILE = "1886347_match.json"

# Select SHIRT NUMBERS, grouped by team. None = everyone on that team;
# [] or an omitted team = nobody. Example for all home players: {"home": None}
PLAYERS = {"home": [6, 28], "away": []}

PERIOD = None                 # None = all halves; 1 or 2 = one half
START_TIME = None             # Optional raw timestamp, e.g. "00:15:30"
END_TIME = None               # Optional raw timestamp, e.g. "00:20:00"
SEPARATE_PLAYERS = False      # False = combined map; True = map per player
NORMALIZE_DIRECTION = True   # Orient each team to attack left-to-right
DETECTED_ONLY = False        # True excludes extrapolated player positions
BINS = (104, 68)             # Finer grid along pitch length and width
SMOOTHING = 3.0              # Gaussian standard deviation in METRES; 0 disables
OUTPUT_FILE = None           # Optional image path, e.g. "heatmap.png"
# ---------------------------------------------------------------


def _seconds(value):
    if isinstance(value, str):
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
    else:
        seconds = float(value)
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Times must be finite and nonnegative.")
    return seconds


def _select_players(match, players):
    if not isinstance(players, dict) or any(k not in ("home", "away") for k in players):
        raise ValueError("players must map 'home' and/or 'away' to shirt-number lists or None.")
    selected = {}
    for side, numbers in players.items():
        team = match[f"{side}_team"]
        roster = [p for p in match["players"] if p["team_id"] == team["id"]]
        if numbers is not None:
            if not isinstance(numbers, (list, tuple, set)) or any(type(n) is not int for n in numbers):
                raise ValueError("Use integer shirt numbers, e.g. {'home': [6, 28]}.")
            unknown = set(numbers) - {p["number"] for p in roster}
            if unknown:
                raise ValueError(f"Unknown {side} shirt numbers: {sorted(unknown)}")
        for player in roster:
            if numbers is None or player["number"] in numbers:
                selected[player["id"]] = {**player, "side": side, "team_name": team["name"]}
    if not selected:
        raise ValueError("Select at least one player in players.")
    return selected


def _draw_pitch(ax, length, width):
    ax.set_facecolor("#143d2a")

    def box(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor="white", linewidth=1, zorder=3))

    box(-length / 2, -width / 2, length, width)
    ax.plot([0, 0], [-width / 2, width / 2], color="white", linewidth=1, zorder=3)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor="white", zorder=3))
    ax.plot(0, 0, "w.", markersize=3, zorder=3)
    for side in (-1, 1):
        goal = side * length / 2
        for depth, span in ((16.5, 40.32), (5.5, 18.32)):
            box(goal if side < 0 else goal - depth, -span / 2, depth, span)
        ax.plot(goal - side * 11, 0, "w.", markersize=3, zorder=3)
    ax.set(xlim=(-length / 2, length / 2), ylim=(-width / 2, width / 2),
           aspect="equal", xticks=[], yticks=[])


def _smooth_histogram(histogram, smoothing, length, width):
    """Apply a separable Gaussian blur using NumPy; preserve total mass.

    Smoothing is specified in metres, so changing the number of bins does not
    change the physical blur width. Symmetric padding keeps density inside
    the pitch. The raw input histogram is never modified. No SciPy required.
    """
    result = histogram.astype(float, copy=True)
    if smoothing == 0 or result.sum() == 0:
        return result
    sigmas = (smoothing * result.shape[0] / length,
              smoothing * result.shape[1] / width)
    for axis, sigma in enumerate(sigmas):
        radius = max(1, ceil(3 * sigma))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="symmetric")
        result = np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"), axis, padded)
    result *= float(histogram.sum()) / result.sum()
    return result


def plot_heatmap(tracking_file, match_file, players, period=None,
                 start=None, end=None, separate_players=False,
                 normalize_direction=True, detected_only=False,
                 bins=(104, 68), output=None, smoothing=3.0):
    """Plot where selected players are most frequently recorded.

    players: {"home": [6, 28], "away": [17]}; use None for a team's full roster.
    period: 1 or 2, or None for all halves. start/end use the raw file clock,
    including fractional seconds; no halftime offsets are added or removed.

    The JSONL is read once, line by line. Only small per-player histograms are
    retained. Invalid/out-of-pitch positions are excluded. Set detected_only
    to remove extrapolated positions; by default these positions are included.

    normalize_direction rotates a team's right-to-left frames by 180 degrees,
    aligning both halves and both teams to attack toward the right. This needs
    home_team_side in the match metadata. False preserves stadium coordinates.

    Cell color = percent of included position samples in that cell. Combined
    maps pool all samples, so players observed longer contribute more. Separate
    maps normalize each player independently and use a shared color scale.
    This is an observation-frequency map, not possession-only or time-weighted.

    smoothing is the Gaussian standard deviation in metres (default 3).
    Larger values produce broader, smoother hotspots; 0 shows raw cells.
    Smoothing estimates density in neighboring cells. Bilinear interpolation
    removes visible cell edges when smoothing is enabled. Only the displayed
    density is smoothed; the returned integer histograms remain unchanged.

    Returns (fig, axes, histograms). axes is a list; histograms maps player IDs
    to RAW integer arrays shaped (x bins, y bins). A player's zero array means
    no usable observations in this interval. Such players get no separate plot.
    """
    if period is not None and (type(period) is not int or period < 1):
        raise ValueError("period must be a positive integer or None.")
    start = _seconds(start) if start is not None else None
    end = _seconds(end) if end is not None else None
    if start is not None and end is not None and end <= start:
        raise ValueError("end must be later than start.")
    if len(bins) != 2 or any(type(n) is not int or n < 1 for n in bins):
        raise ValueError("bins must contain two positive integers.")
    if not isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative (metres).")
    with open(match_file, encoding="utf-8") as source:
        match = json.load(source)
    selected = _select_players(match, players)
    length, width = float(match["pitch_length"]), float(match["pitch_width"])
    if not all(isfinite(d) and d > 0 for d in (length, width)):
        raise ValueError("Pitch dimensions must be finite and positive.")
    histograms = {pid: np.zeros(bins, dtype=np.int64) for pid in selected}
    sides = match.get("home_team_side") or []
    directions = {}
    for index, direction in enumerate(sides, 1):
        if direction in ("left_to_right", "right_to_left"):
            directions[index] = direction == "right_to_left"

    with open(tracking_file, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            frame = json.loads(line)
            frame_period = frame.get("period")
            if frame_period is None or frame.get("timestamp") is None:
                continue
            if period is not None and frame_period != period:
                continue
            time = _seconds(frame["timestamp"])
            if (start is not None and time < start) or (end is not None and time > end):
                continue
            for p in frame.get("player_data", []):
                pid = p["player_id"]
                if pid not in selected or (detected_only and p.get("is_detected") is not True):
                    continue
                x, y = p.get("x"), p.get("y")
                if not all(isinstance(v, (int, float)) and isfinite(v) for v in (x, y)):
                    continue
                if normalize_direction:
                    if frame_period not in directions:
                        raise ValueError("Missing home_team_side for this period. Set normalize_direction=False.")
                    flip = directions[frame_period] ^ (selected[pid]["side"] == "away")
                    if flip:
                        x, y = -x, -y
                if abs(x) > length / 2 or abs(y) > width / 2:
                    continue
                ix = min(bins[0] - 1, int((x + length / 2) / length * bins[0]))
                iy = min(bins[1] - 1, int((y + width / 2) / width * bins[1]))
                histograms[pid][ix, iy] += 1

    observed = {pid: h for pid, h in histograms.items() if h.sum() > 0}
    if not observed:
        raise ValueError("No usable positions for the selected players, time interval and filters.")
    if separate_players:
        panels = []
        for pid, histogram in observed.items():
            p = selected[pid]
            name = p.get("short_name") or " ".join(
                str(p.get(k) or "") for k in ("first_name", "last_name")).strip()
            panels.append((f"{p['team_name']} | #{p['number']} {name}", histogram))
    else:
        labels = []
        for side in ("home", "away"):
            numbers = sorted({selected[pid]["number"] for pid in observed if selected[pid]["side"] == side})
            if numbers:
                shirts = ", ".join(f"#{n}" for n in numbers)
                labels.append(f"{side.title()}: {shirts}" if len(shirts) < 60 else f"{side.title()}: {len(numbers)} players")
        panels = [(" | ".join(labels), sum(observed.values(), np.zeros(bins, dtype=np.int64)))]

    percentages = [_smooth_histogram(histogram, smoothing, length, width)
                   * 100 / histogram.sum() for _, histogram in panels]
    norm = Normalize(vmin=0, vmax=max(float(h.max()) for h in percentages))
    columns = min(2, len(panels))
    rows = ceil(len(panels) / columns)
    fig, grid = plt.subplots(rows, columns, figsize=(10 * columns, 6.5 * rows),
                              squeeze=False, layout="constrained")
    axes = list(grid.flat[:len(panels)])
    for ax in grid.flat[len(panels):]:
        ax.remove()
    for ax, (title, histogram), percent in zip(axes, panels, percentages):
        _draw_pitch(ax, length, width)
        # Draw zero density too when smoothed: masking it would leave a jagged
        # boundary between the dark color map and the green pitch background.
        display = (percent.T if smoothing > 0
                   else np.ma.masked_where(percent.T <= 0, percent.T))
        image = ax.imshow(display,
                           extent=(-length / 2, length / 2, -width / 2, width / 2),
                           origin="lower", cmap="inferno", norm=norm,
                           interpolation="bilinear" if smoothing > 0 else "nearest", zorder=1)
        ax.set_title(title + f"\n{int(histogram.sum()):,} recorded positions", fontsize=11)
    color_label = ("Smoothed share of positions per cell (%)" if smoothing > 0
                   else "Share of recorded positions per cell (%)")
    fig.colorbar(image, ax=axes, label=color_label, shrink=0.75)
    description = f"Period {period}" if period is not None else "All periods"
    orientation = "Each team attacks right" if normalize_direction else "Original pitch coordinates"
    blur_label = f" | Gaussian smoothing: {smoothing:g} m" if smoothing > 0 else ""
    fig.suptitle(f"Player position heatmap | {description}\n{orientation}{blur_label} | Data: SkillCorner", fontsize=14)
    if output is not None:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig, axes, histograms


if __name__ == "__main__":
    fig, axes, histograms = plot_heatmap(
        TRACKING_FILE, MATCH_FILE, players=PLAYERS, period=PERIOD,
        start=START_TIME, end=END_TIME, separate_players=SEPARATE_PLAYERS,
        normalize_direction=NORMALIZE_DIRECTION, detected_only=DETECTED_ONLY,
        bins=BINS, output=OUTPUT_FILE, smoothing=SMOOTHING,
    )
    plt.show()
