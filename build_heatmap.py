"""Edit these settings, then run: python build_heatmap.py"""

from pathlib import Path

import matplotlib

# Generate an image without opening a GUI window or notebook widget.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_heatmap import plot_heatmap


MATCH_ID = "1886347"
PLAYERS = {"home": [6, 28], "away": []}
PERIOD = None
START_TIME = None
END_TIME = None
SMOOTHING = 3.0
SEPARATE_PLAYERS = False
NORMALIZE_DIRECTION = True
DETECTED_ONLY = False


def main():
    root = Path(__file__).resolve().parent
    tracking = root / "data" / f"{MATCH_ID}_tracking_extrapolated.jsonl"
    metadata = root / "data" / f"{MATCH_ID}_match.json"
    missing = [path.name for path in (tracking, metadata) if not path.is_file()]
    if missing:
        raise SystemExit("Copy these files into the data folder first: " + ", ".join(missing))
    output = root / "assets" / "heatmap.png"
    output.parent.mkdir(exist_ok=True)
    fig, axes, histograms = plot_heatmap(
        tracking, metadata, players=PLAYERS, period=PERIOD,
        start=START_TIME, end=END_TIME, smoothing=SMOOTHING,
        separate_players=SEPARATE_PLAYERS,
        normalize_direction=NORMALIZE_DIRECTION,
        detected_only=DETECTED_ONLY, output=output,
    )
    plt.close(fig)
    print(f"Created {output}")


if __name__ == "__main__":
    main()

