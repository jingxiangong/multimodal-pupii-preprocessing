"""Run the pupillometry preprocessing workflow on one iMotions CSV export.

Set the recording-specific options below before running. The original
participant data are not included in this repository.
"""

from pathlib import Path

import pandas as pd

import Pupil_Preprocessing_Functions as methods


INPUT_CSV = Path("data.csv")
OUTPUT_CSV = Path("result.csv")

# Task boundaries in the iMotions timestamp (milliseconds).
START_TIME_MS = 77740
END_TIME_MS = 1192364

# Column positions from the original iMotions export.
EMOTION_COLUMNS = [
    "Unnamed: 1", "Unnamed: 25", "Unnamed: 28", "Unnamed: 32",
    "Unnamed: 34", "Unnamed: 35", "Unnamed: 41", "Unnamed: 27",
    "Unnamed: 42",
]

SHOW_PLOTS = True


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(f"Input file not found: {INPUT_CSV}")
    if START_TIME_MS >= END_TIME_MS:
        raise ValueError("START_TIME_MS must be earlier than END_TIME_MS")

    raw = pd.read_csv(INPUT_CSV, low_memory=False)
    filtered = methods.process_pupil_data(raw, START_TIME_MS, END_TIME_MS)
    if len(filtered) < 5:
        raise ValueError("Not enough valid pupil records for this example")

    # Use a valid 400 ms pre-event window as the reference.
    try:
        baseline_start = methods.finds_baseline_period(
            filtered, min_pupil=2, max_pupil=8, baseline_period=400
        )
    except IndexError as exc:
        raise ValueError("No valid baseline window found in this recording") from exc

    left, right = methods.remove_invalid_pupil(filtered, min_size=2, max_size=8)

    # Retain the MAD settings from the historical single-recording example.
    left = methods.remove_outliers(
        left, pupil_column="ET_PupilLeft", time_column="Timestamp", n=20
    )
    left = methods.remove_outliers_with_moving_average(
        left, pupil_column="ET_PupilLeft", window_size=23, n=18
    )
    right = methods.remove_outliers(
        right, pupil_column="ET_PupilRight", time_column="Timestamp", n=16
    )
    right = methods.remove_outliers_with_moving_average(
        right, pupil_column="ET_PupilRight", window_size=23, n=18
    )
    if len(left) < 3 or len(right) < 3:
        raise ValueError("Not enough pupil records remain after filtering")

    if SHOW_PLOTS:
        methods.plot_left_pupil(left, pupil_column="ET_PupilLeft")

    # Interpolate the two eyes separately; do not fill gaps longer than 250 ms.
    left = methods.pchip_interpolation_1000hz(
        left, pupil_column="ET_PupilLeft", interp_upsampling_freq=1000, max_gap=250
    )
    right = methods.pchip_interpolation_1000hz(
        right, pupil_column="ET_PupilRight", interp_upsampling_freq=1000, max_gap=250
    )

    if SHOW_PLOTS:
        methods.plot_left_pupil(left, pupil_column="ET_PupilLeft")

    binocular = methods.compute_mean_pupil_size(left, right)
    normalized = methods.normalize_pupil_dilation(
        binocular, baseline_start, START_TIME_MS, END_TIME_MS
    )
    integrated = methods.preprocess_integrate_emotions(
        raw, normalized, EMOTION_COLUMNS, START_TIME_MS, END_TIME_MS
    )

    if SHOW_PLOTS:
        methods.plot_pupil_diameter(integrated, START_TIME_MS, END_TIME_MS)

    integrated.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(integrated)} rows to {OUTPUT_CSV.resolve()}")
    print(integrated.head())


if __name__ == "__main__":
    main()
