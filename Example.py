"""Single-recording example for the published pupillometry preprocessing workflow.

The input must be an authorized iMotions 10 CSV export in the layout expected
by Pupil_Preprocessing_Functions.py. No research data is distributed here.

Before running:
    1. Install requirements: python -m pip install -r requirements.txt
    2. Put your authorized CSV file in this directory (or edit INPUT_CSV).
    3. Set the recording-specific task window and emotion-column selection.
    4. Run: python Example.py

This example is not the original 25-participant batch-processing script and
is not guaranteed to reproduce every figure or number in the paper.
"""

from pathlib import Path

import pandas as pd

import Pupil_Preprocessing_Functions as methods


# ---- Recording-specific settings: edit before running -----------------------
INPUT_CSV = Path("data.csv")
OUTPUT_CSV = Path("result.csv")

# Annotated task window, in milliseconds in the original iMotions time base.
# These are values from the historical example, NOT defaults for other files.
START_TIME_MS = 77740
END_TIME_MS = 1192364

# Original export-column names used for the selected facial-expression data.
# These positions depend on the iMotions export layout; update for your file.
EMOTION_COLUMNS = [
    "Unnamed: 1", "Unnamed: 25", "Unnamed: 28", "Unnamed: 32",
    "Unnamed: 34", "Unnamed: 35", "Unnamed: 41", "Unnamed: 27",
    "Unnamed: 42",
]

# Set False to skip interactive diagnostic plots (e.g., on a server).
SHOW_PLOTS = True


# ---- Example pipeline -------------------------------------------------------
def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input not found: {INPUT_CSV}. "
            "Add an authorized iMotions CSV export or edit INPUT_CSV. "
            "The repository does not include participant data."
        )
    if START_TIME_MS >= END_TIME_MS:
        raise SystemExit("START_TIME_MS must be earlier than END_TIME_MS.")

    raw = pd.read_csv(INPUT_CSV, low_memory=False)

    # 1. Select pupil channels and remove invalid/duplicate timestamps.
    # The helper expects the specific iMotions header/preamble format used
    # in the study, not an arbitrary three-column CSV file.
    try:
        filtered = methods.process_pupil_data(raw, START_TIME_MS, END_TIME_MS)
    except IndexError as exc:
        raise ValueError(
            "No valid paired pupil measurements found. Check the input export "
            "layout and task window."
        ) from exc
    if len(filtered) < 5:
        raise ValueError("Too few valid pupil rows for this example workflow.")

    # The study uses a valid pre-event window as a baseline proxy; it was
    # not a separately recorded, controlled baseline condition.
    try:
        baseline_start = methods.finds_baseline_period(
            filtered, min_pupil=2, max_pupil=8, baseline_period=400
        )
    except IndexError as exc:
        raise ValueError(
            "Could not find a valid 400 ms baseline window in this recording."
        ) from exc
    left, right = methods.remove_invalid_pupil(
        filtered, min_size=2, max_size=8
    )

    # 2. Apply two complementary artifact filters to each eye.
    # The asymmetrical MAD multipliers are retained from the published
    # single-recording example. Do not treat them as universal thresholds.
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
        raise ValueError(
            "Too few pupil samples remain after filtering for interpolation."
        )

    if SHOW_PLOTS:
        # Auto-scaled axes avoid the unrelated, hard-coded plot limits
        # present in the earlier example.
        methods.plot_left_pupil(left, pupil_column="ET_PupilLeft")

    # 3. Interpolate on a computational 1000 Hz grid, withholding
    # reconstructions in source gaps longer than 250 ms.
    left = methods.pchip_interpolation_1000hz(
        left, pupil_column="ET_PupilLeft", interp_upsampling_freq=1000,
        max_gap=250
    )
    right = methods.pchip_interpolation_1000hz(
        right, pupil_column="ET_PupilRight", interp_upsampling_freq=1000,
        max_gap=250
    )

    if SHOW_PLOTS:
        methods.plot_left_pupil(left, pupil_column="ET_PupilLeft")

    # 4. Estimate selected single-eye missing samples from paired observations;
    # normalize mean pupil diameter against the selected baseline window.
    binocular = methods.compute_mean_pupil_size(left, right)
    normalized = methods.normalize_pupil_dilation(
        binocular, baseline_start, START_TIME_MS, END_TIME_MS
    )

    # 5. Aggregate pupil measurements toward ~30 Hz and outer-join selected
    # facial-expression channels. This does not guarantee perfectly matched
    # timestamps across modalities; check alignment before further analysis.
    integrated = methods.preprocess_integrate_emotions(
        raw, normalized, EMOTION_COLUMNS, START_TIME_MS, END_TIME_MS
    )

    if SHOW_PLOTS:
        methods.plot_pupil_diameter(
            integrated, START_TIME_MS, END_TIME_MS
        )

    integrated.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(integrated)} rows to {OUTPUT_CSV.resolve()}")
    print(integrated.head())


if __name__ == "__main__":
    main()
