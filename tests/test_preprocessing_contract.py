"""Regression tests for the pupillometry preprocessing functions."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Pupil_Preprocessing_Functions as methods


def pupil_export(records):
    columns = ["Unnamed: 1", "Unnamed: 84", "Unnamed: 85"]
    rows = [["metadata"] * 3 for _ in range(27)]
    rows.append(["Timestamp", "ET_PupilLeft", "ET_PupilRight"])
    rows.extend(records)
    return pd.DataFrame(rows, columns=columns)


def emotion_export(records):
    rows = [["metadata", "metadata"] for _ in range(27)]
    rows.append(["Timestamp", "EmotionScore"])
    rows.extend(records)
    return pd.DataFrame(rows, columns=["Unnamed: 1", "Unnamed: 25"])


class PupilImportTests(unittest.TestCase):
    def test_keeps_valid_single_eye_samples(self):
        raw = pupil_export([
            ["0", "4.0", "4.2"],
            ["100", "4.1", "4.3"],
            ["200", "4.2", None],
            ["300", None, "4.5"],
        ])
        result = methods.process_pupil_data(raw, 0, 300)
        self.assertEqual(result["Timestamp"].tolist(), [0, 100, 200, 300])
        self.assertTrue(pd.isna(result.iloc[2]["ET_PupilRight"]))
        self.assertTrue(pd.isna(result.iloc[3]["ET_PupilLeft"]))

    def test_removes_missing_timestamps_and_empty_eye_pairs(self):
        raw = pupil_export([
            ["0", "4.0", "4.2"],
            ["100", None, None],
            ["unknown", "4.1", "4.3"],
            ["200", "4.4", None],
        ])
        result = methods.process_pupil_data(raw, 0, 200)
        self.assertEqual(result["Timestamp"].tolist(), [0, 200])

    def test_starts_at_first_valid_paired_observation(self):
        raw = pupil_export([
            ["0", "4.0", None],
            ["100", "4.1", "4.3"],
            ["200", "4.2", None],
        ])
        result = methods.process_pupil_data(raw, 0, 200)
        self.assertEqual(result["Timestamp"].tolist(), [100, 200])

    def test_keeps_first_duplicate_timestamp(self):
        raw = pupil_export([
            ["0", "4.0", "4.2"],
            ["100", "4.1", "4.3"],
            ["100", "5.1", "5.3"],
        ])
        result = methods.process_pupil_data(raw, 0, 100)
        self.assertEqual(result["Timestamp"].tolist(), [0, 100])
        self.assertEqual(result.iloc[1]["ET_PupilLeft"], 4.1)

    def test_rejects_missing_export_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing iMotions columns"):
            methods.process_pupil_data(pd.DataFrame({"Timestamp": [0]}), 0, 100)

    def test_rejects_invalid_task_window(self):
        with self.assertRaisesRegex(ValueError, "earlier than end_time"):
            methods.process_pupil_data(pupil_export([]), 200, 100)

    def test_rejects_recordings_without_paired_observations(self):
        raw = pupil_export([["0", "4.0", None], ["100", None, "4.3"]])
        with self.assertRaisesRegex(ValueError, "No valid paired pupil observation"):
            methods.process_pupil_data(raw, 0, 100)


class SignalProcessingTests(unittest.TestCase):
    def test_finds_valid_400ms_baseline_start(self):
        samples = pd.DataFrame({
            "Timestamp": np.arange(0, 550, 50, dtype=float),
            "ET_PupilLeft": [1.0] + [4.0] * 10,
            "ET_PupilRight": [4.2] * 11,
        })
        self.assertEqual(methods.finds_baseline_period(samples), 100.0)

    def test_filters_each_eye_separately(self):
        data = pd.DataFrame({
            "Timestamp": [0.0, 10.0, 20.0],
            "ET_PupilLeft": [4.0, 1.0, 4.2],
            "ET_PupilRight": [1.0, 4.1, 4.3],
        })
        left, right = methods.remove_invalid_pupil(data.copy())
        self.assertEqual(left["Timestamp"].tolist(), [0, 20])
        self.assertEqual(right["Timestamp"].tolist(), [10, 20])

    def test_mad_keeps_flat_signal(self):
        samples = pd.DataFrame({
            "Timestamp": np.arange(0, 100, 10, dtype=float),
            "ET_PupilLeft": [4.0] * 10,
        })
        result = methods.remove_outliers(samples.copy(), n=20)
        self.assertEqual(len(result), 10)

    def test_moving_average_keeps_flat_signal(self):
        samples = pd.DataFrame({
            "Timestamp": np.arange(0, 300, 10, dtype=float),
            "ET_PupilLeft": [4.0] * 30,
        })
        result = methods.remove_outliers_with_moving_average(
            samples.copy(), window_size=23, n=18
        )
        self.assertEqual(len(result), 30)

    def test_pchip_interpolates_short_gaps(self):
        samples = pd.DataFrame({
            "Timestamp": [0.0, 10.0, 20.0, 30.0],
            "ET_PupilLeft": [4.0, 4.1, 4.2, 4.3],
        })
        result = methods.pchip_interpolation_1000hz(samples.copy(), max_gap=250)
        self.assertIn(15.0, result["Timestamp"].tolist())
        self.assertAlmostEqual(
            result.loc[result["Timestamp"].eq(15), "ET_PupilLeft"].iloc[0],
            4.15,
            places=8,
        )

    def test_pchip_does_not_bridge_gaps_over_250ms(self):
        samples = pd.DataFrame({
            "Timestamp": [0.0, 10.0, 20.0, 310.0, 320.0],
            "ET_PupilLeft": [4.0, 4.1, 4.2, 4.3, 4.4],
        })
        result = methods.pchip_interpolation_1000hz(samples.copy(), max_gap=250)
        timestamps = set(result["Timestamp"].tolist())
        self.assertIn(0, timestamps)
        self.assertIn(310, timestamps)
        self.assertFalse(any(20 < timestamp < 310 for timestamp in timestamps))

    def test_akima_and_makima_respect_long_gaps(self):
        samples = pd.DataFrame({
            "Timestamp": [0.0, 10.0, 20.0, 310.0, 320.0],
            "ET_PupilLeft": [4.0, 4.1, 4.2, 4.3, 4.4],
        })
        for method in (
            methods.akima_interpolation_1000hz,
            methods.makima_interpolation_1000hz,
        ):
            with self.subTest(method=method.__name__):
                result = method(samples.copy(), max_gap=250)
                times = result["Timestamp"].to_numpy()
                self.assertFalse(((times > 20) & (times < 310)).any())

    def test_paired_mean(self):
        left = pd.DataFrame({
            "Timestamp": [0, 1, 2], "ET_PupilLeft": [4.0, 4.1, 4.2]
        })
        right = pd.DataFrame({
            "Timestamp": [0, 1, 2], "ET_PupilRight": [4.2, 4.3, 4.4]
        })
        result = methods.compute_mean_pupil_size(left, right)
        np.testing.assert_allclose(result["meanDia"], [4.1, 4.2, 4.3])

    def test_regression_estimates_missing_eye(self):
        left = pd.DataFrame({
            "Timestamp": [0, 1, 2, 3], "ET_PupilLeft": [4.0, 5.0, 6.0, 7.0]
        })
        right = pd.DataFrame({
            "Timestamp": [0, 1, 2, 4], "ET_PupilRight": [5.0, 6.0, 7.0, 10.0]
        })
        result = methods.compute_mean_pupil_size(left, right).set_index("Timestamp")
        self.assertAlmostEqual(result.loc[3, "ET_PupilRight"], 8.0)
        self.assertAlmostEqual(result.loc[4, "ET_PupilLeft"], 9.0)
        self.assertAlmostEqual(result.loc[3, "meanDia"], 7.5)

    def test_normalization_uses_400ms_baseline(self):
        samples = pd.DataFrame({
            "Timestamp": [0, 100, 200, 300, 500],
            "meanDia": [4.0, 4.2, 4.4, 4.6, 5.0],
        })
        result = methods.normalize_pupil_dilation(samples.copy(), 0, 0, 500)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result.iloc[0]["baseline"], 4.3)
        self.assertAlmostEqual(
            result.iloc[0]["normalized_dilation"], (4.0 - 4.3) / 4.3
        )

    def test_interpolation_comparison_returns_finite_errors(self):
        timestamps = np.arange(0, 700, 10, dtype=float)
        samples = pd.DataFrame({
            "Timestamp": timestamps,
            "ET_PupilLeft": 4.0 + 0.15 * np.sin(timestamps / 120),
        })
        with redirect_stdout(StringIO()):
            result = methods.evaluate_interpolations(samples, random_state=42)
        self.assertEqual(set(result), {"PCHIP", "Akima", "Makima"})
        self.assertTrue(all(np.isfinite(errors).all() for errors in result.values()))


class EmotionIntegrationTests(unittest.TestCase):
    COLUMNS = ["Unnamed: 1", "Unnamed: 25"]

    def test_aligns_observations_in_same_33ms_bin(self):
        raw = emotion_export([["1005", "0.8"], ["1036", "0.9"]])
        pupil = pd.DataFrame({
            "Timestamp": [1001.0, 1002.0, 1034.0, 1035.0],
            "meanDia": [4.0, 4.1, 4.2, 4.3],
        })
        merged = methods.preprocess_integrate_emotions(
            raw, pupil, self.COLUMNS, 1000, 1100
        )
        self.assertEqual(merged["Timestamp"].tolist(), [1000, 1033])
        self.assertTrue(merged[["meanDia", "EmotionScore"]].notna().all(axis=1).all())
        np.testing.assert_allclose(merged["meanDia"], [4.05, 4.25])

    def test_keeps_zero_emotion_score(self):
        raw = emotion_export([["1005", "0"], ["1037", "0.6"]])
        pupil = pd.DataFrame({
            "Timestamp": [1001.0, 1034.0], "meanDia": [4.0, 4.5]
        })
        merged = methods.preprocess_integrate_emotions(
            raw, pupil, self.COLUMNS, 1000, 1100
        )
        self.assertEqual(merged.loc[0, "EmotionScore"], 0)
        self.assertEqual(len(merged), 2)

    def test_outer_merge_keeps_unmatched_bins(self):
        raw = emotion_export([["1068", "0.4"]])
        pupil = pd.DataFrame({"Timestamp": [1001.0], "meanDia": [4.0]})
        merged = methods.preprocess_integrate_emotions(
            raw, pupil, self.COLUMNS, 1000, 1100
        )
        self.assertEqual(merged["Timestamp"].tolist(), [1000, 1066])
        self.assertTrue(pd.isna(merged.iloc[0]["EmotionScore"]))
        self.assertTrue(pd.isna(merged.iloc[1]["meanDia"]))

    def test_uses_first_emotion_value_within_bin(self):
        raw = emotion_export([["1004", "0.25"], ["1010", "0.75"]])
        pupil = pd.DataFrame({"Timestamp": [1001.0], "meanDia": [4.0]})
        merged = methods.preprocess_integrate_emotions(
            raw, pupil, self.COLUMNS, 1000, 1100
        )
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged.iloc[0]["EmotionScore"], 0.25)

    def test_does_not_modify_inputs(self):
        raw = emotion_export([["1005", "0.8"]])
        pupil = pd.DataFrame({"Timestamp": [1001.0], "meanDia": [4.0]})
        raw_copy = raw.copy(deep=True)
        pupil_copy = pupil.copy(deep=True)
        methods.preprocess_integrate_emotions(
            raw, pupil, self.COLUMNS, 1000, 1100
        )
        pd.testing.assert_frame_equal(raw, raw_copy)
        pd.testing.assert_frame_equal(pupil, pupil_copy)

    def test_rejects_invalid_frequency(self):
        raw = emotion_export([["1005", "0.8"]])
        pupil = pd.DataFrame({"Timestamp": [1001.0], "meanDia": [4.0]})
        with self.assertRaisesRegex(ValueError, "target_hz"):
            methods.preprocess_integrate_emotions(
                raw, pupil, self.COLUMNS, 1000, 1100, target_hz=0
            )


class ExampleTests(unittest.TestCase):
    def test_single_recording_example_writes_csv(self):
        columns = ["Unnamed: 1", "Unnamed: 84", "Unnamed: 85"]
        emotion_columns = [
            "Unnamed: 25", "Unnamed: 28", "Unnamed: 32", "Unnamed: 34",
            "Unnamed: 35", "Unnamed: 41", "Unnamed: 27", "Unnamed: 42",
        ]
        columns.extend(emotion_columns)
        header = ["Timestamp", "ET_PupilLeft", "ET_PupilRight"]
        header.extend(f"Emotion{i}" for i in range(len(emotion_columns)))
        rows = [["metadata"] * len(columns) for _ in range(27)]
        rows.append(header)
        for timestamp in range(0, 1000, 10):
            rows.append([timestamp, 4.0, 4.2] + [0.2] * len(emotion_columns))
        raw = pd.DataFrame(rows, columns=columns)

        spec = importlib.util.spec_from_file_location("pupil_example", ROOT / "Example.py")
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)

        with tempfile.TemporaryDirectory() as tmp:
            example.INPUT_CSV = Path(tmp) / "input.csv"
            example.OUTPUT_CSV = Path(tmp) / "output.csv"
            example.START_TIME_MS = 0
            example.END_TIME_MS = 1000
            example.SHOW_PLOTS = False
            raw.to_csv(example.INPUT_CSV, index=False)
            with redirect_stdout(StringIO()):
                example.main()
            output = pd.read_csv(example.OUTPUT_CSV)
            self.assertGreater(len(output), 0)
            self.assertIn("meanDia", output)
            self.assertIn("normalized_dilation", output)
            self.assertIn("Emotion0", output)
            self.assertTrue(output["meanDia"].notna().any())


if __name__ == "__main__":
    unittest.main()
