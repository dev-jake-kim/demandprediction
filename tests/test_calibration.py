from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from calibrate import resolve_train_end, validate_output_path
from models import (
    CalibrationBin,
    CalibrationTable,
    GridDemandConfig,
    GridDemandModel,
    fit_rmse_calibration,
)


def make_table(
    bin_indices: list[int],
    slopes: list[float],
    intercepts: list[float],
    bin_width: float = 0.1,
) -> CalibrationTable:
    rows = tuple(
        CalibrationBin(
            bin_index=bin_index,
            lower_bound=bin_index * bin_width,
            upper_bound=(bin_index + 1) * bin_width,
            count=1,
            a=slope,
            b=intercept,
            raw_rmse=0.0,
            affine_rmse=0.0,
            final_rmse=0.0,
            clamped_count=0,
        )
        for bin_index, slope, intercept in zip(bin_indices, slopes, intercepts, strict=True)
    )
    return CalibrationTable(bin_width=bin_width, bins=rows)


def tiny_config(**kwargs) -> GridDemandConfig:
    return GridDemandConfig(
        H=2,
        W=2,
        a=0,
        d_model=4,
        n_layers=1,
        n_heads=1,
        dim_feedforward=8,
        lstm_hidden=4,
        **kwargs,
    )


class CalibrationTests(unittest.TestCase):
    def test_rmse_fit_recovers_independent_affine_bins(self) -> None:
        predictions = np.array([0.01, 0.02, 0.11, 0.12], dtype=np.float32)
        labels = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)

        table = fit_rmse_calibration(predictions, labels, bin_width=0.1)
        corrected, matched = table.apply_numpy(predictions)

        self.assertEqual(table.bin_indices, [0, 1])
        self.assertTrue(matched.all())
        np.testing.assert_allclose(corrected, labels, rtol=1e-6, atol=1e-6)
        self.assertTrue(
            all(row.final_rmse <= row.affine_rmse <= row.raw_rmse for row in table.bins)
        )

    def test_sparse_lookup_boundaries_gap_and_identity_fallback(self) -> None:
        table = make_table(bin_indices=[0, 2], slopes=[2.0, 3.0], intercepts=[1.0, -1.0])
        predictions = np.array([0.0, 0.099, 0.1, 0.199, 0.2, 0.299, 0.3])

        corrected, matched = table.apply_numpy(predictions)

        np.testing.assert_array_equal(matched, [True, True, False, False, True, True, False])
        np.testing.assert_allclose(
            corrected,
            [1.0, 1.198, 0.1, 0.199, 0.0, 0.0, 0.3],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_constant_prediction_bin_uses_mean_and_clamp_is_nonnegative(self) -> None:
        predictions = np.array([0.05, 0.05, 0.05])
        labels = np.array([0.0, 1.0, 2.0])
        table = fit_rmse_calibration(predictions, labels)

        self.assertEqual(table.bins[0].a, 0.0)
        self.assertAlmostEqual(table.bins[0].b, 1.0)

        negative_table = make_table(bin_indices=[0], slopes=[0.0], intercepts=[-2.0])
        corrected, matched = negative_table.apply_numpy(predictions)
        self.assertTrue(matched.all())
        np.testing.assert_array_equal(corrected, np.zeros(3))

    def test_invalid_bin_width_is_rejected(self) -> None:
        for bin_width in [0.0, -0.1, np.inf, np.nan]:
            with self.subTest(bin_width=bin_width):
                with self.assertRaisesRegex(ValueError, 'bin_width'):
                    fit_rmse_calibration(np.array([0.1]), np.array([0.0]), bin_width)

    def test_invalid_fit_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'shape'):
            fit_rmse_calibration(np.array([0.1, 0.2]), np.array([0.0]))
        with self.assertRaisesRegex(ValueError, 'NaN'):
            fit_rmse_calibration(np.array([np.nan]), np.array([0.0]))
        with self.assertRaisesRegex(ValueError, '0 이상'):
            fit_rmse_calibration(np.array([0.1]), np.array([-1.0]))

    def test_model_applies_table_once_and_can_temporarily_disable_it(self) -> None:
        model = GridDemandModel(tiny_config()).eval()
        model.install_calibration(make_table([0, 2], [2.0, 3.0], [1.0, -1.0]))
        raw = torch.tensor([0.05, 0.15, 0.25])

        corrected, matched = model.apply_calibration(raw)
        torch.testing.assert_close(corrected, torch.tensor([1.1, 0.15, 0.0]))
        torch.testing.assert_close(matched, torch.tensor([True, False, True]))

        with model.calibration_disabled():
            disabled, disabled_matched = model.apply_calibration(raw)
            torch.testing.assert_close(disabled, raw)
            self.assertFalse(disabled_matched.any())

        with self.assertRaisesRegex(ValueError, '이미'):
            model.install_calibration(make_table([0], [1.0], [0.0]))

    def test_model_save_load_round_trip_preserves_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.manual_seed(7)
            model = GridDemandModel(tiny_config()).eval()
            model.install_calibration(
                make_table([0, 1, 2, 3, 4, 5, 6, 7], [0.5] * 8, [0.1] * 8)
            )
            demands = torch.zeros(1, 2, 2, 2)

            with torch.no_grad():
                before = model(demands=demands, return_raw_logits=True)
            model.save_pretrained(temp_dir)
            restored = GridDemandModel.from_pretrained(temp_dir).eval()
            with torch.no_grad():
                after = restored(demands=demands, return_raw_logits=True)

            self.assertTrue(restored.has_calibration)
            torch.testing.assert_close(after['raw_logits'], before['raw_logits'])
            torch.testing.assert_close(after['logits'], before['logits'])

    def test_model_forward_applies_calibration_exactly_once(self) -> None:
        torch.manual_seed(11)
        model = GridDemandModel(tiny_config()).eval()
        demands = torch.zeros(1, 2, 2, 2)
        with torch.no_grad():
            raw = model(demands=demands)['logits']

        bin_indices = sorted(
            set(torch.floor(torch.nextafter(
                raw.to(torch.float64) / 0.1,
                torch.full_like(raw.to(torch.float64), torch.inf),
            )).to(torch.int64).reshape(-1).tolist())
        )
        model.install_calibration(
            make_table(bin_indices, [2.0] * len(bin_indices), [0.0] * len(bin_indices))
        )
        with torch.no_grad():
            output = model(demands=demands, return_raw_logits=True)

        torch.testing.assert_close(output['raw_logits'], raw)
        torch.testing.assert_close(output['logits'], raw * 2)

    def test_legacy_checkpoint_without_calibration_buffers_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = GridDemandModel(tiny_config()).eval()
            legacy_state = {
                name: value
                for name, value in model.state_dict().items()
                if not name.startswith('calibration_')
            }
            model.save_pretrained(temp_dir, state_dict=legacy_state)

            restored = GridDemandModel.from_pretrained(temp_dir).eval()
            self.assertFalse(restored.has_calibration)
            with torch.no_grad():
                output = restored(demands=torch.zeros(1, 2, 2, 2))
            self.assertTrue(torch.isfinite(output['logits']).all())

    def test_no_table_model_remains_uncalibrated(self) -> None:
        model = GridDemandModel(tiny_config()).eval()
        predictions = torch.tensor([0.1, 0.2])
        corrected, matched = model.apply_calibration(predictions)

        self.assertFalse(model.has_calibration)
        torch.testing.assert_close(corrected, predictions)
        self.assertFalse(matched.any())

    def test_output_path_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source'
            source.mkdir()
            with self.assertRaisesRegex(ValueError, '같을 수 없음'):
                validate_output_path(source, source)
            with self.assertRaisesRegex(ValueError, '원본 checkpoint 내부'):
                validate_output_path(source, source / 'calibrated')

            destination = root / 'destination'
            destination.mkdir()
            (destination / 'existing.txt').write_text('occupied', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '비어 있지 않음'):
                validate_output_path(source, destination)

    def test_train_end_matches_time_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            npy_path = Path(temp_dir) / 'grid.npy'
            np.save(npy_path, np.zeros((100, 2, 2), dtype=np.float32))

            self.assertEqual(
                resolve_train_end(npy_path, time_step=24, train_ratio=None, t_end=None),
                (70, 100),
            )
            self.assertEqual(
                resolve_train_end(npy_path, time_step=24, train_ratio=None, t_end=60),
                (60, 100),
            )
            with self.assertRaisesRegex(ValueError, '유효한 train target'):
                resolve_train_end(npy_path, time_step=24, train_ratio=None, t_end=24)


if __name__ == '__main__':
    unittest.main()
