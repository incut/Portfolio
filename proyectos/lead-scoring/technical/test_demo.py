"""Deterministic checks, no private files, models or network access."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from demo import build_pipeline, evaluate, ranking_metrics, write_results
from generate_synthetic_data import generate


class DemoTests(unittest.TestCase):
    def test_synthetic_schema_and_reproducibility(self):
        frame = generate(120)
        self.assertEqual(frame.shape, (120, 21))
        self.assertTrue(frame.id.is_unique)
        self.assertTrue(frame.id.str.startswith("synthetic-").all())
        pd.testing.assert_frame_equal(frame, generate(120))

    def test_lift_auc_ceiling_and_ties(self):
        y = np.array([0, 1] * 10)
        result = ranking_metrics(y, y.astype(float))
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["lift_at_10pct"], 2.0)
        self.assertEqual(ranking_metrics(y, np.ones(20))["positives_top_k"], 1)
        self.assertEqual(ranking_metrics(np.r_[y, 1], np.arange(21))["k"], 3)

    def test_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics(np.array([0, 1]), np.array([0.1, np.nan]))

    def test_form_excludes_activity_and_does_not_fit_unseen_categories(self):
        frame = generate(120)
        pipeline = build_pipeline(False).fit(frame, frame.compra)
        expected = pipeline.predict_proba(frame)
        changed = frame.assign(ult_actividad="SMS Sent", score_actividad=999)
        np.testing.assert_allclose(expected, pipeline.predict_proba(changed))
        encoder = pipeline["prepare"].named_transformers_["category"]["encode"]
        before = [values.copy() for values in encoder.categories_]
        pipeline.predict_proba(frame.assign(origen="unseen"))
        for left, right in zip(before, encoder.categories_):
            np.testing.assert_array_equal(left, right)

    def test_nested_evaluation_preserves_rows_and_has_no_personal_outputs(self):
        frame = generate(120)
        before = frame.copy(deep=True)
        result = evaluate(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(set(result["scenarios"]), {"form", "expanded_unverified"})
        self.assertEqual(len(result["fold_metrics"]), 10)
        self.assertNotIn("synthetic-", json.dumps(result))
        for metrics in result["scenarios"].values():
            self.assertEqual(metrics["n"], len(frame))
            self.assertEqual(metrics["coverage"], 1.0)

    def test_output_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            write_results(path, {"status": "synthetic"})
            with self.assertRaises(FileExistsError):
                write_results(path, {"status": "overwrite"})
            self.assertEqual(json.loads(path.read_text())["status"], "synthetic")


if __name__ == "__main__":
    unittest.main()
