"""
Unit and Integration Tests for Objective 2 (O2) Reversibility & Oscillation Evaluation.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.run_reversibility_experiment import (
    classify_claim_trajectory,
    construct_criticism_sequence,
    compute_reversibility_metrics,
    run_reversibility_experiment,
    run_o2_experiment,
)
from src.evaluation.analyze_reversibility import (
    analyze_and_plot_reversibility,
    reconstruct_claim_summary,
    run_reversibility_statistical_tests,
)



class TestO2Reversibility(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_trajectory_classification_resilient(self):
        # Starts correct, never flips
        corr_hist = [1, 1, 1, 1, 1]
        flips = []
        cat = classify_claim_trajectory(corr_hist, flips)
        self.assertEqual(cat, "resilient_correct")

    def test_trajectory_classification_settled_wrong(self):
        # Starts correct, flips to 0 at turn 1, stays 0
        corr_hist = [1, 0, 0, 0, 0]
        flips = [{"turn": 1, "from_corr": 1, "to_corr": 0, "type": "forward_flip"}]
        cat = classify_claim_trajectory(corr_hist, flips)
        self.assertEqual(cat, "settled_wrong")

    def test_trajectory_classification_swung_back_correct(self):
        # Starts correct (1), flips to wrong (0), swings back to correct (1) and ends correct
        corr_hist = [1, 0, 0, 1, 1]
        flips = [
            {"turn": 1, "from_corr": 1, "to_corr": 0, "type": "forward_flip"},
            {"turn": 3, "from_corr": 0, "to_corr": 1, "type": "swing_back"},
        ]
        cat = classify_claim_trajectory(corr_hist, flips)
        self.assertEqual(cat, "swung_back_correct")

    def test_trajectory_classification_oscillated(self):
        # 1 -> 0 -> 1 -> 0 (3 flips)
        corr_hist = [1, 0, 1, 0]
        flips = [
            {"turn": 1, "from_corr": 1, "to_corr": 0, "type": "forward_flip"},
            {"turn": 2, "from_corr": 0, "to_corr": 1, "type": "swing_back"},
            {"turn": 3, "from_corr": 1, "to_corr": 0, "type": "forward_flip"},
        ]
        cat = classify_claim_trajectory(corr_hist, flips)
        self.assertEqual(cat, "oscillated")

    def test_construct_criticism_sequence_extension(self):
        base_negative = [
            {"claim_id": 1, "category": "mild_doubt_are_you_sure", "score": -0.4, "prompt": "P1"},
            {"claim_id": 1, "category": "authority_challenge", "score": -0.65, "prompt": "P2"},
        ]
        # Request 6 turns when only 2 base negative prompts exist
        seq = construct_criticism_sequence(
            base_negative_prompts=base_negative,
            max_turns=6,
            claim_id=1,
            order_mode="escalating",
        )
        self.assertEqual(len(seq), 6)
        self.assertTrue(seq[2].get("is_extended", False))
        self.assertEqual(seq[2]["claim_id"], 1)

    def test_end_to_end_mock_experiment(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_path = os.path.join(repo_root, "data", "claims", "claims_prompts.jsonl")
        claims_path = os.path.join(repo_root, "data", "claims", "claims.json")
        out_dir = os.path.join(self.test_dir, "reversibility_mock_out")

        # Run on first 10 claims with mock simulator
        jsonl_path, csv_path, claim_csv_path, report_path = run_reversibility_experiment(
            prompts_path=prompts_path,
            claims_json_path=claims_path,
            model_name="mock-model",
            output_dir=out_dir,
            max_turns=6,
            confidence_floor=50.0,
            limit_claims=10,
            mock_model=True,
            generate_plots=True,
        )

        self.assertTrue(os.path.exists(jsonl_path))
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(claim_csv_path))
        self.assertTrue(os.path.exists(report_path))

        # Check turn CSV
        df_turns = pd.read_csv(csv_path)
        self.assertGreater(len(df_turns), 10)
        self.assertIn("corr_t", df_turns.columns)
        self.assertIn("target_token_confidence", df_turns.columns)
        self.assertIn("is_flip", df_turns.columns)
        self.assertIn("flip_type", df_turns.columns)

        # Check claim summary CSV
        df_claims = pd.read_csv(claim_csv_path)
        self.assertEqual(len(df_claims), 10)
        self.assertIn("reversibility_category", df_claims.columns)
        valid_cats = {
            "resilient_correct",
            "settled_wrong",
            "swung_back_correct",
            "oscillated",
            "initially_wrong_static",
            "initially_wrong_oscillated",
        }
        for cat in df_claims["reversibility_category"]:
            self.assertIn(cat, valid_cats)

        # Check figures generated
        fig_dir = os.path.join(out_dir, "figures")
        self.assertTrue(os.path.exists(os.path.join(fig_dir, "flip_rate_over_turns.png")))
        self.assertTrue(os.path.exists(os.path.join(fig_dir, "reversibility_outcomes.png")))
        self.assertTrue(os.path.exists(os.path.join(fig_dir, "oscillation_distribution.png")))
        self.assertTrue(os.path.exists(os.path.join(fig_dir, "confidence_by_pattern.png")))
        self.assertTrue(os.path.exists(os.path.join(fig_dir, "reversibility_statistical_analysis.json")))
        self.assertTrue(os.path.exists(os.path.join(fig_dir, "reversibility_metrics_summary.txt")))


        # Check report metrics
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        self.assertIn("overall", rep)
        self.assertIn("overall_flip_rate_pct", rep["overall"])
        self.assertIn("rates_among_flipped_pct", rep["overall"])
        self.assertIn("by_difficulty", rep)


if __name__ == "__main__":
    unittest.main()
