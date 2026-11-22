"""Unit & integration test suite for autoresearch-autopilot CI/CD release gates."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ci_gate import (
    HERE,
    PREPARE_PATH,
    TRAIN_PATH,
    holm_bonferroni,
    run_ci_gate,
)
from loop import (
    check_contract,
    paired_test,
    pinned_prepare_hash,
    run_evaluator,
    structure_hash,
)


class TestAutoresearchCIGate(unittest.TestCase):
    def test_01_sha256_benchmark_lock_and_tamper_rejection(self):
        """Verify prepare.py matches prepare.sha256 and evaluator aborts on wrong hash."""
        expected = pinned_prepare_hash()
        self.assertEqual(len(expected), 64)

        # Passing an invalid expected hash must trigger defense-in-depth SystemExit in run_evaluator
        bad_hash = "0" * 64
        with self.assertRaises(SystemExit):
            run_evaluator(TRAIN_PATH, "score", bad_hash, seeds=[42])

    def test_02_rng_tamper_detection_blocks_cheating_policy(self):
        """Verify a policy that reseeds or touches numpy's global RNG is caught."""
        expected = pinned_prepare_hash()
        cheating_code = """\
import numpy as np
PARAMS = {"k": 0.0}

def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    np.random.seed(999)
    return 0.0
"""
        fd, tmp_path = tempfile.mkstemp(suffix="_cheat.py")
        os.close(fd)
        try:
            with open(tmp_path, "w") as f:
                f.write(cheating_code)
            res = run_evaluator(tmp_path, "score", expected, seeds=[42])
            self.assertFalse(res.get("ok"))
            self.assertIn("TamperError", res.get("error", ""))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_03_ast_structure_hash_and_contract_validation(self):
        """Verify AST structure hash ignores numeric constant tweaks & docstrings."""
        code_a = '''\
PARAMS = {"a": 1.5, "b": 2.0}
def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    """Docstring version A."""
    return params["a"] * state["room_temp_f"] + params["b"]
'''
        code_b = '''\
PARAMS = {"a": 99.9, "b": 42.123}
def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    """Completely different docstring and constants."""
    return params["a"] * state["room_temp_f"] + params["b"]
'''
        self.assertIsNone(check_contract(code_a))
        self.assertIsNone(check_contract(code_b))
        self.assertEqual(structure_hash(code_a), structure_hash(code_b))

        # Missing PARAMS must fail contract check
        bad_code = "def autopilot_policy(state): return 0.0\n"
        self.assertIsNotNone(check_contract(bad_code))

    def test_04_paired_ttest_and_holm_bonferroni_fwer(self):
        """Verify 20-seed paired t-test LCB and Holm-Bonferroni FWER step-down."""
        incumbent = [10.0 + i * 0.1 for i in range(20)]
        candidate = [x + 2.5 for x in incumbent]
        res = paired_test(candidate, incumbent)
        self.assertTrue(res["accept"])
        self.assertAlmostEqual(res["mean_gain"], 2.5, places=5)
        self.assertGreater(res["lcb"], 2.4)

        fwer = holm_bonferroni([1e-8, 5e-7, 0.002], alpha=0.01)
        self.assertTrue(fwer["all_significant"])
        self.assertLess(fwer["max_adjusted_p"], 0.01)

    def test_05_end_to_end_ci_gate_and_mlflow_tracking(self):
        """Run full 41-seed CI release gate on train.py and verify MLflow output."""
        fd, out_json = tempfile.mkstemp(suffix="_ci_report.json")
        os.close(fd)
        try:
            report = run_ci_gate(policy_path=TRAIN_PATH, out_path=out_json)
            self.assertTrue(report["all_gates_passed"])
            self.assertGreater(report["metrics"]["accept_mean_sri"], 39.70)
            self.assertGreater(report["metrics"]["holdout_mean_sri"], 39.50)
            self.assertGreater(report["metrics"]["seed42_sri"], 39.30)
            self.assertLess(report["metrics"]["baseline_seed42_sri"], -29.0)
            self.assertLess(report["metrics"]["holm_max_adjusted_p"], 1e-10)
            self.assertTrue(os.path.exists(out_json))
            with open(out_json) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["mlflow_run_id"], report["mlflow_run_id"])
        finally:
            if os.path.exists(out_json):
                os.remove(out_json)


if __name__ == "__main__":
    unittest.main()
