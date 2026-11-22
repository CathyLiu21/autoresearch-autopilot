"""ci_gate.py — CI/CD Statistical Release Gate & MLflow Experiment Tracker.

Enforces production release gates on train.py before allowing PR merge or deployment:
1. SHA-256 benchmark integrity lock (prepare.py vs. prepare.sha256).
2. Static AST contract & structure-hash verification (max 12 PARAMS, no forbidden globals).
3. Isolated-process 41-seed evaluation (20 ACCEPT_SEEDS + 20 HOLDOUT_SEEDS + seed 42)
   with runtime RNG & monkey-patch tamper detection via evaluator.py.
4. 20-seed paired t-test (95% LCB) + Holm-Bonferroni step-down FWER control across
   acceptance and holdout cohorts against the zero-offset baseline.
5. MLflow experiment tracking (logs PARAMS, metrics, and ci_gate_report.json to mlruns/).
"""
import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import uuid

import numpy as np
from scipy import stats

from loop import (
    ACCEPT_SEEDS,
    HOLDOUT_SEEDS,
    PREPARE_LOCK,
    PREPARE_PATH,
    REPORT_SEED,
    check_contract,
    paired_test,
    pinned_prepare_hash,
    run_evaluator,
    score_all,
    sha256_file,
    structure_hash,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "train.py")
DEFAULT_REPORT_PATH = os.path.join(HERE, "runs", "ci_gate_report.json")
MLRUNS_DIR = os.path.join(HERE, "mlruns")

# Release gate thresholds (calibrated against commit e58b051 incumbent)
MIN_ACCEPT_MEAN_SRI = 39.70
MIN_HOLDOUT_MEAN_SRI = 39.50
MIN_SEED42_SRI = 39.30
MAX_FWER_ALPHA = 0.01

BASELINE_POLICY_CODE = """\
PARAMS = {"offset": 0.0}

def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    return float(params["offset"])
"""


def holm_bonferroni(p_values: list[float], alpha: float = MAX_FWER_ALPHA) -> dict:
    """Step-down Holm-Bonferroni FWER correction across m hypotheses."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    rejected = [False] * m
    running_max = 0.0
    all_reject = True
    for rank, (orig_idx, p) in enumerate(indexed):
        multiplier = m - rank
        adj_p = min(1.0, p * multiplier)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
        threshold = alpha / multiplier
        if all_reject and bool(p <= threshold):
            rejected[orig_idx] = True
        else:
            all_reject = False
            rejected[orig_idx] = False
    return {
        "raw_p_values": [float(p) for p in p_values],
        "adjusted_p_values": [float(p) for p in adjusted],
        "rejected_null": [bool(r) for r in rejected],
        "all_significant": bool(all(rejected)),
        "max_adjusted_p": float(max(adjusted)) if adjusted else 1.0,
        "alpha": float(alpha),
    }


def paired_p_value_one_sided(candidate: list[float], baseline: list[float]) -> float:
    """One-sided paired t-test p-value testing H1: candidate > baseline."""
    d = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    n = len(d)
    if n < 2:
        return 1.0
    t_stat, p_two = stats.ttest_1samp(d, popmean=0.0)
    if t_stat <= 0:
        return 1.0
    return float(p_two / 2.0)


def log_to_mlflow(run_name: str, params: dict, metrics: dict, report_path: str) -> str:
    """Log run to MLflow if installed, and always write standardized mlruns/ structure."""
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    run_id = uuid.uuid4().hex
    try:
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment("autoresearch-autopilot-ci-gate")
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.log_params({k: round(float(v), 6) if isinstance(v, (int, float)) else str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            if os.path.exists(report_path):
                mlflow.log_artifact(report_path)
            return run_id
    except ImportError:
        # Zero-dependency MLflow-compatible local file store layout
        exp_dir = os.path.join(MLRUNS_DIR, "0", run_id)
        os.makedirs(os.path.join(exp_dir, "params"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "artifacts"), exist_ok=True)
        ts_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
        for k, v in params.items():
            with open(os.path.join(exp_dir, "params", str(k)), "w") as f:
                f.write(str(v))
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                with open(os.path.join(exp_dir, "metrics", str(k)), "w") as f:
                    f.write(f"{ts_ms} {float(v)} 1\n")
        meta = {
            "run_id": run_id,
            "run_name": run_name,
            "status": "FINISHED",
            "start_time": ts_ms,
            "end_time": ts_ms,
            "artifact_uri": os.path.join(exp_dir, "artifacts"),
        }
        with open(os.path.join(exp_dir, "meta.yaml"), "w") as f:
            for mk, mv in meta.items():
                f.write(f"{mk}: {mv}\n")
        return run_id


def run_ci_gate(policy_path: str = TRAIN_PATH, out_path: str = DEFAULT_REPORT_PATH) -> dict:
    # 1. Verify SHA-256 lock on prepare.py
    expected_hash = pinned_prepare_hash()
    actual_hash = sha256_file(PREPARE_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 lock mismatch: {actual_hash} != {expected_hash}")

    # 2. Static AST contract check & structure hash
    with open(policy_path) as f:
        code = f.read()
    contract_err = check_contract(code)
    if contract_err:
        raise ValueError(f"AST Contract Violation: {contract_err}")
    s_hash = structure_hash(code)

    # 3. Evaluate candidate policy across all 41 disjoint seeds
    raw_res, candidate_scores = score_all(policy_path, expected_hash)
    if not raw_res.get("ok") or candidate_scores is None:
        raise RuntimeError(f"Candidate evaluation failed: {raw_res.get('error')}")

    # 4. Evaluate zero-action baseline across the exact same 41 seeds
    fd, tmp_base = tempfile.mkstemp(suffix="_baseline.py")
    os.close(fd)
    try:
        with open(tmp_base, "w") as f:
            f.write(BASELINE_POLICY_CODE)
        base_res, baseline_scores = score_all(tmp_base, expected_hash)
        if not base_res.get("ok") or baseline_scores is None:
            raise RuntimeError(f"Baseline evaluation failed: {base_res.get('error')}")
    finally:
        if os.path.exists(tmp_base):
            os.remove(tmp_base)

    # 5. Compute paired t-tests and Holm-Bonferroni FWER across Acceptance & Holdout cohorts
    accept_stats = paired_test(candidate_scores["accept"], baseline_scores["accept"])
    holdout_stats = paired_test(candidate_scores["holdout"], baseline_scores["holdout"])

    p_accept = paired_p_value_one_sided(candidate_scores["accept"], baseline_scores["accept"])
    p_holdout = paired_p_value_one_sided(candidate_scores["holdout"], baseline_scores["holdout"])
    fwer = holm_bonferroni([p_accept, p_holdout], alpha=MAX_FWER_ALPHA)

    accept_mean = float(np.mean(candidate_scores["accept"]))
    holdout_mean = float(np.mean(candidate_scores["holdout"]))
    seed42_sri = float(candidate_scores["seed42"])
    baseline_seed42 = float(baseline_scores["seed42"])

    gates = {
        "sha256_lock_verified": bool(actual_hash == expected_hash),
        "ast_contract_valid": bool(contract_err is None),
        "accept_sri_above_threshold": bool(accept_mean >= MIN_ACCEPT_MEAN_SRI),
        "holdout_sri_above_threshold": bool(holdout_mean >= MIN_HOLDOUT_MEAN_SRI),
        "seed42_sri_above_threshold": bool(seed42_sri >= MIN_SEED42_SRI),
        "paired_lcb_positive": bool(holdout_stats["lcb"] > 0.0 and accept_stats["lcb"] > 0.0),
        "holm_bonferroni_fwer_passed": bool(fwer["all_significant"]),
    }
    all_passed = bool(all(gates.values()))

    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "all_gates_passed": all_passed,
        "prepare_sha256": expected_hash,
        "policy_structure_hash": s_hash,
        "params": raw_res.get("params", {}),
        "metrics": {
            "accept_mean_sri": round(accept_mean, 5),
            "holdout_mean_sri": round(holdout_mean, 5),
            "seed42_sri": round(seed42_sri, 5),
            "baseline_seed42_sri": round(baseline_seed42, 5),
            "accept_paired_gain_mean": round(accept_stats["mean_gain"], 5),
            "accept_paired_gain_lcb95": round(accept_stats["lcb"], 5),
            "holdout_paired_gain_mean": round(holdout_stats["mean_gain"], 5),
            "holdout_paired_gain_lcb95": round(holdout_stats["lcb"], 5),
            "holm_max_adjusted_p": fwer["max_adjusted_p"],
        },
        "holm_bonferroni": fwer,
        "gates": gates,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    mlflow_params = dict(raw_res.get("params") or {})
    mlflow_params["structure_hash"] = s_hash
    mlflow_params["prepare_sha256"] = expected_hash[:16]
    run_id = log_to_mlflow("ci_release_gate", mlflow_params, report["metrics"], out_path)
    report["mlflow_run_id"] = run_id

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser(description="Run CI/CD Statistical Release Gate for autoresearch-autopilot")
    ap.add_argument("--policy", default=TRAIN_PATH)
    ap.add_argument("--out", default=DEFAULT_REPORT_PATH)
    ap.add_argument("--strict", action="store_true", help="Exit with code 1 if any gate fails")
    args = ap.parse_args()

    report = run_ci_gate(policy_path=args.policy, out_path=args.out)
    m = report["metrics"]
    print(
        f"[CI GATE] status={'PASS' if report['all_gates_passed'] else 'FAIL'} | "
        f"seed42={m['seed42_sri']:.2f} (baseline={m['baseline_seed42_sri']:.2f}) | "
        f"holdout_mean={m['holdout_mean_sri']:.2f} (LCB95=+{m['holdout_paired_gain_lcb95']:.2f}) | "
        f"Holm-p={m['holm_max_adjusted_p']:.2e} | mlflow_run={report['mlflow_run_id'][:12]}"
    )
    if args.strict and not report["all_gates_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
