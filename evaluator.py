"""evaluator.py — TRUSTED SCORER (run by loop_v2.py in a separate process).

The agent-written policy file is never executed as a script and its stdout is
never parsed. Instead this process:

1. Verifies the SHA-256 of prepare.py against the hash pinned by the parent.
2. Imports prepare.py itself and keeps private references to run_simulation
   and the numpy RNG functions it depends on.
3. Imports the policy module (stdout swallowed) and checks none of those
   references were monkey-patched.
4. Either FITS the numeric constants in the policy's PARAMS dict with
   scipy.optimize (``fit`` mode), or SCORES the policy on a list of seeds
   (``score`` mode), wrapping every call to reject NaN/inf outputs and any
   policy that touches the global RNG.
5. Writes a JSON result to the path given by the parent.

This is process isolation plus tamper checks, NOT a security sandbox. Code
written by the LLM still runs with the user's permissions.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
import traceback

import numpy as np


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TamperError(RuntimeError):
    pass


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(run_fn):
    """Objects the simulator relies on. Identity must not change."""
    return {
        "run_simulation": run_fn,
        "run_simulation.__code__": run_fn.__code__,
        "np.random.seed": np.random.seed,
        "np.random.normal": np.random.normal,
        "np.random.uniform": np.random.uniform,
        "np.random.poisson": np.random.poisson,
        "np.clip": np.clip,
        "np.mean": np.mean,
    }


def _check_snapshot(snap, prepare_module):
    current = {
        "run_simulation": prepare_module.run_simulation,
        "run_simulation.__code__": snap["run_simulation"].__code__,
        "np.random.seed": np.random.seed,
        "np.random.normal": np.random.normal,
        "np.random.uniform": np.random.uniform,
        "np.random.poisson": np.random.poisson,
        "np.clip": np.clip,
        "np.mean": np.mean,
    }
    for key, obj in snap.items():
        if current[key] is not obj:
            raise TamperError(f"{key} was modified by the policy module")


def _wrap_policy(policy_fn, params, check_rng: bool):
    """Call policy with params; reject non-finite outputs and RNG use."""

    def wrapped(state):
        if check_rng:
            before = np.random.get_state()
        out = policy_fn(state, params) if params is not None else policy_fn(state)
        out = float(out)
        if not math.isfinite(out):
            raise ValueError(f"policy returned non-finite value {out!r}")
        if check_rng:
            after = np.random.get_state()
            if before[2] != after[2] or not np.array_equal(before[1], after[1]):
                raise TamperError("policy consumed or reseeded the global RNG")
        return out

    return wrapped


def _score(run_fn, policy_fn, params, seeds, check_rng):
    wrapped = _wrap_policy(policy_fn, params, check_rng)
    scores = []
    with contextlib.redirect_stdout(io.StringIO()):
        for s in seeds:
            scores.append(float(run_fn(wrapped, seed=int(s))))
    return scores


def _fit(run_fn, policy_fn, params0, fit_seeds, maxfev):
    from scipy.optimize import minimize

    keys = list(params0.keys())
    x0 = np.array([float(params0[k]) for k in keys])
    n_evals = 0

    def objective(x):
        nonlocal n_evals
        n_evals += 1
        p = dict(zip(keys, map(float, x)))
        try:
            return -float(np.mean(_score(run_fn, policy_fn, p, fit_seeds, check_rng=False)))
        except Exception:
            return 1e6

    f0 = objective(x0)
    res = minimize(objective, x0, method="Powell", options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4})
    if res.fun < f0:
        best_x, best_f = res.x, res.fun
    else:  # optimizer failed to improve: keep the LLM's numbers
        best_x, best_f = x0, f0
    fitted = {k: round(float(v), 6) for k, v in zip(keys, best_x)}
    return fitted, -f0, -best_f, n_evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fit", "score"], required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--prepare", required=True)
    ap.add_argument("--expected-hash", required=True)
    ap.add_argument("--seeds", default="", help="comma-separated seeds (score mode)")
    ap.add_argument("--fit-seeds", default="", help="comma-separated seeds (fit mode)")
    ap.add_argument("--maxfev", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    result = {"ok": False, "mode": args.mode}
    try:
        if sha256_file(args.prepare) != args.expected_hash:
            raise TamperError("prepare.py hash mismatch before evaluation")

        sys.path.insert(0, os.path.dirname(os.path.abspath(args.policy)))
        prepare = _load_module("prepare", args.prepare)
        run_fn = prepare.run_simulation
        snap = _snapshot(run_fn)

        with contextlib.redirect_stdout(io.StringIO()):
            policy_mod = _load_module("candidate_policy", args.policy)
        _check_snapshot(snap, prepare)

        policy_fn = getattr(policy_mod, "autopilot_policy", None)
        if not callable(policy_fn):
            raise ValueError("policy module has no callable autopilot_policy")
        params = getattr(policy_mod, "PARAMS", None)
        if params is not None:
            if not isinstance(params, dict) or not all(
                isinstance(k, str) and isinstance(v, (int, float)) for k, v in params.items()
            ):
                raise ValueError("PARAMS must be a dict[str, float]")
            params = {k: float(v) for k, v in params.items()}
        result["params"] = params

        if args.mode == "fit":
            if not params:
                result.update(fitted=None, note="no PARAMS; nothing to fit")
            else:
                fit_seeds = [int(s) for s in args.fit_seeds.split(",") if s]
                fitted, before, after, n = _fit(run_fn, policy_fn, params, fit_seeds, args.maxfev)
                result.update(fitted=fitted, fit_score_before=before, fit_score_after=after, n_evals=n)
        else:
            seeds = [int(s) for s in args.seeds.split(",") if s]
            result["scores"] = _score(run_fn, policy_fn, params, seeds, check_rng=True)
            result["seeds"] = seeds

        _check_snapshot(snap, prepare)
        if sha256_file(args.prepare) != args.expected_hash:
            raise TamperError("prepare.py hash mismatch after evaluation")
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-2000:]
    result["seconds"] = round(time.time() - t0, 2)

    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
