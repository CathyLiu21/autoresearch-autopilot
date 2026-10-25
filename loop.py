"""loop_v2.py — AUTORESEARCH RATCHET, v2.

Improvements over loop.py (v1):

1. Trusted multi-seed scoring. Candidates are scored by evaluator.py in a
   separate process that imports the policy itself (stdout is never parsed),
   prepare.py is pinned by SHA-256, and a candidate is accepted only if its
   paired, seed-by-seed gain over the incumbent has a 95% lower confidence
   bound above MIN_EFFECT. Final numbers are reported on held-out seeds that
   are never used for fitting or acceptance.
2. Experiment memory. Every proposal (rationale, diff vs. incumbent, scores,
   decision) is appended to experiments.jsonl and summarized back into the
   prompt. Proposals whose structure (AST with numeric constants erased)
   matches an earlier one are skipped without evaluation.
3. LLM picks the structure, an optimizer picks the numbers. Policies expose a
   PARAMS dict; scipy.optimize fits it on dedicated fitting seeds before the
   candidate is scored.

Rejected candidates never touch train.py (they live in runs/<name>/candidates/),
so there is no `git reset --hard`; only train.py is ever committed.

Usage:
    uv run python loop_v2.py --iterations 10          # Vertex AI
    uv run python loop_v2.py --mock                   # offline smoke test
"""
import argparse
import ast
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PREPARE_PATH = os.path.join(HERE, "prepare.py")
EVALUATOR_PATH = os.path.join(HERE, "evaluator.py")
PREPARE_LOCK = os.path.join(HERE, "prepare.sha256")

# Disjoint seed sets: fitting, acceptance, and held-out reporting.
FIT_SEEDS = list(range(1000, 1003))
ACCEPT_SEEDS = list(range(0, 20))
HOLDOUT_SEEDS = list(range(5000, 5020))
REPORT_SEED = 42  # for continuity with v1 numbers only

MIN_EFFECT = 0.05      # smallest SRI gain worth accepting
CONFIDENCE = 0.95      # two-sided level for the paired-t lower bound
MAX_PARAMS = 12
EVAL_TIMEOUT_S = 900

PROJECT_ID = "gemini-build-day-481122"
LOCATION = "global"
MODEL_NAME = "gemini-3.8-flash"


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pinned_prepare_hash():
    """Pin prepare.py: first run writes prepare.sha256, later runs must match."""
    current = sha256_file(PREPARE_PATH)
    if os.path.exists(PREPARE_LOCK):
        pinned = open(PREPARE_LOCK).read().strip()
        if pinned != current:
            sys.exit(f"ABORT: prepare.py hash {current[:12]} != pinned {pinned[:12]}")
    else:
        with open(PREPARE_LOCK, "w") as f:
            f.write(current + "\n")
        print(f"Pinned prepare.py sha256 {current[:12]}… to {os.path.basename(PREPARE_LOCK)}")
    return current


# --------------------------------------------------------------------------- #
# Trusted evaluation (separate process)
# --------------------------------------------------------------------------- #
def run_evaluator(policy_path, mode, expected_hash, seeds=None, maxfev=500):
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    cmd = [
        sys.executable, EVALUATOR_PATH,
        "--mode", mode,
        "--policy", policy_path,
        "--prepare", PREPARE_PATH,
        "--expected-hash", expected_hash,
        "--out", out_path,
        "--maxfev", str(maxfev),
    ]
    key = "--fit-seeds" if mode == "fit" else "--seeds"
    cmd += [key, ",".join(map(str, seeds or []))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT_S, cwd=HERE)
        with open(out_path) as f:
            content = f.read()
        if not content:
            return {"ok": False, "error": f"evaluator produced no result (rc={proc.returncode}): {proc.stderr[-500:]}"}
        result = json.loads(content)
    except subprocess.TimeoutExpired:
        result = {"ok": False, "error": f"evaluation timed out after {EVAL_TIMEOUT_S}s"}
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    # Defense in depth: the parent re-checks the benchmark on disk.
    if sha256_file(PREPARE_PATH) != expected_hash:
        sys.exit("ABORT: prepare.py changed on disk during evaluation")
    return result


def score_all(policy_path, expected_hash):
    seeds = ACCEPT_SEEDS + HOLDOUT_SEEDS + [REPORT_SEED]
    res = run_evaluator(policy_path, "score", expected_hash, seeds=seeds)
    if not res.get("ok"):
        return res, None
    s = res["scores"]
    n_a, n_h = len(ACCEPT_SEEDS), len(HOLDOUT_SEEDS)
    return res, {
        "accept": s[:n_a],
        "holdout": s[n_a:n_a + n_h],
        "seed42": s[-1],
    }


def paired_test(candidate, incumbent):
    d = np.asarray(candidate) - np.asarray(incumbent)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(1 - (1 - CONFIDENCE) / 2, df=max(n - 1, 1)))
    lcb = mean - tcrit * se
    return {"mean_gain": mean, "sd_gain": sd, "lcb": lcb, "accept": lcb > MIN_EFFECT}


# --------------------------------------------------------------------------- #
# Code utilities: structure hashing, PARAMS rewriting, contract checks
# --------------------------------------------------------------------------- #
class _EraseNumbers(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def structure_hash(code):
    """Hash of the policy's form: comments, docstrings and all numbers erased."""
    tree = _strip_docstrings(ast.parse(code))
    tree = _EraseNumbers().visit(tree)
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()[:16]


def _find_params_node(tree):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in targets):
                return node
    return None


def check_contract(code):
    """Static checks before spending compute. Returns an error string or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "autopilot_policy"]
    if not fn:
        return "missing top-level autopilot_policy"
    node = _find_params_node(tree)
    if node is None:
        return "missing top-level PARAMS dict"
    if not isinstance(node.value, ast.Dict):
        return "PARAMS must be a dict literal"
    if len(node.value.keys) > MAX_PARAMS:
        return f"PARAMS has {len(node.value.keys)} entries (max {MAX_PARAMS})"
    if len(fn[0].args.args) < 2:
        return "autopilot_policy must accept (state, params=PARAMS)"
    return None


def rewrite_params(code, fitted):
    """Replace the PARAMS literal with fitted values, preserving everything else."""
    tree = ast.parse(code)
    node = _find_params_node(tree)
    lines = code.splitlines(keepends=True)
    body = ",\n".join(f"    {k!r}: {v!r}" for k, v in fitted.items())
    new_block = "PARAMS = {  # fitted by scipy.optimize (Powell) in loop_v2.py\n" + body + ",\n}\n"
    return "".join(lines[: node.lineno - 1]) + new_block + "".join(lines[node.end_lineno:])


def short_diff(old, new, max_lines=30):
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1))[2:]
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... ({len(diff) - max_lines} more diff lines)"]
    return "\n".join(diff)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class ExperimentLog:
    def __init__(self, path):
        self.path = path
        self.entries = []
        if os.path.exists(path):
            with open(path) as f:
                self.entries = [json.loads(line) for line in f if line.strip()]

    def append(self, entry):
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def seen(self, s_hash):
        for e in self.entries:
            if e.get("structure_hash") == s_hash:
                return e
        return None

    def prompt_summary(self, n_total=15):
        if not self.entries:
            return "No previous attempts yet."
        valid = [e for e in self.entries if e.get("status") not in ("LLM_ERROR", "PARSE_ERROR")]
        if not valid:
            return "No previous attempts yet."
        recent = valid[-n_total:]
        out = []
        for e in recent:
            head = f"#{e['iteration']} {e['status']}"
            if e.get("mean_gain") is not None:
                head += f"  gain={e['mean_gain']:+.3f} (95% LCB {e['lcb']:+.3f})"
            if e.get("accept_mean") is not None:
                head += f"  accept-mean={e['accept_mean']:.3f}"
            if e.get("reason"):
                head += f"  reason: {e['reason']}"
            out.append(head)
            if e.get("rationale"):
                out.append(f"  hypothesis: {e['rationale'][:250]}")
            if e.get("fitted_params"):
                out.append(f"  params: {json.dumps(e['fitted_params'])}")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Proposers
# --------------------------------------------------------------------------- #
RESPONSE_RE = re.compile(r"RATIONALE:\s*(.*?)\s*```python\s*\n(.*?)\n```", re.DOTALL)


def parse_response(text):
    m = RESPONSE_RE.search(text or "")
    if m:
        return m.group(1).strip(), m.group(2)
    code = re.search(r"```python\s*\n(.*?)\n```", text or "", re.DOTALL)
    return "", (code.group(1) if code else None)


class VertexProposer:
    def __init__(self):
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            for p in ("service_account.json", "../service_account.json"):
                if os.path.exists(os.path.join(HERE, p)):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(os.path.join(HERE, p))
                    break
        from google import genai
        self.client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            http_options={"timeout": 180000},
        )

    def __call__(self, prompt, iteration):
        import time
        max_retries = 3
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
                return resp.text
            except Exception as e:
                last_err = e
                print(f"  [Vertex AI API retry {attempt}/{max_retries} due to {type(e).__name__}: {e}]")
                time.sleep(3 * attempt)
        raise RuntimeError(f"Vertex AI failed after {max_retries} attempts: {last_err}")


class MockProposer:
    """Replays canned responses (mock_proposals.py) to exercise every code path."""

    def __init__(self):
        from mock_proposals import PROPOSALS
        self.proposals = PROPOSALS

    def __call__(self, prompt, iteration):
        return self.proposals[(iteration - 1) % len(self.proposals)]


def build_prompt(program_doc, incumbent_code, incumbent, log):
    return f"""You are an expert ML research engineer optimizing an IoT thermal-control policy.

{program_doc}

## Current Incumbent
Acceptance-seed mean SRI: {np.mean(incumbent['accept']):.4f}
Held-out mean SRI: {np.mean(incumbent['holdout']):.4f}

```python
{incumbent_code}
```

## Experiment Log (most recent last; do NOT repeat these structures)
{log.prompt_summary()}

Propose ONE new policy structure that is meaningfully different from every attempt above.
"""


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=HERE)


def write_history(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--maxfev", type=int, default=500, help="optimizer evaluations per fit")
    ap.add_argument("--mock", action="store_true", help="offline: canned proposals, scratch policy copy, no commits")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()

    run_name = args.run_name or ("mock" if args.mock else "v2")
    run_dir = os.path.join(HERE, "runs", run_name)
    cand_dir = os.path.join(run_dir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    commit = not (args.mock or args.no_commit)

    expected_hash = pinned_prepare_hash()

    policy_path = os.path.join(HERE, "train.py")
    if args.mock:
        scratch = os.path.join(run_dir, "train.py")
        shutil.copy(policy_path, scratch)
        policy_path = scratch
    elif commit:
        if git("diff", "--quiet", "--", "train.py").returncode != 0:
            sys.exit("ABORT: train.py has uncommitted changes; commit or stash them first.")

    log = ExperimentLog(os.path.join(run_dir, "experiments.jsonl"))
    history_path = os.path.join(run_dir, "history.json")
    history = json.load(open(history_path)) if os.path.exists(history_path) else []
    start_iter = (max((e["iteration"] for e in log.entries), default=0)) + 1

    program_doc = open(os.path.join(HERE, "program.md")).read()
    proposer = MockProposer() if args.mock else VertexProposer()

    incumbent_code = open(policy_path).read()
    res, incumbent = score_all(policy_path, expected_hash)
    if incumbent is None:
        sys.exit(f"ABORT: incumbent failed to evaluate: {res.get('error')}")
    print(f"Incumbent: accept-mean {np.mean(incumbent['accept']):.4f} | "
          f"held-out {np.mean(incumbent['holdout']):.4f} ± {np.std(incumbent['holdout'], ddof=1):.4f} | "
          f"seed42 {incumbent['seed42']:.4f}\n")
    if not history:
        history.append({
            "iteration": 0, "candidate_sri": round(float(np.mean(incumbent["accept"])), 5),
            "best_sri": round(float(np.mean(incumbent["accept"])), 5), "status": "BASELINE",
            "commit": git("rev-parse", "--short", "HEAD").stdout.strip() or None,
            "description": "Incumbent at start of v2 run",
            "holdout_mean": round(float(np.mean(incumbent["holdout"])), 5),
            "seed42": round(incumbent["seed42"], 5),
        })
        write_history(history_path, history)

    for i in range(start_iter, start_iter + args.iterations):
        print(f"=== Iteration {i} ===")
        entry = {"iteration": i, "timestamp": dt.datetime.now().isoformat(timespec="seconds")}

        try:
            text = proposer(build_prompt(program_doc, incumbent_code, incumbent, log), i)
        except Exception as e:
            entry.update(status="LLM_ERROR", reason=f"{type(e).__name__}: {e}")
            log.append(entry); print(f"  {entry['reason']}"); continue

        rationale, code = parse_response(text)
        entry["rationale"] = rationale
        if code is None:
            entry.update(status="PARSE_ERROR", reason="no ```python block in response")
            log.append(entry); print("  no code block"); continue

        problem = check_contract(code)
        if problem:
            entry.update(status="CONTRACT_ERROR", reason=problem, diff=short_diff(incumbent_code, code))
            log.append(entry); print(f"  contract: {problem}"); continue

        s_hash = structure_hash(code)
        entry["structure_hash"] = s_hash
        prior = log.seen(s_hash)
        if prior is not None:
            entry.update(status="DUPLICATE", reason=f"same structure as #{prior['iteration']} (only numbers differ)")
            log.append(entry); print(f"  {entry['reason']} — skipped"); continue

        cand_path = os.path.join(cand_dir, f"iter_{i:03d}.py")
        with open(cand_path, "w") as f:
            f.write(code)

        # (3) optimizer picks the numbers
        fit = run_evaluator(cand_path, "fit", expected_hash, seeds=FIT_SEEDS, maxfev=args.maxfev)
        if not fit.get("ok"):
            entry.update(status="EVAL_ERROR", reason=fit.get("error"), diff=short_diff(incumbent_code, code))
            log.append(entry); print(f"  fit failed: {fit.get('error')}"); continue
        if fit.get("fitted"):
            code = rewrite_params(code, fit["fitted"])
            with open(cand_path, "w") as f:
                f.write(code)
            entry.update(initial_params=fit["params"], fitted_params=fit["fitted"],
                         fit_gain=round(fit["fit_score_after"] - fit["fit_score_before"], 4),
                         fit_evals=fit["n_evals"])
            print(f"  fit: {fit['fit_score_before']:.3f} -> {fit['fit_score_after']:.3f} "
                  f"on fit seeds ({fit['n_evals']} evals, {fit['seconds']}s)")
        entry["diff"] = short_diff(incumbent_code, code)

        # (1) trusted multi-seed scoring of the exact file that would be committed
        res, cand = score_all(cand_path, expected_hash)
        if cand is None:
            entry.update(status="EVAL_ERROR", reason=res.get("error"))
            log.append(entry); print(f"  score failed: {res.get('error')}"); continue

        test = paired_test(cand["accept"], incumbent["accept"])
        entry.update(
            mean_gain=round(test["mean_gain"], 5), sd_gain=round(test["sd_gain"], 5), lcb=round(test["lcb"], 5),
            accept_mean=round(float(np.mean(cand["accept"])), 5),
            holdout_mean=round(float(np.mean(cand["holdout"])), 5),
            holdout_sd=round(float(np.std(cand["holdout"], ddof=1)), 5),
            seed42=round(cand["seed42"], 5),
        )
        print(f"  accept-mean {entry['accept_mean']:.4f}  gain {test['mean_gain']:+.4f} "
              f"(LCB {test['lcb']:+.4f}, need > {MIN_EFFECT})  held-out {entry['holdout_mean']:.4f}")

        commit_id = None
        if test["accept"]:
            entry["status"] = "ACCEPTED"
            incumbent_code, incumbent = code, cand
            target = policy_path
            with open(target, "w") as f:
                f.write(code)
            if commit:
                git("add", "train.py")
                msg = f"sri-v2: {entry['accept_mean']:.4f} accept / {entry['holdout_mean']:.4f} held-out (iter {i})"
                git("commit", "-m", msg)
                commit_id = git("rev-parse", "--short", "HEAD").stdout.strip()
                entry["commit"] = commit_id
            print(f"  ACCEPTED{f' ({commit_id})' if commit_id else ''}")
        else:
            entry["status"] = "REJECTED"
            entry["reason"] = "gain not distinguishable from noise" if test["mean_gain"] > 0 else "no improvement"
            print(f"  REJECTED: {entry['reason']}")
        log.append(entry)

        history.append({
            "iteration": i, "candidate_sri": entry["accept_mean"],
            "best_sri": round(float(np.mean(incumbent["accept"])), 5), "status": entry["status"],
            "commit": commit_id, "description": (rationale or "")[:200],
            "holdout_mean": entry["holdout_mean"], "seed42": entry["seed42"],
            "mean_gain": entry["mean_gain"], "lcb": entry["lcb"],
        })
        write_history(history_path, history)
        print("-" * 60)

    print(f"\nDone. Incumbent accept-mean {np.mean(incumbent['accept']):.4f} | "
          f"held-out {np.mean(incumbent['holdout']):.4f} ± {np.std(incumbent['holdout'], ddof=1):.4f} | "
          f"seed42 {incumbent['seed42']:.4f}")
    print(f"Logs: {os.path.relpath(run_dir, HERE)}/experiments.jsonl, history.json")


if __name__ == "__main__":
    main()
