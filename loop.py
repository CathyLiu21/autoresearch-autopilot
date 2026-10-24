# loop.py — AUTONOMOUS ORCHESTRATOR (VERTEX AI POWERED)
import json
import os
import re
import subprocess
import sys
from google import genai

# Import plot generator if available
try:
    from plot_progress import generate_demo_plots
except ImportError:
    generate_demo_plots = lambda: None

# Configure Google Cloud Credentials for Vertex AI (DeepMind Credit Tier 2)
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    for path in ["service_account.json", "../service_account.json"]:
        if os.path.exists(path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(path)
            break

# Initialize Vertex AI Client
PROJECT_ID = "gemini-build-day-481122"
LOCATION = "global"
MODEL_NAME = "gemini-3.8-flash"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def run_cmd(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def get_current_metric() -> float:
    output = run_cmd(f"{sys.executable} train.py")
    match = re.search(r"METRIC_SRI:\s*(-?[0-9.]+)", output)
    if not match:
        raise ValueError(f"Failed to parse METRIC_SRI. Output:\n{output}")
    return float(match.group(1))

def record_history(iteration, candidate, best, status, commit=None, desc=""):
    history = []
    if os.path.exists("history.json"):
        try:
            with open("history.json") as f:
                history = json.load(f)
        except Exception:
            history = []
    
    history.append({
        "iteration": iteration,
        "candidate_sri": round(candidate, 5),
        "best_sri": round(best, 5),
        "status": status,
        "commit": commit,
        "description": desc
    })

    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Automatically refresh plot
    try:
        generate_demo_plots()
    except Exception as e:
        print(f"Note: Could not refresh plot: {e}")

def main(iterations: int = 10):
    best_score = get_current_metric()
    print(f"Starting baseline METRIC_SRI: {best_score:.5f}\n")

    for i in range(1, iterations + 1):
        print(f"=== Iteration {i}/{iterations} ===")
        
        train_code = open("train.py").read()
        program_doc = open("program.md").read()

        prompt = f"""
You are an expert ML Research Engineer optimizing an IoT thermal control policy.
Here are your instructions:
{program_doc}

Current best METRIC_SRI: {best_score:.5f}

Here is the current train.py:
```python
{train_code}
```

Propose a single concrete mathematical change to autopilot_policy in train.py to increase METRIC_SRI.
Respond ONLY with the complete, valid Python code for train.py wrapped in ```python ... ``` fences.
Do not provide explanations outside the code block.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        # Extract generated code
        code_match = re.search(r"```python\s*\n(.*?)\n```", response.text, re.DOTALL)
        if not code_match:
            print("Failed to extract code block from LLM response. Skipping.")
            continue

        new_code = code_match.group(1)
        with open("train.py", "w") as f:
            f.write(new_code)

        # Run validation
        eval_output = run_cmd(f"{sys.executable} train.py")
        metric_match = re.search(r"METRIC_SRI:\s*(-?[0-9.]+)", eval_output)

        if metric_match:
            candidate_score = float(metric_match.group(1))
            print(f"Candidate METRIC_SRI: {candidate_score:.5f}")

            if candidate_score > best_score:
                gain = candidate_score - best_score
                best_score = candidate_score
                print(f" Improvement detected (+{gain:.5f})! Committing.")
                commit_out = run_cmd(f'git commit -am "sri: {best_score:.5f} (iter {i})"')
                commit_hash = re.search(r"\[master ([0-9a-f]+)\]", commit_out)
                c_id = commit_hash.group(1) if commit_hash else "new"
                record_history(i, candidate_score, best_score, "ACCEPTED", commit=c_id, desc=f"Improvement (+{gain:.3f})")
            else:
                print(f" Regression/Stagnation ({candidate_score:.5f} <= {best_score:.5f}). Resetting.")
                run_cmd("git reset --hard HEAD")
                record_history(i, candidate_score, best_score, "REJECTED", desc="Stagnation/Regression")
        else:
            print(f" Execution error or crash. Resetting.\n{eval_output}")
            run_cmd("git reset --hard HEAD")
            record_history(i, 0.0, best_score, "REJECTED", desc="Crash/Syntax error")

        print("-" * 50)

    print(f"\nOptimization loop finished. Best METRIC_SRI achieved: {best_score:.5f}")

if __name__ == "__main__":
    main(iterations=10)

