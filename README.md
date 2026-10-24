# Autonomous IoT Thermal Autopilot Optimization

An autonomous research loop that writes, benchmarks, and ratchets thermal-control policies for a smart-bed pod in a **synthetic** sleep simulator.

An LLM (**`gemini-3.8-flash` on Google Cloud Vertex AI**) edits the policy code (`train.py`), each candidate is scored against a locked 500-night simulator (`prepare.py`), and a **Git ratchet** keeps only changes that raise the score. No human is in the inner loop.

> [!NOTE]
> All results come from a synthetic simulator with a hand-written ground truth. They are not clinical measurements.

---

## Key Results

Measured on the evaluation seed (42) unless noted. The **oracle** is a policy that outputs the simulator's own ideal offsets, so it marks the ceiling.

| Metric | Baseline | AI Discovered | Oracle Ceiling |
| :--- | :---: | :---: | :---: |
| **`METRIC_SRI` (seed 42)** | -29.49 | **33.06** | 39.56 |
| **`METRIC_SRI` (held-out seeds 0–19, mean ± sd)** | -24.35 ± 2.62 | **33.54 ± 0.32** | 39.94 ± 0.34 |
| Deep sleep % | 8.36 | **15.19** | 17.96 |
| REM sleep % | 24.02 | 22.79 | 25.00 |
| HRV delta | 3.88 | **5.12** | 6.87 |
| Manual overrides / night | 1.39 | **0.03** | 0.03 |

- The agent closed **~91% of the gap** between baseline and oracle in 10 iterations (2 accepted, 8 rejected).
- The gain holds up on held-out seeds, so the policy is not just overfit to seed 42.
- REM % went slightly *down*. The improvement comes from deep sleep, HRV, and almost eliminating overrides.

---

## Methodology

Inspired by Karpathy's *autoresearch* idea: a locked benchmark, an agent-editable file, and a Git commit/reset ratchet.

```
[ Research directives (program.md) ]
                │
                ▼
[ gemini-3.8-flash on Vertex AI ] ──> proposes new train.py
                │
                ▼
[ Locked simulator (prepare.py) ] ──> METRIC_SRI over 500 nights
                │
                ├── SRI > best  : git commit        (keep)
                └── SRI <= best : git reset --hard  (discard)
```

### Metric: Sleep Recovery Index (SRI)
Each run covers 500 simulated nights × 16 half-hour epochs (8,000 decisions). Policies are ranked by:

$$
\text{SRI} = (\text{Deep Sleep \%} \times 100) + (\text{REM \%} \times 50) + (\Delta\text{HRV} \times 1.5) - (\text{Overrides} \times 40)
$$

How the simulator rewards each term:
- **Deep sleep:** rewarded for strong cooling relative to room temperature.
- **REM:** rewarded for milder cooling or slight warming.
- **HRV delta:** falls as average thermal error rises.
- **Overrides:** any epoch with thermal error above 3.8°F costs 40 points.

---

## Discovered Policy (commit `dada974`)

```python
def autopilot_policy(state: dict) -> float:
    room_temp = state["room_temp_f"]
    stage = state["sleep_stage"]
    hour = state["hour_into_sleep"]
    toss = state["toss_turn_count"]

    # Ambient compensation, steeper above 73°F
    offset = -0.35 * (room_temp - 70.0)
    if room_temp > 73.0:
        offset -= 0.45 * (room_temp - 73.0)

    # Stage / time-of-night offsets
    if stage == "DEEP":
        offset -= 2.2 if hour <= 2.5 else (1.7 if hour <= 4.5 else 0.7)
    elif stage == "REM":
        offset += 0.7
    elif stage == "LIGHT":
        offset += 0.3 if hour > 5.5 else -0.2

    # Toss/turn damping (introduced in iteration 1)
    if toss > 0:
        offset *= 1.0 / (1.0 + 0.25 * toss)

    return max(-5.0, min(5.0, offset))
```

**Ablation:** removing the toss/turn damping *raises* SRI from 33.06 to 33.56 (seed 42) and from 33.54 to 34.07 (held-out). In this simulator toss/turns don't affect the ideal offset, so the damping only adds error. The ratchet accepted it because it arrived bundled with other, larger improvements in iteration 1.

---

## Convergence

![Autoresearch Demo Plot](autoresearch_demo_plot.png)

```text
* dada974  sri: 33.06498 (iter 9)
* 586e1fc  sri: 32.44663 (iter 1)
* f45695b  chore: configure loop.py for Vertex AI
* 8e1dd8f  baseline setup: static offset policy
```

- **Iteration 1 (32.45):** added stage-specific offsets, a piecewise ambient slope, and toss/turn damping.
- **Iterations 2–8:** rejected. Proposals weren't logged in this run. Iterations 3 & 5 and 7 & 8 scored exactly the same, suggesting repeated proposals.
- **Iteration 9 (33.06):** made the ambient slope continuous, used 3-tier deep-sleep cooling, and adjusted REM and late-night light offsets.

---

## Limitations & Next Steps

1. **Single evaluation seed.** Acceptance uses only seed 42. Small gains (e.g. +0.62 in iteration 9) should be checked across several seeds before accepting.
2. **Score can be spoofed.** `loop.py` reads `METRIC_SRI` from whatever `train.py` prints, and generated code runs unsandboxed. Scoring should happen in a separate process the agent can't control.
3. **No memory of failed attempts.** The LLM never sees past rejected proposals, so it repeats them.
4. **Bundled changes aren't ablated.** Iteration 1 bundled a harmful change (toss damping) with helpful ones.
5. **LLMs are poor at tuning constants.** A hybrid approach (LLM proposes the structure, `scipy.optimize` fits the coefficients) should get close to the oracle.

---

## Project Structure

```
autoresearch-autopilot/
├── prepare.py                 # LOCKED: synthetic 500-night simulator
├── train.py                   # Agent-editable policy
├── loop.py                    # Vertex AI ratchet loop
├── program.md                 # Research directives given to the LLM
├── plot_progress.py           # Generates autoresearch_demo_plot.png
├── dashboard.html             # Interactive dashboard + policy simulator
├── history.json               # Per-iteration scores and commits
└── autoresearch_demo_plot.png
```

---

## Quickstart

```bash
git clone https://github.com/CathyLiu21/autoresearch-autopilot.git && cd autoresearch-autopilot
uv sync
# Put a Google Cloud service-account key at service_account.json (gitignored)
uv run python loop.py
uv run python plot_progress.py   # regenerate the plot
open dashboard.html              # interactive dashboard
```

---

## References

- Kräuchi, K., Cajochen, C., Werth, E., & Wirz-Justice, A. (1999). *Warm feet promote the rapid onset of sleep.* **Nature**, 401, 36–37.
- Okamoto-Mizuno, K., & Mizuno, K. (2012). *Effects of thermal environment on sleep and circadian rhythm.* **Journal of Physiological Anthropology**, 31, 14.
- Shaffer, F., & Ginsberg, J. P. (2017). *An overview of heart rate variability metrics and norms.* **Frontiers in Public Health**, 5, 258.
- Ma, Y. J., et al. (2023). *Eureka: Human-level reward design via coding large language models.* arXiv:2310.12931.
- Romera-Paredes, B., et al. (2024). *Mathematical discoveries from program search with large language models* (FunSearch). **Nature**, 625, 468–475.
- Karpathy, A. *autoresearch* — github.com/karpathy/autoresearch.
