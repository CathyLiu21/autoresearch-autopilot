# Autopilot Policy Optimization Program (v2)

## Core Objective
Maximize the Sleep Recovery Index (SRI) of `autopilot_policy` in `train.py`.
You no longer run anything yourself: a trusted evaluator imports your policy,
scores it in a separate process across many simulation seeds, and decides.

## Division of Labor: You Pick the Structure, an Optimizer Picks the Numbers
- Your job is the **functional form**: which state features matter, which
  branches exist, how terms interact (slopes, offsets, thresholds, ramps).
- Every tunable number must live in a module-level `PARAMS` dict. After you
  propose a structure, `scipy.optimize` (Powell) fits all `PARAMS` values on
  separate fitting seeds. **Do not spend effort hand-tuning constants**; give
  sensible initial guesses and the optimizer will refine them.
- Two proposals that differ only in `PARAMS` values are the **same structure**
  and will be skipped as duplicates.

## Code Contract (proposals violating it are rejected)
1. `PARAMS` is a top-level dict literal mapping `str` -> `float`, e.g.
   `PARAMS = {"deep_offset": -2.0, "ambient_slope": -0.35}`.
   Keep it to **at most 12 parameters** (the optimizer budget is limited).
2. Signature: `def autopilot_policy(state: dict, params: dict = PARAMS) -> float`.
   Read every tunable number from `params[...]`, not from literals.
3. Return a finite float; it is clipped to [-5.0, 5.0].
4. Pure and deterministic: no global RNG use, no I/O, no imports beyond
   `math`/`numpy`. Must run in < 1 ms per call.
5. Only `train.py` may change. Never read, import-patch, or modify
   `prepare.py`, the evaluator, or numpy. Tampering is detected and rejected.
6. Keep `from prepare import run_simulation` and the
   `if __name__ == "__main__": run_simulation(autopilot_policy)` footer.

## State Available to the Policy
- `room_temp_f`: float, ambient bedroom temperature (°F)
- `sleep_stage`: `"LIGHT" | "DEEP" | "REM"`
- `baseline_hrv`: float, user's 30-day RMSSD baseline
- `hour_into_sleep`: float, 0.0–8.0
- `toss_turn_count`: int, recent micro-movements

## Biological Hypotheses (to test, not facts)
- **Deep sleep:** active cooling early in the night may aid the core-temperature nadir.
- **REM sleep:** thermoregulation is blunted; over-cooling may cause arousals.
- **Ambient elasticity:** hotter rooms may need steeper negative compensation,
  and the right slope may differ by stage.
- **Toss/turn:** restlessness may or may not indicate thermal discomfort.
  Check the experiment log before re-trying ideas that already failed.

## Acceptance Rule (for your information)
A candidate is accepted only if, paired seed-by-seed against the incumbent on
20 acceptance seeds, the 95% lower confidence bound of the mean SRI gain is
above a small minimum effect. Tiny or noisy "improvements" are rejected.

## Response Format
Reply with exactly two parts:

RATIONALE: <2–4 sentences: the structural hypothesis, why it should help,
and how it differs from every attempt in the experiment log>

```python
<complete train.py>
```
