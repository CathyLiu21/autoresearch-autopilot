# Autopilot Policy Optimization Program

## Core Objective
Maximize `METRIC_SRI` output by `train.py`.

## Rules
1. Only modify `train.py`. NEVER modify or inspect `prepare.py`.
2. The function `autopilot_policy(state: dict) -> float` must execute in < 1 millisecond per call.
3. Keep external dependencies limited to `numpy` or standard Python library math.

## Biological Constraints & Hypotheses
- **Deep Sleep:** Requires active thermal cooling during the first half of the night (hours 1–4) to facilitate the core body temperature nadir.
- **REM Sleep:** Thermoregulation is largely suspended during REM. Over-cooling triggers shivering and micro-arousals; modest warmth or neutral offset prevents waking.
- **Ambient Elasticity:** When `room_temp_f` is high (>73°F), beds need steeper negative thermal compensation.
- **Toss/Turn Modulation:** High micro-movements indicate restlessness/discomfort; investigate damping extreme offsets when restlessness is high.

## Ratchet Protocol
For each iteration:
1. Hypothesize a specific mathematical adjustment to `autopilot_policy` in `train.py`.
2. Run `python train.py`.
3. Check the printed `METRIC_SRI`.
4. If `METRIC_SRI` > best previous commit:
   `git commit -am "sri: <score> - <strategy description>"`
5. If `METRIC_SRI` <= best previous commit or throws an exception:
   `git reset --hard HEAD`
6. Repeat.
