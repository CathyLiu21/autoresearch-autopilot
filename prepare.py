# prepare.py — LOCKED BENCHMARK (DO NOT MODIFY)
import numpy as np
from typing import Callable

def run_simulation(policy_fn: Callable[[dict], float], seed: int = 42) -> float:
    """
    Simulates multi-stage sleep nights across a population of 500 users.
    policy_fn takes a state dict:
      - 'room_temp_f': float (ambient bedroom temp)
      - 'sleep_stage': str ('LIGHT', 'DEEP', 'REM')
      - 'baseline_hrv': float (user 30-day RMSSD baseline)
      - 'hour_into_sleep': float (0.0 to 8.0)
      - 'toss_turn_count': int (recent micro-movements)
    policy_fn returns:
      - thermal_offset: float in [-5.0, 5.0] (degrees F delta applied to pod)
    """
    np.random.seed(seed)
    n_nights = 500

    deep_pcts = []
    rem_pcts = []
    hrv_deltas = []
    override_counts = []

    for _ in range(n_nights):
        # Latent user biological and environmental traits
        room_temp = float(np.random.normal(71.0, 3.5))
        baseline_hrv = float(np.random.normal(55.0, 15.0))
        thermal_sensitivity = float(np.random.uniform(0.7, 1.4))
        
        # Ground-truth physiological thermoregulation setpoints:
        # Core body temp drops naturally; needs aggressive cooling during early Deep sleep,
        # moderate cooling during Light, and slight warming during late REM to avoid shivering.
        night_deep_minutes = 0
        night_rem_minutes = 0
        night_overrides = 0
        thermal_comfort_penalties = []

        # Simulate 8-hour sleep broken into 16 half-hour epochs
        for step in range(16):
            hour = step * 0.5
            
            # Synthetic hypnogram stage distribution
            if hour < 1.0:
                stage = "LIGHT"
            elif 1.0 <= hour < 3.5:
                stage = "DEEP" if (step % 3 != 0) else "LIGHT"
            elif 3.5 <= hour < 6.5:
                stage = "REM" if (step % 2 == 0) else "LIGHT"
            else:
                stage = "REM" if (step % 3 == 0) else "LIGHT"

            toss_turns = int(np.random.poisson(lam=1.2 if stage == "LIGHT" else 0.3))

            state = {
                "room_temp_f": room_temp,
                "sleep_stage": stage,
                "baseline_hrv": baseline_hrv,
                "hour_into_sleep": hour,
                "toss_turn_count": toss_turns,
            }

            # Query the agent policy
            action = float(policy_fn(state))
            action = float(np.clip(action, -5.0, 5.0))

            # Ground-truth optimal thermal offset target
            if stage == "DEEP":
                ideal_offset = -0.55 * (room_temp - 67.0) - 1.5
            elif stage == "REM":
                ideal_offset = -0.30 * (room_temp - 68.0) + 0.5
            else:  # LIGHT
                ideal_offset = -0.40 * (room_temp - 68.0)

            thermal_error = abs(action - ideal_offset) * thermal_sensitivity
            thermal_comfort_penalties.append(thermal_error)

            # Stage duration impacts
            if stage == "DEEP":
                night_deep_minutes += max(0.0, 30.0 - (thermal_error * 4.5))
            elif stage == "REM":
                night_rem_minutes += max(0.0, 30.0 - (thermal_error * 3.5))

            # Severe thermal mismatch triggers user manual override
            if thermal_error > 3.8:
                night_overrides += 1

        total_sleep_minutes = 480.0
        deep_pct = night_deep_minutes / total_sleep_minutes
        rem_pct = night_rem_minutes / total_sleep_minutes
        
        # Overnight HRV recovers when thermal strain is minimized
        avg_strain = np.mean(thermal_comfort_penalties)
        hrv_delta = float(7.0 - (avg_strain * 2.2) + np.random.normal(0, 1.2))

        deep_pcts.append(deep_pct)
        rem_pcts.append(rem_pct)
        hrv_deltas.append(hrv_delta)
        override_counts.append(night_overrides)

    # Invariant composite score calculation
    sri = (
        (np.mean(deep_pcts) * 100.0)
        + (np.mean(rem_pcts) * 50.0)
        + (np.mean(hrv_deltas) * 1.5)
        - (np.mean(override_counts) * 40.0)
    )

    print(f"METRIC_SRI: {sri:.5f}")
    return sri

if __name__ == "__main__":
    # Test with dummy zero-policy
    run_simulation(lambda state: 0.0)
