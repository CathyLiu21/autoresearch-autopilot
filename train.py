# train.py — AGENT PLAYGROUND
from prepare import run_simulation


def autopilot_policy(state: dict) -> float:
    """State keys:

    - room_temp_f: float
    - sleep_stage: 'LIGHT' | 'DEEP' | 'REM'
    - baseline_hrv: float
    - hour_into_sleep: float
    - toss_turn_count: int
    """
    room_temp = state["room_temp_f"]
    stage = state["sleep_stage"]
    hour = state["hour_into_sleep"]
    toss = state["toss_turn_count"]

    # 1. Ambient Elasticity: steeper negative compensation when hot
    ambient_diff = room_temp - 70.0
    if room_temp > 73.0:
        offset = -0.6 * ambient_diff
    else:
        offset = -0.3 * ambient_diff

    # 2. Sleep stage & circadian modulation
    if stage == "DEEP":
        # Active thermal cooling in first half of night (hours 1-4) for body temperature nadir
        if hour <= 4.0:
            offset -= 1.8
        else:
            offset -= 0.8
    elif stage == "REM":
        # Thermoregulation suspended: counteract over-cooling with modest warmth
        offset += 0.8
    elif stage == "LIGHT":
        offset -= 0.2

    # 3. Restlessness / Toss & Turn Modulation: damp offsets when discomfort is high
    if toss > 0:
        damping = 1.0 / (1.0 + 0.25 * toss)
        offset *= damping

    # Clamp to hardware limits [-5.0, 5.0] F
    return max(-5.0, min(5.0, offset))


if __name__ == "__main__":
    run_simulation(autopilot_policy)