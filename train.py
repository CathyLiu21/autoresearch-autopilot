# train.py — AGENT PLAYGROUND
from prepare import run_simulation

PARAMS = {  # fitted by scipy.optimize (Powell) in loop_v2.py
    'ambient_ref': 67.72621,
    'ambient_slope': -0.403188,
    'deep_slope_adj': -0.143754,
    'rem_slope_adj': 0.105432,
    'deep_offset': -1.787153,
    'rem_offset': 0.727916,
    'light_offset': 0.232037,
    'hour_linear': -0.105029,
    'hour_quad': 0.012239,
    'hrv_ref': -42.033718,
    'hrv_slope': 0.000424,
}


def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    """Calculates thermal offset using stage-specific ambient elasticity,
    a quadratic circadian trajectory (nadir dip and wake warming),
    and baseline HRV personalization.
    """
    room_temp = state["room_temp_f"]
    stage = state["sleep_stage"]
    hour = state["hour_into_sleep"]
    hrv = state["baseline_hrv"]

    # 1. Stage-dependent ambient elasticity across full room temperature range
    slope = params["ambient_slope"]
    if stage == "DEEP":
        slope += params["deep_slope_adj"]
    elif stage == "REM":
        slope += params["rem_slope_adj"]
    ambient_comp = slope * (room_temp - params["ambient_ref"])

    # 2. Stage-specific base thermal setpoint
    if stage == "DEEP":
        stage_offset = params["deep_offset"]
    elif stage == "REM":
        stage_offset = params["rem_offset"]
    else:  # LIGHT
        stage_offset = params["light_offset"]

    # 3. Non-linear circadian trajectory (parabolic nadir curve)
    circadian_offset = (
        params["hour_linear"] * hour + params["hour_quad"] * (hour**2)
    )

    # 4. Baseline HRV personalization
    hrv_offset = params["hrv_slope"] * (hrv - params["hrv_ref"])

    total_offset = ambient_comp + stage_offset + circadian_offset + hrv_offset

    # Clamp to hardware limits [-5.0, 5.0] F
    return max(-5.0, min(5.0, float(total_offset)))


if __name__ == "__main__":
    run_simulation(autopilot_policy)