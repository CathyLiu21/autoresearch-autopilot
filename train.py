# train.py — AGENT PLAYGROUND
from prepare import run_simulation

PARAMS = {  # fitted by scipy.optimize (Powell) in loop_v2.py
    'ambient_ref': 68.074612,
    'ambient_slope': -0.406405,
    'deep_slope_adj': -0.143311,
    'rem_slope_adj': 0.106406,
    'deep_offset': -2.070476,
    'deep_hour_slope': 0.073124,
    'rem_offset': 0.705788,
    'light_offset': 0.183407,
    'hour_linear': -0.106673,
    'hour_quad': 0.011499,
    'rem_min_offset': -6.96538,
}


def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    """Calculates thermal offset using stage-specific ambient elasticity,
    a quadratic circadian trajectory, early-night deep sleep nadir assistance,
    and a REM over-cooling floor to protect blunted thermoregulation.
    """
    room_temp = state["room_temp_f"]
    stage = state["sleep_stage"]
    hour = state["hour_into_sleep"]

    # 1. Stage-dependent ambient elasticity across room temperature range
    slope = params["ambient_slope"]
    if stage == "DEEP":
        slope += params["deep_slope_adj"]
    elif stage == "REM":
        slope += params["rem_slope_adj"]
    ambient_comp = slope * (room_temp - params["ambient_ref"])

    # 2. Stage-specific base thermal setpoint with early deep sleep nadir assistance
    if stage == "DEEP":
        stage_offset = params["deep_offset"] + params["deep_hour_slope"] * hour
    elif stage == "REM":
        stage_offset = params["rem_offset"]
    else:  # LIGHT
        stage_offset = params["light_offset"]

    # 3. Non-linear circadian trajectory (parabolic nadir curve)
    circadian_offset = (
        params["hour_linear"] * hour + params["hour_quad"] * (hour**2)
    )

    total_offset = ambient_comp + stage_offset + circadian_offset

    # 4. REM arousal protection: guard against over-cooling when thermoregulation is blunted
    if stage == "REM":
        total_offset = max(params["rem_min_offset"], total_offset)

    # Clamp to hardware limits [-5.0, 5.0] F
    return max(-5.0, min(5.0, float(total_offset)))


if __name__ == "__main__":
    run_simulation(autopilot_policy)