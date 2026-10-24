# train.py — AGENT PLAYGROUND
from prepare import run_simulation

PARAMS = {  # fitted by scipy.optimize (Powell) in loop_v2.py
    'ambient_ref': 67.384071,
    'ambient_slope': -0.406628,
    'hot_thresh': 103.042802,
    'hot_slope': -0.4,
    'deep_hot_extra': -0.2,
    'deep_offset': -2.151491,
    'rem_offset': 1.057594,
    'light_offset': 0.256966,
    'hour_slope': 0.001782,
    'toss_damp': 0.009169,
}


def autopilot_policy(state: dict, params: dict = PARAMS) -> float:
    """Calculates thermal offset based on continuous ambient elasticity, stage-

    dependent hot-room compensation, and continuous circadian progression.
    """
    room_temp = state["room_temp_f"]
    stage = state["sleep_stage"]
    hour = state["hour_into_sleep"]
    toss = state["toss_turn_count"]

    # 1. Base ambient compensation
    ambient_comp = params["ambient_slope"] * (room_temp - params["ambient_ref"])

    # 2. Hot-room elasticity with stage-differentiated slope
    if room_temp > params["hot_thresh"]:
        excess_heat = room_temp - params["hot_thresh"]
        effective_hot_slope = params["hot_slope"]
        if stage == "DEEP":
            effective_hot_slope += params["deep_hot_extra"]
        ambient_comp += effective_hot_slope * excess_heat

    # 3. Stage-specific base offsets
    if stage == "DEEP":
        stage_offset = params["deep_offset"]
    elif stage == "REM":
        stage_offset = params["rem_offset"]
    else:  # LIGHT
        stage_offset = params["light_offset"]

    # 4. Continuous circadian ramp over sleep duration
    circadian_offset = params["hour_slope"] * hour

    total_offset = ambient_comp + stage_offset + circadian_offset

    # 5. Restlessness damping
    if toss > 0:
        total_offset = total_offset / (1.0 + params["toss_damp"] * toss)

    # Clamp to hardware limits [-5.0, 5.0] F
    return max(-5.0, min(5.0, total_offset))


if __name__ == "__main__":
    run_simulation(autopilot_policy)