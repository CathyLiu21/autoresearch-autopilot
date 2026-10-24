# train.py — AGENT PLAYGROUND
from prepare import run_simulation

def autopilot_policy(state: dict) -> float:
    """
    State keys:
      - room_temp_f: float
      - sleep_stage: 'LIGHT' | 'DEEP' | 'REM'
      - baseline_hrv: float
      - hour_into_sleep: float
      - toss_turn_count: int
    """
    room_temp = state["room_temp_f"]
    
    # Baseline: Naive static linear ambient compensation
    offset = -0.2 * (room_temp - 70.0)
    
    # Clamp to hardware limits [-5.0, 5.0] F
    return max(-5.0, min(5.0, offset))

if __name__ == "__main__":
    run_simulation(autopilot_policy)
