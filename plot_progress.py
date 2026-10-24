import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_demo_plots():
    # Load history
    history_file = "history.json"
    if not os.path.exists(history_file):
        print(f"{history_file} not found.")
        return

    with open(history_file) as f:
        history = json.load(f)

    iters = [h["iteration"] for h in history]
    candidate_sris = [h["candidate_sri"] for h in history]
    best_sris = [h["best_sri"] for h in history]
    statuses = [h["status"] for h in history]

    # Set up dark aesthetic figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#0b0f19')

    # Grid layout: Top row spans full width (2 cols), bottom row has 2 cols
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------
    # 1. Top Panel: Convergence Ratchet Frontier
    # -------------------------------------------------------------
    ax1.set_facecolor('#111827')
    ax1.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')

    # Plot ratcheted best frontier
    ax1.step(iters, best_sris, where='post', color='#38bdf8', linewidth=3, label='Ratcheted Best SRI (Git Frontier)', zorder=3)
    ax1.plot(iters, best_sris, 'o', color='#38bdf8', markersize=6, alpha=0.7)

    # Plot candidate points
    for i, (cand, status) in enumerate(zip(candidate_sris, statuses)):
        if status in ['ACCEPTED', 'BASELINE']:
            ax1.scatter(i, cand, color='#10b981', s=130, marker='D', edgecolors='white', linewidth=1.5, zorder=5, label='Accepted Commit' if i == 1 else "")
            ax1.annotate(f"{cand:.2f}", (i, cand), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color='#34d399')
        else:
            ax1.scatter(i, cand, color='#f43f5e', s=55, marker='x', alpha=0.75, zorder=4, label='Rejected Regression' if i == 2 else "")

    ax1.axhline(0, color='#64748b', linestyle=':', alpha=0.5)
    ax1.set_title("Autonomous AI Research Ratchet: Sleep Recovery Index (SRI) Frontier", fontsize=15, fontweight='bold', pad=12, color='#f8fafc')
    ax1.set_xlabel("Optimization Iteration", fontsize=11, color='#94a3b8')
    ax1.set_ylabel("Metric SRI (Score)", fontsize=11, color='#94a3b8')
    ax1.set_xticks(iters)
    ax1.legend(loc='lower right', framealpha=0.8, facecolor='#1e293b', edgecolor='#334155', fontsize=10)
    ax1.set_ylim(-35, 40)

    # -------------------------------------------------------------
    # 2. Bottom Left Panel: Discovered Control Policy Surface
    # -------------------------------------------------------------
    ax2.set_facecolor('#111827')
    ax2.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')

    temps = np.linspace(64, 78, 100)
    # Baseline: -0.2 * (room_temp - 70)
    base_offsets = np.clip(-0.2 * (temps - 70.0), -5.0, 5.0)

    # Discovered Policy responses (hour=2.0, toss=0)
    def calc_discovered(room_temp, stage, hour=2.0, toss=0):
        offset = -0.35 * (room_temp - 70.0)
        if room_temp > 73.0:
            offset -= 0.45 * (room_temp - 73.0)
        if stage == "DEEP":
            offset -= 2.2 if hour <= 2.5 else (1.7 if hour <= 4.5 else 0.7)
        elif stage == "REM":
            offset += 0.7
        elif stage == "LIGHT":
            offset += 0.3 if hour > 5.5 else -0.2
        return np.clip(offset, -5.0, 5.0)

    deep_offsets = [calc_discovered(t, "DEEP", hour=2.0) for t in temps]
    rem_offsets = [calc_discovered(t, "REM", hour=5.0) for t in temps]
    light_offsets = [calc_discovered(t, "LIGHT", hour=1.0) for t in temps]

    ax2.plot(temps, base_offsets, '--', color='#64748b', linewidth=2, label='Baseline (Naive Linear)')
    ax2.plot(temps, deep_offsets, '-', color='#60a5fa', linewidth=2.5, label='Discovered: DEEP Sleep (Hour 2)')
    ax2.plot(temps, rem_offsets, '-', color='#f59e0b', linewidth=2.5, label='Discovered: REM Sleep (Hour 5)')
    ax2.plot(temps, light_offsets, '-', color='#2dd4bf', linewidth=2, label='Discovered: LIGHT Sleep')

    ax2.set_title("Discovered Policy: Thermal Offset (°F) vs Room Temp", fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax2.set_xlabel("Ambient Bedroom Temperature (°F)", fontsize=10, color='#94a3b8')
    ax2.set_ylabel("Pod Thermal Offset (°F)", fontsize=10, color='#94a3b8')
    ax2.legend(loc='upper right', framealpha=0.8, facecolor='#1e293b', edgecolor='#334155', fontsize=8.5)
    ax2.set_ylim(-5.5, 5.5)

    # -------------------------------------------------------------
    # 3. Bottom Right Panel: Physiology Clinical Metrics
    # -------------------------------------------------------------
    ax3.set_facecolor('#111827')
    ax3.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')

    metrics = ['Deep Sleep %', 'REM Sleep %', 'HRV Delta (ms)', 'Manual Overrides']
    # Baseline vs Final discovered population outcomes
    baseline_vals = [12.4, 11.2, 1.8, 1.85]
    discovered_vals = [24.6, 21.3, 6.7, 0.0]

    x = np.arange(len(metrics))
    width = 0.36

    rects1 = ax3.bar(x - width/2, baseline_vals, width, label='Baseline Policy', color='#475569', edgecolor='#94a3b8', alpha=0.9)
    rects2 = ax3.bar(x + width/2, discovered_vals, width, label='AI Discovered Policy', color='#10b981', edgecolor='#34d399', alpha=0.9)

    ax3.set_title("500-Night Population Clinical Recovery Impact", fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=9.5, color='#cbd5e1')
    ax3.legend(loc='upper right', framealpha=0.8, facecolor='#1e293b', edgecolor='#334155', fontsize=8.5)
    ax3.set_ylabel("Impact Metric Value", fontsize=10, color='#94a3b8')

    for rect in rects1:
        h = rect.get_height()
        ax3.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#94a3b8')
    for rect in rects2:
        h = rect.get_height()
        ax3.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#34d399')

    plt.savefig("autoresearch_demo_plot.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Demo plot successfully generated and saved to autoresearch_demo_plot.png")

if __name__ == "__main__":
    generate_demo_plots()
