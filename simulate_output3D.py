#!/usr/bin/env python3
"""
simulate_output_3d_controls.py
3D visualization with working Play/Pause and Frame Slider for interactive control.
"""

import re
import time
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D


# -------------------------------------------------------
# Parse output frames
# -------------------------------------------------------
def parse_output_txt(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    frame_blocks = re.split(r"🕒 Frame #[0-9]+ \| Time:", text)
    frames = []

    for block in frame_blocks[1:]:
        objects = []
        for match in re.finditer(
            r"🧩 Object ID:\s*(?P<id>\d+).*?"
            r"Type:\s*(?P<type>[A-Za-z]+).*?"
            r"Bounding Box:\s*top=\s*(?P<top>[-\d.]+),\s*bottom=\s*(?P<bottom>[-\d.]+),\s*left=\s*(?P<left>[-\d.]+),\s*right=\s*(?P<right>[-\d.]+)",
            block,
            re.DOTALL,
        ):
            t = match.group("type")
            left, right = float(match.group("left")), float(match.group("right"))
            top, bottom = float(match.group("top")), float(match.group("bottom"))
            x = (left + right) / 2
            y = (top + bottom) / 2
            objects.append({"type": t, "x": x, "y": y})
        frames.append(objects)

    return frames


# -------------------------------------------------------
# Draw roads
# -------------------------------------------------------
def draw_roads(ax):
    """Draw 3D roads flat at z=0"""
    ax.plot([-0.6, 1.2], [1.1, -0.3], zs=0, color="gray", linewidth=15, alpha=0.4)
    ax.text(0.6, 0.1, 0.01, "Foothill Blvd", color="black", fontsize=10,
            rotation=-40, ha="center", va="center", alpha=0.8)

    ax.plot([-0.6, 1.2], [-0.3, 1.1], zs=0, color="gray", linewidth=15, alpha=0.4)
    ax.text(0.4, 0.9, 0.01, "Santa Rosa St", color="black", fontsize=10,
            rotation=40, ha="center", va="center", alpha=0.8)


# -------------------------------------------------------
# Main animation with controls
# -------------------------------------------------------
def animate_with_controls(frames):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.25)  # space for buttons + slider

    # State
    is_paused = False
    current_frame = {"idx": 0}

    # Plot initial frame
    def draw_frame(idx):
        ax.clear()
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(-0.1, 0.1)
        ax.set_title(f"Frame {idx+1}/{len(frames)} | Foothill x Santa Rosa")
        ax.set_xlabel("X position (normalized)")
        ax.set_ylabel("Y position (normalized)")
        ax.set_zlabel("Z (flat plane)")
        draw_roads(ax)

        xs, ys, zs, colors = [], [], [], []
        for obj in frames[idx]:
            xs.append(obj["x"])
            ys.append(obj["y"])
            zs.append(0)
            t = obj["type"].lower()
            colors.append({
                "car": "red",
                "person": "green",
                "truck": "orange",
                "bus": "yellow"
            }.get(t, "blue"))

        ax.scatter(xs, ys, zs, c=colors, s=80, edgecolors="black")
        ax.view_init(elev=85, azim=-90)
        ax.grid(True, linestyle="--", alpha=0.3)

    draw_frame(0)

    # ---------------- Buttons + Slider ----------------
    ax_play = plt.axes([0.35, 0.05, 0.1, 0.06])
    ax_pause = plt.axes([0.47, 0.05, 0.1, 0.06])
    ax_slider = plt.axes([0.1, 0.15, 0.8, 0.03])

    btn_play = Button(ax_play, "▶️ Play")
    btn_pause = Button(ax_pause, "⏸ Pause")
    slider = Slider(ax_slider, "Frame", 1, len(frames), valinit=1, valfmt="%0.0f")

    # --- Button handlers ---
    def play(event):
        nonlocal is_paused
        is_paused = False

    def pause(event):
        nonlocal is_paused
        is_paused = True

    btn_play.on_clicked(play)
    btn_pause.on_clicked(pause)

    # --- Slider handler ---
    def on_slider_change(val):
        idx = int(slider.val) - 1
        current_frame["idx"] = idx
        draw_frame(idx)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    # --- Animation update ---
    def update(_):
        if is_paused:
            return
        idx = current_frame["idx"]
        draw_frame(idx)
        current_frame["idx"] = (idx + 1) % len(frames)
        slider.set_val(current_frame["idx"] + 1)

    anim = FuncAnimation(fig, update, interval=100, repeat=True)
    plt.show()


# -------------------------------------------------------
# Run
# -------------------------------------------------------
def main():
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.txt"))[-1]
    print(f"🎬 Simulating from: {latest}")

    frames = parse_output_txt(latest)
    print(f"✅ Parsed {len(frames)} frames.")
    time.sleep(1)
    animate_with_controls(frames)


if __name__ == "__main__":
    main()
