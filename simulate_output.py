#!/usr/bin/env python3
"""
simulate_output.py
Reads parsed_output.txt from /outputs and animates the detected objects
(frame-by-frame) using matplotlib — now with background roads!
"""

import re
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

def load_frames_from_json(file_path: Path):
    """Read frames from a structured JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    # Normalize coordinates (if lat/lon or bounding boxes exist)
    for frame in frames:
        for obj in frame:
            # Some objects might only have lat/lon, others x/y
            if "lon" in obj and "lat" in obj and obj["lon"] is not None and obj["lat"] is not None:
                # Normalize to roughly 0–1
                obj["x"] = (obj["lon"] - min_lon) / (max_lon - min_lon) if (max_lon := 1.0) and (min_lon := 0.0) else 0.5
                obj["y"] = (obj["lat"] - min_lat) / (max_lat - min_lat) if (max_lat := 1.0) and (min_lat := 0.0) else 0.5
            elif "x" not in obj or "y" not in obj:
                obj["x"], obj["y"] = 0.5, 0.5
    return frames

# def parse_output_txt(file_path: Path):
#     """Parse the output text and return list of frames (each a list of objects)."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()

#     # Split by frames
#     frame_blocks = re.split(r"🕒 Frame #[0-9]+ \| Time:", text)
#     frames = []

#     for block in frame_blocks[1:]:
#         objects = []
#         for match in re.finditer(
#             r"🧩 Object ID:\s*(?P<id>\d+).*?"
#             r"Type:\s*(?P<type>[A-Za-z]+).*?"
#             r"Bounding Box:\s*top=\s*(?P<top>[-\d.]+),\s*bottom=\s*(?P<bottom>[-\d.]+),\s*left=\s*(?P<left>[-\d.]+),\s*right=\s*(?P<right>[-\d.]+)",
#             block,
#             re.DOTALL,
#         ):
#             obj_type = match.group("type")
#             left = float(match.group("left"))
#             right = float(match.group("right"))
#             top = float(match.group("top"))
#             bottom = float(match.group("bottom"))

#             x = (left + right) / 2
#             y = (top + bottom) / 2
#             objects.append({"type": obj_type, "x": x, "y": y})

#         frames.append(objects)
#     return frames


# def draw_roads(ax):
#     """Draw two diagonal roads labeled Foothill and Santa Rosa."""
#     # Foothill: top-left → bottom-right
#     ax.plot([-0.6, 1.2], [1.1, -0.3], color="gray", linewidth=15, alpha=0.4, zorder=0)
#     ax.text(0.6, 0.1, "Foothill Blvd", color="black", fontsize=10,
#             rotation=-40, ha="center", va="center", alpha=0.8)

#     # Santa Rosa: bottom-left → top-right
#     ax.plot([-0.6, 1.2], [-0.3, 1.1], color="gray", linewidth=15, alpha=0.4, zorder=0)
#     ax.text(0.4, 0.9, "Santa Rosa St", color="black", fontsize=10,
#             rotation=40, ha="center", va="center", alpha=0.8)

def draw_roads(ax):
    """Draw 3D roads flat at z=0."""
    ax.plot([-0.6, 1.2], [1.1, -0.3], zs=0, color="gray", linewidth=15, alpha=0.4)
    ax.text(0.6, 0.1, 0.01, "Foothill Blvd", color="black", fontsize=10,
            rotation=-40, ha="center", va="center", alpha=0.8)

    ax.plot([-0.6, 1.2], [-0.3, 1.1], zs=0, color="gray", linewidth=15, alpha=0.4)
    ax.text(0.4, 0.9, 0.01, "Santa Rosa St", color="black", fontsize=10,
            rotation=40, ha="center", va="center", alpha=0.8)


# def animate(frames):
#     """Animate frames using matplotlib."""
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.set_xlim(-0.5, 1.0)
#     ax.set_ylim(0.0, 1.0)
#     ax.set_xlabel("X position (normalized)")
#     ax.set_ylabel("Y position (normalized)")
#     ax.set_title("Foothill & Santa Rosa Intersection")
#     scat = ax.scatter([], [])

#     def update(frame_idx):
#         ax.clear()
#         ax.set_xlim(-0.5, 1.0)
#         ax.set_ylim(0.0, 1.0)
#         ax.set_title(f"Frame {frame_idx + 1}/{len(frames)} | Foothill x Santa Rosa")
#         draw_roads(ax)

#         xs, ys, colors = [], [], []
#         for obj in frames[frame_idx]:
#             xs.append(obj["x"])
#             ys.append(obj["y"])
#             t = obj["type"].lower()
#             if t == "car":
#                 colors.append("red")
#             elif t == "person":
#                 colors.append("green")
#             elif t == "truck":
#                 colors.append("orange")
#             elif t == "bus":
#                 colors.append("yellow")
#             else:
#                 colors.append("blue")

#         ax.scatter(xs, ys, c=colors, s=80, edgecolors="black")
#         return scat,

#     anim = FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)
#     plt.show()

def animate_with_controls(frames):
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])  # 3D scene + graph
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_graph = fig.add_subplot(gs[1])
    plt.subplots_adjust(bottom=0.25)

    # --- State ---
    is_paused = False
    current_frame = {"idx": 0}

    # --- Precompute counts ---
    car_counts = [sum(1 for o in f if o.get("type", "").lower() == "car") for f in frames]
    person_counts = [sum(1 for o in f if o.get("type", "").lower() == "person") for f in frames]
    truck_counts = [sum(1 for o in f if o.get("type", "").lower() == "truck") for f in frames]
    frame_numbers = list(range(1, len(frames) + 1))

    # --- Plot initial graph ---
    line_cars, = ax_graph.plot(frame_numbers, car_counts, color="red", label="Cars")
    line_people, = ax_graph.plot(frame_numbers, person_counts, color="green", label="People")
    line_trucks, = ax_graph.plot(frame_numbers, truck_counts, color="orange", label="Trucks")
    marker, = ax_graph.plot([1], [car_counts[0]], "bo", label="Current Frame")

    ax_graph.set_title("Object Counts Over Time")
    ax_graph.set_xlabel("Frame #")
    ax_graph.set_ylabel("Count")
    ax_graph.legend(loc="upper right")
    ax_graph.set_xlim(1, len(frames))
    max_count = max(max(car_counts), max(person_counts), max(truck_counts), 1)
    ax_graph.set_ylim(0, max_count + 2)

    # --- Draw one frame ---
    def draw_frame(idx):
        ax3d.clear()
        ax3d.set_xlim(-0.5, 1.0)
        ax3d.set_ylim(0.0, 1.0)
        ax3d.set_zlim(-0.1, 0.1)
        ax3d.set_title(f"Frame {idx+1}/{len(frames)} | Foothill x Santa Rosa")
        ax3d.set_xlabel("X position (normalized)")
        ax3d.set_ylabel("Y position (normalized)")
        ax3d.set_zlabel("Z (flat plane)")
        draw_roads(ax3d)

        xs, ys, zs, colors = [], [], [], []
        for obj in frames[idx]:
            xs.append(obj.get("x", 0.5))
            ys.append(obj.get("y", 0.5))
            zs.append(0)
            t = obj.get("type", "").lower()
            colors.append({
                "car": "red",
                "person": "green",
                "truck": "orange",
                "bus": "yellow"
            }.get(t, "blue"))

        ax3d.scatter(xs, ys, zs, c=colors, s=80, edgecolors="black")
        ax3d.view_init(elev=85, azim=-90)
        ax3d.grid(True, linestyle="--", alpha=0.3)

        # Move marker on count graph
        marker.set_data([idx + 1], [car_counts[idx]])

    draw_frame(0)

    # --- UI Controls ---
    ax_play = plt.axes([0.35, 0.05, 0.1, 0.06])
    ax_pause = plt.axes([0.47, 0.05, 0.1, 0.06])
    ax_slider = plt.axes([0.1, 0.15, 0.8, 0.03])

    btn_play = Button(ax_play, "▶️ Play")
    btn_pause = Button(ax_pause, "⏸ Pause")
    slider = Slider(ax_slider, "Frame", 1, len(frames), valinit=1, valfmt="%0.0f")

    # --- Event Handlers ---
    def play(_):
        nonlocal is_paused
        is_paused = False

    def pause(_):
        nonlocal is_paused
        is_paused = True

    btn_play.on_clicked(play)
    btn_pause.on_clicked(pause)

    def on_slider_change(val):
        idx = int(slider.val) - 1
        current_frame["idx"] = idx
        draw_frame(idx)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    def update(_):
        if is_paused:
            return
        idx = current_frame["idx"]
        draw_frame(idx)
        current_frame["idx"] = (idx + 1) % len(frames)
        slider.set_val(current_frame["idx"] + 1)

    FuncAnimation(fig, update, interval=100, repeat=True)
    plt.show()

def main():
    out_dir = Path("outputs")
    latest_json = sorted(out_dir.glob("output*.json"))[-1]
    print(f"🎬 Simulating from: {latest_json}")

    frames = load_frames_from_json(latest_json)
    print(f"✅ Loaded {len(frames)} frames from JSON.")
    time.sleep(1)
    animate_with_controls(frames)


if __name__ == "__main__":
    main()