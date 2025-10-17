#!/usr/bin/env python3
"""
simulate_output.py
Reads parsed_output.txt from /outputs and animates the detected objects
(frame-by-frame) using matplotlib — now with background roads!
"""

import re
import time
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def parse_output_txt(file_path: Path):
    """Parse the output text and return list of frames (each a list of objects)."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by frames
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
            obj_type = match.group("type")
            left = float(match.group("left"))
            right = float(match.group("right"))
            top = float(match.group("top"))
            bottom = float(match.group("bottom"))

            x = (left + right) / 2
            y = (top + bottom) / 2
            objects.append({"type": obj_type, "x": x, "y": y})

        frames.append(objects)
    return frames


def draw_roads(ax):
    """Draw two diagonal roads labeled Foothill and Santa Rosa."""
    # Foothill: top-left → bottom-right
    ax.plot([-0.6, 1.2], [1.1, -0.3], color="gray", linewidth=15, alpha=0.4, zorder=0)
    ax.text(0.6, 0.1, "Foothill Blvd", color="black", fontsize=10,
            rotation=-40, ha="center", va="center", alpha=0.8)

    # Santa Rosa: bottom-left → top-right
    ax.plot([-0.6, 1.2], [-0.3, 1.1], color="gray", linewidth=15, alpha=0.4, zorder=0)
    ax.text(0.4, 0.9, "Santa Rosa St", color="black", fontsize=10,
            rotation=40, ha="center", va="center", alpha=0.8)


def animate(frames):
    """Animate frames using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("X position (normalized)")
    ax.set_ylabel("Y position (normalized)")
    ax.set_title("Foothill & Santa Rosa Intersection")
    scat = ax.scatter([], [])

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Frame {frame_idx + 1}/{len(frames)} | Foothill x Santa Rosa")
        draw_roads(ax)

        xs, ys, colors = [], [], []
        for obj in frames[frame_idx]:
            xs.append(obj["x"])
            ys.append(obj["y"])
            t = obj["type"].lower()
            if t == "car":
                colors.append("red")
            elif t == "person":
                colors.append("green")
            elif t == "truck":
                colors.append("orange")
            elif t == "bus":
                colors.append("yellow")
            else:
                colors.append("blue")

        ax.scatter(xs, ys, c=colors, s=80, edgecolors="black")
        return scat,

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)
    plt.show()


def main():
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.txt"))[-1]
    print(f"🎬 Simulating from: {latest}")

    frames = parse_output_txt(latest)
    print(f"✅ Parsed {len(frames)} frames.")
    time.sleep(1)
    animate(frames)


if __name__ == "__main__":
    main()