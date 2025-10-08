#!/usr/bin/env python3
"""
simulate_output.py
Reads parsed_output.txt from /outputs and animates the detected objects
(frame-by-frame) using matplotlib.
"""

import re
import time
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation


def parse_output_txt(file_path: Path):
    """Parse the output text and return list of frames (each a list of objects)."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by frames
    frame_blocks = re.split(r"🕒 Frame #[0-9]+ \| Time:", text)
    frames = []

    for block in frame_blocks[1:]:
        # Each frame -> list of objects with type + position
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

            # center of bounding box
            x = (left + right) / 2
            y = (top + bottom) / 2
            objects.append({"type": obj_type, "x": x, "y": y})

        frames.append(objects)

    return frames


def animate(frames):
    """Animate frames using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("X position (normalized)")
    ax.set_ylabel("Y position (normalized)")
    ax.set_title("Object Movement Simulation")
    scat = ax.scatter([], [])

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_title(f"Frame {frame_idx + 1}/{len(frames)}")
        colors, xs, ys = [], [], []

        for obj in frames[frame_idx]:
            xs.append(obj["x"])
            ys.append(obj["y"])
            ## colors.append("red" if obj["type"].lower() == "car" else if obj["type"].lower() == "person" else "blue") e
            if obj["type"].lower() == "car":
                colors.append("red")
            elif obj["type"].lower() == "person":
                colors.append("green")
            elif obj["type"].lower() == "truck":
                colors.append("orange")
            elif obj["type"].lower() == "bus":
                colors.append("yellow")

        ax.scatter(xs, ys, c=colors, s=80)
        return scat,

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)
    plt.show()


def main():
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.txt"))[-1]  # use latest run
    print(f"🎬 Simulating from: {latest}")

    frames = parse_output_txt(latest)
    print(f"✅ Parsed {len(frames)} frames.")
    time.sleep(1)
    animate(frames)


if __name__ == "__main__":
    main()
