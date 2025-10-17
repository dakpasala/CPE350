#!/usr/bin/env python3
"""
simulate_output_plotly.py
3D visualization of detected objects with interactive Play/Pause + slider (Plotly version)
"""

import re
from pathlib import Path
import plotly.graph_objects as go


# -------------------------------------------------------
# Parse frames from your output text file
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
# Create a 3D Plotly animation
# -------------------------------------------------------
def make_plotly_animation(frames):
    # Map object type → color
    color_map = {
        "car": "red",
        "person": "green",
        "truck": "orange",
        "bus": "yellow"
    }

    # --- Base frame (first one) ---
    first_frame = frames[0]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[obj["x"] for obj in first_frame],
                y=[obj["y"] for obj in first_frame],
                z=[0 for _ in first_frame],
                mode="markers",
                marker=dict(
                    size=6,
                    color=[color_map.get(obj["type"].lower(), "blue") for obj in first_frame],
                    opacity=0.9,
                    line=dict(width=1, color="black")
                ),
            ),
            # Draw 2D "road" planes
            go.Surface(
                z=[[0, 0], [0, 0]],
                x=[[-0.6, 1.2], [-0.6, 1.2]],
                y=[[1.1, -0.3], [1.1, -0.3]],
                showscale=False,
                opacity=0.3,
                colorscale=[[0, "gray"], [1, "gray"]],
                name="Foothill Blvd"
            ),
            go.Surface(
                z=[[0, 0], [0, 0]],
                x=[[-0.6, 1.2], [-0.6, 1.2]],
                y=[[-0.3, 1.1], [-0.3, 1.1]],
                showscale=False,
                opacity=0.3,
                colorscale=[[0, "gray"], [1, "gray"]],
                name="Santa Rosa St"
            )
        ]
    )

    # --- Build animation frames ---
    anim_frames = []
    for i, frame_data in enumerate(frames):
        anim_frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=[obj["x"] for obj in frame_data],
                    y=[obj["y"] for obj in frame_data],
                    z=[0 for _ in frame_data],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=[color_map.get(obj["type"].lower(), "blue") for obj in frame_data],
                        opacity=0.9,
                        line=dict(width=1, color="black")
                    ),
                )
            ],
            name=f"Frame {i+1}"
        ))

    fig.frames = anim_frames

    # --- Layout & Animation Controls ---
    fig.update_layout(
        title="Foothill & Santa Rosa 3D Intersection",
        scene=dict(
            xaxis=dict(title="X position (normalized)", range=[-0.5, 1.0]),
            yaxis=dict(title="Y position (normalized)", range=[0.0, 1.0]),
            zaxis=dict(title="Z (flat plane)", range=[-0.1, 0.1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.25, y=1.25, z=0.8)),  # nice initial angle
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True},
                                        "fromcurrent": True,
                                        "mode": "immediate"}],
                        "label": "▶️ Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}],
                        "label": "⏸ Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "type": "buttons",
                "x": 0.4,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }
        ],
        sliders=[
            {
                "pad": {"b": 10, "t": 50},
                "len": 0.8,
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
                "steps": [
                    {
                        "args": [[f"Frame {k+1}"],
                                 {"frame": {"duration": 0, "redraw": True},
                                  "mode": "immediate"}],
                        "label": str(k+1),
                        "method": "animate"
                    }
                    for k in range(len(frames))
                ]
            }
        ]
    )

    fig.show()
    # Optional: save for web sharing
    # fig.write_html("intersection_3d.html")


# -------------------------------------------------------
# Run
# -------------------------------------------------------
def main():
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.txt"))[-1]
    print(f"🎬 Simulating from: {latest}")

    frames = parse_output_txt(latest)
    print(f"✅ Parsed {len(frames)} frames.")
    make_plotly_animation(frames)


if __name__ == "__main__":
    main()
