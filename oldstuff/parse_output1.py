#!/usr/bin/env python3
"""
parse_output1.py
Parses Bosch / ONVIF metadata XML (multiple <MetadataStream> roots),
extracts all <Frame> elements, and writes the full parsed output
to sequentially numbered text files inside the /outputs directory.
"""

import re
from lxml import etree
from pathlib import Path
from textwrap import indent


def extract_frames(xml_content: str):
    """Extract <Frame> blocks even if XML has multiple roots."""
    xml_clean = re.sub(r'\sxmlns(:\w+)?="[^"]+"', '', xml_content)
    xml_clean = re.sub(r'<(/?)(\w+:)', r'<\1', xml_clean)
    return re.findall(r'<Frame[^>]*>.*?</Frame>', xml_clean, flags=re.DOTALL)


def next_output_path() -> Path:
    """Find the next available numbered output file in ./outputs/."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    # Find highest existing outputN.txt and increment
    existing = sorted(out_dir.glob("output*.txt"))
    next_num = 1
    if existing:
        last = existing[-1].stem  # e.g., "output3"
        try:
            last_num = int(last.replace("output", ""))
            next_num = last_num + 1
        except ValueError:
            pass
    return out_dir / f"output{next_num}.txt"


def parse_frame(frame_xml: str, index: int, out):
    """Parse one <Frame> XML segment and write data to output file."""
    parser = etree.XMLParser(recover=True)
    try:
        frame = etree.fromstring(frame_xml.encode("utf-8"), parser)
    except Exception:
        out.write(f"⚠️ Could not parse frame #{index}\n")
        return

    frame_time = frame.get("UtcTime", "Unknown")
    out.write(f"\n🕒 Frame #{index} | Time: {frame_time}\n")

    objects = frame.findall("Object")
    if not objects:
        out.write(indent("No detected objects in this frame.\n", "   "))
        return

    for obj in objects:
        obj_id = obj.get("ObjectId", "N/A")

        appearance = obj.find("Appearance")
        velocity = appearance.get("velocity") if appearance is not None else "N/A"
        area = appearance.get("area") if appearance is not None else "N/A"

        bbox = appearance.find(".//BoundingBox") if appearance is not None else None
        if bbox is not None:
            top = bbox.get("top", "?")
            bottom = bbox.get("bottom", "?")
            left = bbox.get("left", "?")
            right = bbox.get("right", "?")
        else:
            top = bottom = left = right = "?"

        obj_type = obj.findtext(".//Class/Type", default="Unknown")
        likelihood = obj.findtext(".//ClassCandidate/Likelihood", default="N/A")

        geo = obj.find(".//GeoLocation")
        lat = geo.get("lat") if geo is not None else "N/A"
        lon = geo.get("lon") if geo is not None else "N/A"
        elevation = geo.get("elevation") if geo is not None else "N/A"

        speed = obj.findtext(".//Behaviour/Speed", default="N/A")

        out.write(indent(f"🧩 Object ID: {obj_id}\n", "   "))
        out.write(indent(f"Type: {obj_type} (Likelihood: {likelihood})\n", "      "))
        out.write(indent(f"Velocity: {velocity} | Area: {area} | Speed: {speed}\n", "      "))
        out.write(indent(f"Bounding Box: top={top}, bottom={bottom}, left={left}, right={right}\n", "      "))
        out.write(indent(f"Geo: (lat={lat}, lon={lon}, elev={elevation})\n", "      "))


def parse_xml(file_path: str):
    """Main entry point: parse XML, write results to sequential output file."""
    xml_path = Path(file_path)
    if not xml_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        xml_content = f.read()

    frames = extract_frames(xml_content)

    # pick output filename automatically
    output_file = next_output_path()

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"✅ Parsed XML: {xml_path.name}\n")
        out.write("=" * 90 + "\n")

        if not frames:
            out.write("⚠️ No <Frame> segments found in file.\n")
            print("⚠️ No <Frame> segments found in file.")
            return

        out.write(f"🎞️ Found {len(frames)} frame(s) in total.\n")
        out.write("-" * 90 + "\n")

        for i, frame_xml in enumerate(frames, start=1):
            parse_frame(frame_xml, i, out)

        out.write("\n" + "=" * 90 + "\n")

    print(f"✅ Done! Output saved to: {output_file.resolve()}")


if __name__ == "__main__":
    parse_xml("output1.xml")
