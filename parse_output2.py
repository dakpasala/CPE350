#!/usr/bin/env python3
"""
parse_output2.py
Modified from your parse_output1.py to return parsed frames as Python objects.
"""

import re
from lxml import etree
from pathlib import Path


def extract_frames(xml_content: str):
    """Extract <Frame> blocks even if XML has multiple roots."""
    xml_clean = re.sub(r'\sxmlns(:\w+)?="[^"]+"', '', xml_content)
    xml_clean = re.sub(r'<(/?)(\w+:)', r'<\1', xml_clean)
    return re.findall(r'<Frame[^>]*>.*?</Frame>', xml_clean, flags=re.DOTALL)


def parse_frame(frame_xml: str):
    parser = etree.XMLParser(recover=True)
    try:
        frame = etree.fromstring(frame_xml.encode("utf-8"), parser)
    except Exception:
        return []

    frame_time = frame.get("UtcTime", "Unknown")
    objects = frame.findall("Object")
    parsed_objects = []

    for obj in objects:
        obj_id = obj.get("ObjectId", "N/A")
        obj_type = obj.findtext(".//Class/Type", default="Unknown")
        geo = obj.find(".//GeoLocation")
        lat = float(geo.get("lat")) if geo is not None else None
        lon = float(geo.get("lon")) if geo is not None else None
        speed = float(obj.findtext(".//Behaviour/Speed", default="0"))

        parsed_objects.append({
            "id": obj_id,
            "type": obj_type,
            "lat": lat,
            "lon": lon,
            "speed": speed,
            "time": frame_time
        })

    return parsed_objects

def parse_xml(file_path: str):
    xml_path = Path(file_path)
    if not xml_path.exists():
        print(f"❌ File not found: {file_path}")
        return []

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        xml_content = f.read()

    frames = extract_frames(xml_content)
    all_frames = [parse_frame(f) for f in frames]
    return all_frames
