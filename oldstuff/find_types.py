#!/usr/bin/env python3
"""
find_unique_types.py
Scans output text (from parsed metadata) and prints all unique object types.
"""

import re
from pathlib import Path

def find_unique_types(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # extract all "Type: Something" patterns
    types = re.findall(r"Type:\s+([A-Za-z0-9_]+)", text)

    # store unique types in a set
    unique_types = set(types)

    print("✅ Unique object types found:")
    print(unique_types)


if __name__ == "__main__":
    # adjust path if needed
    file_path = Path("outputs/output1.txt")
    find_unique_types(file_path)
