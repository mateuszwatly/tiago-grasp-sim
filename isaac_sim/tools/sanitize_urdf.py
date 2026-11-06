#!/usr/bin/env python3
"""Remove simulator-specific tags and make package mesh paths portable."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()
    for child in list(root):
        if child.tag in {"gazebo", "ros2_control", "transmission"}:
            root.remove(child)

    for element in root.iter():
        filename = element.attrib.get("filename")
        if not filename:
            continue
        portable = re.sub(
            r"^package://([^/]+)/", r"packages/\1/", filename
        )
        portable = re.sub(
            r"^file:///opt/tiago_ws/install/([^/]+)/share/\1/",
            r"packages/\1/",
            portable,
        )
        element.attrib["filename"] = portable

    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
