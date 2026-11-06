#!/usr/bin/env python3
"""Convert the exported PAL TIAGo URDF into an Isaac Sim 6.2 USD asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = SimulationApp({"headless": True})
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.asset.importer.urdf")
        app.update()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        config = URDFImporterConfig(
            urdf_path=str(args.urdf.resolve()),
            usd_path=str(args.output.parent.resolve()),
            fix_base=True,
            merge_fixed_joints=False,
            merge_mesh=False,
            collision_from_visuals=False,
            collision_type="Convex Hull",
            allow_self_collision=False,
            robot_type="Mobile Manipulators",
            joint_drive_type="force",
            joint_target_type="position",
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        imported_path = Path(URDFImporter(config).import_urdf())
        if imported_path.resolve() != args.output.resolve():
            args.output.write_text(
                f"#usda 1.0\n(\n    subLayers = [@{imported_path.as_posix()}@]\n)\n",
                encoding="utf-8",
            )
        print(f"TIAGo USD ready: {args.output}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
