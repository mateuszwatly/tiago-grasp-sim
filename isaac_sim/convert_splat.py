#!/usr/bin/env python3
"""Convert PLY or SPZ Gaussian splats to OpenUSD particle fields."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.converter.gsplat")
        app.update()
        from omni.kit.converter.gsplat import (
            read_ply,
            read_spz,
            write_gaussian_splat_usd,
        )

        suffix = args.input.suffix.lower()
        if suffix not in {".ply", ".spz"}:
            raise ValueError("Gaussian splat input must be a .ply or .spz file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        source_path = str(args.input.resolve())
        splat = read_ply(source_path) if suffix == ".ply" else read_spz(source_path)
        write_gaussian_splat_usd(
            splat,
            output_path=str(args.output.resolve()),
            source_file=source_path,
            prim_name=args.output.stem,
            up_axis="Z",
        )
        print(f"Gaussian splat USD ready: {args.output}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
