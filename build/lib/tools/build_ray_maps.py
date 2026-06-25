"""Build ray maps from a manifest.

Usage:
    python tools/build_ray_maps.py path/to/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)


def build_ray_map(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
    width: int,
    height: int,
    coordinate_system: str = "opencv",
) -> np.ndarray:
    """Return an (H, W, 3) ray directions array in world coordinates."""
    ys, xs = np.indices((height, width))
    x = (xs - cx) / fx
    y = (ys - cy) / fy

    # OpenCV / COLMAP: camera looks at +Z
    if coordinate_system == "opencv":
        dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
    elif coordinate_system == "opengl":
        dirs_cam = np.stack([x, -y, -np.ones_like(x)], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    # Normalize before the rotation
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-9

    # Rotate into world frame using camera basis (c2w[:3,:3])
    R = c2w[:3, :3]
    dirs_world = dirs_cam @ R.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-9
    return dirs_world.astype(np.float32)


def process_manifest(manifest_path: Path, force: bool = False) -> None:
    data: Dict[str, Any] = json.loads(manifest_path.read_text())
    cs = data.get("coordinate_system", "opencv")
    out_dir = manifest_path.parent
    rays_dir = out_dir / "rays"
    rays_dir.mkdir(parents=True, exist_ok=True)

    for frame in data["frames"]:
        rays_rel = frame.get("rays")
        if rays_rel:
            out_path = out_dir / rays_rel
        else:
            out_path = rays_dir / f"{frame['id']}.npy"

        if out_path.exists() and not force:
            log.info("Skip existing %s", out_path)
            continue

        cam = frame["camera"]
        c2w = np.asarray(cam["c2w"], dtype=np.float64)
        rays = build_ray_map(
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
            c2w=c2w,
            width=cam["width"],
            height=cam["height"],
            coordinate_system=cs,
        )
        np.save(out_path, rays)
        # Update manifest with relative path
        try:
            frame["rays"] = str(out_path.relative_to(out_dir))
        except ValueError:
            frame["rays"] = str(out_path)

    manifest_path.write_text(json.dumps(data, indent=2))
    log.info("Ray maps written next to %s", manifest_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    process_manifest(args.manifest, force=args.force)


if __name__ == "__main__":
    main()