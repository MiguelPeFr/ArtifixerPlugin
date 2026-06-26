"""Dataset writer.

Persists render buffers + camera intrinsics/extrinsics to disk and builds the
ArtiFixer manifest.

Layout produced (per ``mode``):
  preview  -> rgb/, cameras.json
  training -> rgb/, opacity/, manifest.json, cameras.json
  research -> rgb/, opacity/, depth/, normal/, manifest.json, cameras.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _Paths:
    rgb: Path
    opacity: Path
    depth: Path
    normal: Path
    rays: Path
    cameras: Path
    manifest: Path


class DatasetWriter:
    def __init__(self) -> None:
        self._paths: Optional[_Paths] = None

    # ---- Public API ---------------------------------------------------------
    def init_manifest(
        self,
        scene_id: str,
        source: str,
        ply_path: Optional[Path],
        width: int,
        height: int,
        coordinate_system: str = "opencv",
    ) -> Dict[str, Any]:
        return {
            "scene_id": scene_id,
            "source": source,
            "ply_path": str(ply_path) if ply_path else None,
            "image_width": int(width),
            "image_height": int(height),
            "coordinate_system": coordinate_system,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frames": [],
        }

    def write_frame(
        self,
        manifest: Dict[str, Any],
        frame_id: str,
        rgb: np.ndarray,
        camera,
        out_dir: Path,
        mode: str,
        opacity: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> None:
        paths = self._ensure_paths(out_dir)

        rgb_path = paths.rgb / f"{frame_id}.png"
        op_path = paths.opacity / f"{frame_id}.png" if opacity is not None else None
        depth_path = paths.depth / f"{frame_id}.exr" if depth is not None else None
        normal_path = paths.normal / f"{frame_id}.png" if normal is not None else None

        _save_image(rgb_path, rgb)

        rel_rgb = str(rgb_path.relative_to(out_dir))
        rel_op = None
        rel_depth = None
        rel_normal = None

        if opacity is not None:
            _save_image(op_path, opacity)
            rel_op = str(op_path.relative_to(out_dir))

        if depth is not None:
            depth_path = _save_depth(paths.depth / f"{frame_id}.exr", depth)
            rel_depth = str(depth_path.relative_to(out_dir))
        if normal is not None:
            _save_normal_as_png(normal_path, normal)
            rel_normal = str(normal_path.relative_to(out_dir))

        manifest["frames"].append(
            {
                "id": frame_id,
                "rgb": rel_rgb,
                "opacity": rel_op,
                "depth": rel_depth,
                "normal": rel_normal,
                "rays": None,  # filled later by tools/build_ray_maps.py
                "camera": camera.to_metadata(),
            }
        )

    def write_cameras_json(self, cameras, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cameras": [self._camera_payload(c) for c in cameras],
            "count": len(cameras),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        log.info("cameras.json written: %s", output_path)
        return output_path

    def write_manifest(self, out_dir: Path, manifest: Dict[str, Any]) -> Path:
        paths = self._ensure_paths(out_dir)
        paths.manifest.write_text(json.dumps(manifest, indent=2))
        log.info("manifest.json written: %s (%d frames)", paths.manifest, len(manifest["frames"]))
        return paths.manifest

    # ---- Helpers ------------------------------------------------------------
    @staticmethod
    def _camera_payload(camera) -> Dict[str, Any]:
        return camera.to_metadata() | {"name": getattr(camera, "name", "cam")}

    @staticmethod
    def _ensure_paths(out_dir: Path) -> _Paths:
        paths = _Paths(
            rgb=out_dir / "rgb",
            opacity=out_dir / "opacity",
            depth=out_dir / "depth",
            normal=out_dir / "normal",
            rays=out_dir / "rays",
            cameras=out_dir / "cameras.json",
            manifest=out_dir / "manifest.json",
        )
        for p in (paths.rgb, paths.opacity, paths.depth, paths.normal, paths.rays):
            p.mkdir(parents=True, exist_ok=True)
        return paths


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        if img.ndim == 3 and img.shape[-1] == 3:
            Image.fromarray(img, mode="RGB").save(path)
        else:
            Image.fromarray(img, mode="L").save(path)
    except Exception:  # noqa: BLE001
        log.exception("PIL save failed for %s", path)
        np.save(path.with_suffix(".npy"), img)


def _save_depth(path: Path, depth: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _exr_supported():
        try:
            import imageio.v3 as iio  # type: ignore

            iio.imwrite(path, depth.astype(np.float32))
            return path
        except Exception:  # noqa: BLE001
            log.warning("EXR write failed for %s; falling back to NPY", path)
    fallback = path.with_suffix(".npy")
    np.save(fallback, depth.astype(np.float32))
    return fallback


def _exr_supported() -> bool:
    try:
        import imageio.v3 as iio  # type: ignore

        # Probe: try a 1x1 EXR write to a temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".exr", delete=True) as tmp:
            iio.imwrite(tmp.name, np.zeros((1, 1), dtype=np.float32))
        return True
    except Exception:
        return False


def _save_normal_as_png(path: Path, normal: np.ndarray) -> None:
    """Encode normals ([-1,1]) as 8-bit RGB for easy previewing."""
    encoded = ((normal * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    _save_image(path, encoded)