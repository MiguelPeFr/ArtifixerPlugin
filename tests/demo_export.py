"""Standalone harness that exercises the full export pipeline.

Lets you verify the plugin end-to-end without LichtFeld installed: a fake
``app`` object and a tiny synthetic scene are used to drive the same code
paths that run inside the host.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

# Make ``plugin.py`` importable when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugin import ArtiFixerExportPlugin
from ui.export_panel import ExportSettings, ExportMode, run_headless
from services.camera_sampler import SamplerConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
class FakeCamera:
    """Stand-in for a LichtFeld camera."""
    def __init__(self, name, c2w, width=1024, height=1024, fx=900.0, fy=900.0):
        self.name = name
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = width / 2
        self.cy = height / 2
        self.c2w = np.asarray(c2w, dtype=np.float64)
        self.w2c = np.linalg.inv(self.c2w)


class FakeScene:
    def __init__(self, name: str, ply_path: Optional[Path], aabb: np.ndarray, cams: List[FakeCamera]):
        self.name = name
        self.ply_path = ply_path
        self.aabb = aabb
        self.cameras = cams
        self.renderer = None  # forces synthetic fallback in RenderService


@dataclass
class FakeApp:
    project_name: str = "demo_project"
    active_scene: Any = None


# --------------------------------------------------------------------------- #
def _build_demo_cameras(n: int = 4) -> List[FakeCamera]:
    """Build a small ring of cameras looking at the origin."""
    cams = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        eye = np.array([2.5 * np.cos(theta), 2.5 * np.sin(theta), 1.2])
        target = np.array([0.0, 0.0, 0.0])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)
        rot = np.stack([right, -new_up, forward], axis=1)
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = eye
        cams.append(FakeCamera(name=f"cam_{i:02d}", c2w=c2w))
    return cams


def run_demo(plugin_factory=ArtiFixerExportPlugin) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    aabb = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    fake_scene = FakeScene(
        name="object_001",
        ply_path=Path("./dummy.ply"),
        aabb=aabb,
        cams=_build_demo_cameras(n=6),
    )
    fake_app = FakeApp(project_name="demo_project", active_scene=fake_scene)

    plugin = plugin_factory()
    plugin.on_load(fake_app)

    # Export from the existing cameras, training preset, 512x512
    return run_headless(
        run_export=plugin._run_export,
        output_dir=str(ROOT / "_demo_out"),
        camera_mode="original",
        mode="training",
        resolution=(512, 512),
        num_views=12,
    )


def main() -> None:
    manifest = run_demo()
    print(f"\nManifest written to: {manifest}")

    # Validate against the schema if jsonschema is installed
    schema_path = ROOT / "schemas" / "manifest_schema.json"
    data = json.loads(manifest.read_text())
    try:
        import jsonschema  # type: ignore

        schema = json.loads(schema_path.read_text())
        jsonschema.validate(data, schema)
        print("Manifest schema: OK")
    except ImportError:
        print("jsonschema not installed; skipping validation")
    except Exception as e:  # noqa: BLE001
        print(f"Manifest schema validation FAILED: {e}")

    # Build ray maps
    from tools.build_ray_maps import process_manifest

    process_manifest(manifest)
    print("Ray maps generated.")


if __name__ == "__main__":
    main()