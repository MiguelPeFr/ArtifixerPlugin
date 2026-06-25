"""Scene adapter.

Reads the active LichtFeld scene: PLY / gaussians path, intrinsic and
extrinsic cameras. Also produces synthetic cameras via :mod:`camera_sampler`
when the project does not provide enough coverage.

The adapter is intentionally written against a thin duck-typed interface so
that it works with both the real LichtFeld runtime *and* a fake app used by
the demo harness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

from services.camera import Camera
from services.camera_sampler import (
    HemisphereSampler,
    ManualSampler,
    MultiRingSampler,
    OrbitSampler,
    SamplerConfig,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Scene data class
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    name: str = "scene"
    ply_path: Optional[Path] = None
    cameras: List[Camera] = field(default_factory=list)
    aabb: Optional[np.ndarray] = None  # (2, 3) min/max
    raw: Any = None  # reference to the live LichtFeld scene object, if any


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class SceneAdapter:
    """Reads cameras and scene metadata from the active LichtFeld project."""

    # ---- High level ---------------------------------------------------------
    def get_active_scene(self, app: Any) -> Scene:
        """Return the currently loaded scene from the host application."""
        if app is None:
            # Inside LichtFeld, plugins can still access the active scene via
            # the module-level API even if no explicit app handle is passed in.
            scene_obj = self._read_scene_from_lf()
            if scene_obj is not None:
                return self._wrap_scene(scene_obj)

            # No host: return a minimal scene so the rest of the pipeline
            # can still run in standalone / test mode.
            log.warning("No app context; returning empty scene")
            return Scene(name="empty_scene")

        scene_obj = getattr(app, "active_scene", None)
        if scene_obj is None:
            return Scene(name=getattr(app, "project_name", "scene"))

        return self._wrap_scene(scene_obj)

    @staticmethod
    def _read_scene_from_lf() -> Any:
        try:
            import lichtfeld as lf  # type: ignore

            return lf.get_scene()
        except Exception:  # noqa: BLE001
            return None

    def collect_cameras(
        self,
        scene: Scene,
        mode: str = "original",
        sampler_cfg: Optional[SamplerConfig] = None,
    ) -> List[Camera]:
        """Return the cameras to render/export.

        mode:
            original   -> use scene.cameras as is
            orbit      -> orbit around scene AABB
            hemisphere -> upper hemisphere orbit
            multi_ring -> concentric rings at multiple elevations
            manual     -> user supplied keyframes (sampler_cfg.manual_poses)
        """
        mode = mode.lower()
        if mode == "original":
            return list(scene.cameras)

        cfg = sampler_cfg or SamplerConfig()
        if scene.aabb is None:
            scene.aabb = self._estimate_aabb(scene)
        center, radius = self._aabb_center_radius(scene.aabb)

        if mode == "orbit":
            cams = OrbitSampler(cfg).sample(center, radius)
        elif mode == "hemisphere":
            cams = HemisphereSampler(cfg).sample(center, radius)
        elif mode == "multi_ring":
            cams = MultiRingSampler(cfg).sample(center, radius)
        elif mode == "manual":
            cams = ManualSampler(cfg).sample(center, radius)
        else:
            raise ValueError(f"Unknown camera mode: {mode!r}")

        return cams

    # ---- Internals ----------------------------------------------------------
    def _wrap_scene(self, scene_obj: Any) -> Scene:
        """Convert a host object into our internal Scene."""
        cameras = self._read_cameras(scene_obj)
        aabb = self._read_aabb(scene_obj)
        return Scene(
            name=getattr(scene_obj, "name", "scene"),
            ply_path=self._read_ply_path(scene_obj),
            cameras=cameras,
            aabb=aabb,
            raw=scene_obj,
        )

    @staticmethod
    def _read_ply_path(scene_obj: Any) -> Optional[Path]:
        for attr in ("ply_path", "model_path", "gaussian_path"):
            v = getattr(scene_obj, attr, None)
            if v:
                return Path(v)
        return None

    @staticmethod
    def _read_cameras(scene_obj: Any) -> List[Camera]:
        """Best-effort camera extraction.

        Supports a few common shapes:
            - scene.cameras: iterable of objects with intrinsics/extrinsics
            - scene.cameras: iterable of dicts (COLMAP-style)
            - scene.dataset.cameras / scene.train_cameras
        """
        candidates: Iterable[Any] = (
            getattr(scene_obj, "cameras", None),
            getattr(getattr(scene_obj, "dataset", None), "cameras", None),
            getattr(scene_obj, "train_cameras", None),
        )
        for raw in candidates:
            if raw is None:
                continue
            return [SceneAdapter._coerce_camera(c) for c in raw]
        return []

    @staticmethod
    def _coerce_camera(c: Any) -> Camera:
        if isinstance(c, Camera):
            return c

        if isinstance(c, dict):
            intr = c.get("intrinsics", {})
            ext = c.get("extrinsics", {})
            return Camera(
                name=c.get("name", c.get("id", "cam")),
                width=int(c.get("width", intr.get("width", 1024))),
                height=int(c.get("height", intr.get("height", 1024))),
                fx=float(intr.get("fx", 1000.0)),
                fy=float(intr.get("fy", 1000.0)),
                cx=float(intr.get("cx", intr.get("width", 1024) / 2)),
                cy=float(intr.get("cy", intr.get("height", 1024) / 2)),
                c2w=np.asarray(ext.get("c2w", np.eye(4)), dtype=np.float64),
                w2c=np.asarray(ext.get("w2c", np.eye(4)), dtype=np.float64),
            )

        return Camera(
            name=getattr(c, "name", "cam"),
            width=int(getattr(c, "width", 1024)),
            height=int(getattr(c, "height", 1024)),
            fx=float(getattr(c, "fx", 1000.0)),
            fy=float(getattr(c, "fy", 1000.0)),
            cx=float(getattr(c, "cx", getattr(c, "width", 1024) / 2)),
            cy=float(getattr(c, "cy", getattr(c, "height", 1024) / 2)),
            c2w=np.asarray(getattr(c, "c2w", np.eye(4)), dtype=np.float64),
            w2c=np.asarray(getattr(c, "w2c", np.eye(4)), dtype=np.float64),
        )

    @staticmethod
    def _read_aabb(scene_obj: Any) -> Optional[np.ndarray]:
        aabb = getattr(scene_obj, "aabb", None)
        if aabb is None:
            return None
        try:
            arr = np.asarray(aabb, dtype=np.float64)
            return arr.reshape(2, 3)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _estimate_aabb(scene: Scene) -> np.ndarray:
        """Fall back to a unit cube if the host did not expose an AABB."""
        cams = scene.cameras
        if cams:
            centers = np.stack([c.c2w[:3, 3] for c in cams])
            mn, mx = centers.min(0), centers.max(0)
            pad = (mx - mn).max() * 0.25 + 1e-3
            return np.stack([mn - pad, mx + pad], axis=0)
        return np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float64)

    @staticmethod
    def _aabb_center_radius(aabb: np.ndarray) -> tuple[np.ndarray, float]:
        center = (aabb[0] + aabb[1]) * 0.5
        radius = float(np.linalg.norm(aabb[1] - aabb[0]) * 0.5)
        return center, radius
