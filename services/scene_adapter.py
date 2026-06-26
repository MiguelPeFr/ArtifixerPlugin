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
import json
import urllib.request

from services.camera import Camera
from services.camera_sampler import (
    CameraRigSampler,
    HemisphereSampler,
    ManualSampler,
    MultiRingSampler,
    OrbitSampler,
    RigConfig,
    SamplerConfig,
)

log = logging.getLogger(__name__)


# #region debug-point A:scene-adapter
def _debug_report(hypothesis_id: str, location: str, msg: str, data: Optional[dict] = None) -> None:
    _p = ".dbg/preset-render-fallback.env"
    _u, _s = "http://127.0.0.1:7777/event", "preset-render-fallback"
    try:
        with open(_p, "r", encoding="utf-8") as f:
            c = f.read()
        for line in c.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                _u = line.split("=", 1)[1]
            elif line.startswith("DEBUG_SESSION_ID="):
                _s = line.split("=", 1)[1]
    except Exception:
        pass
    try:
        payload = {
            "sessionId": _s,
            "runId": "pre",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
        }
        req = urllib.request.Request(
            _u,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=1).read()
    except Exception:
        pass
# #endregion


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
def _write_camera_probe(out_dir: Optional[Path], payload: dict) -> None:
    """Persist camera probe diagnostics to ``camera_probe.json``.

    The file is best-effort: failures are swallowed silently so the export
    pipeline is never blocked by a diagnostic step.
    """
    if out_dir is None:
        return
    try:
        target = Path(out_dir) / "camera_probe.json"
        target.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class SceneAdapter:
    """Reads cameras and scene metadata from the active LichtFeld project."""

    # ---- High level ---------------------------------------------------------
    def get_active_scene(self, app: Any, out_dir: Optional[Path] = None) -> Scene:
        """Return the currently loaded scene from the host application.

        The ``out_dir`` argument is optional and is used to drop a diagnostic
        file (``camera_probe.json``) listing every LichtFeld API entry point
        we tried while looking for the active scene and its cameras.
        """
        probe: dict = {
            "app_provided": app is not None,
            "app_type": type(app).__name__ if app is not None else None,
            "lf_available": False,
            "lf_scene": None,
            "app_active_scene": None,
            "probe_results": {},
        }

        try:
            import lichtfeld as lf  # type: ignore

            probe["lf_available"] = True
            lf_scene = self._read_scene_from_lf()
            probe["lf_scene"] = {
                "found": lf_scene is not None,
                "type": type(lf_scene).__name__ if lf_scene is not None else None,
                "name": getattr(lf_scene, "name", None) if lf_scene is not None else None,
            }
            if lf_scene is not None:
                cams = self._read_cameras(lf_scene)
                probe["lf_cameras"] = {
                    "count": len(cams),
                    "names": [c.name for c in cams[:8]],
                    "first_fx": cams[0].fx if cams else None,
                }
                aabb = self._read_aabb(lf_scene)
                probe["lf_aabb"] = (
                    aabb.tolist() if aabb is not None else None
                )
                _write_camera_probe(out_dir, probe)
                return self._wrap_scene(lf_scene)
        except Exception as exc:  # noqa: BLE001
            probe["lf_error"] = repr(exc)

        if app is not None:
            scene_obj = getattr(app, "active_scene", None)
            probe["app_active_scene"] = {
                "found": scene_obj is not None,
                "type": type(scene_obj).__name__ if scene_obj is not None else None,
                "name": getattr(scene_obj, "name", None) if scene_obj is not None else None,
            }
            if scene_obj is not None:
                cams = self._read_cameras(scene_obj)
                probe["app_cameras"] = {
                    "count": len(cams),
                    "names": [c.name for c in cams[:8]],
                    "first_fx": cams[0].fx if cams else None,
                }
                _write_camera_probe(out_dir, probe)
                return self._wrap_scene(scene_obj)

        _write_camera_probe(out_dir, probe)
        log.warning("No LichtFeld scene or app context found; returning empty scene")
        return Scene(name="empty_scene")

    @staticmethod
    def _read_scene_from_lf() -> Any:
        try:
            import lichtfeld as lf  # type: ignore

            return lf.get_scene()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _enumerate_attrs(obj: Any) -> dict:
        """Return a snapshot of every public attribute on ``obj``.

        Only metadata is captured (type name, length if available, first
        item's type if it's iterable). Values are never materialised in
        full so the probe stays cheap on large scenes.
        """
        info: dict = {}
        if obj is None:
            return info
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception as exc:  # noqa: BLE001
                info[name] = {"accessible": False, "error": repr(exc)}
                continue
            entry: dict = {"type": type(value).__name__}
            try:
                if hasattr(value, "__len__"):
                    entry["len"] = len(value)
            except Exception:  # noqa: BLE001
                pass
            try:
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
                    iterator = iter(value)
                    first = next(iterator, None)
                    if first is not None:
                        entry["first_type"] = type(first).__name__
            except Exception:  # noqa: BLE001
                pass
            info[name] = entry
        return info

    def collect_cameras(
        self,
        scene: Scene,
        mode: str = "original",
        sampler_cfg: Optional[SamplerConfig] = None,
    ) -> List[Camera]:
        """Return the cameras to render/export.

        mode:
            original   -> use scene.cameras as is (preferred when available)
            orbit      -> orbit around scene AABB
            hemisphere -> upper hemisphere orbit
            multi_ring -> concentric rings at multiple elevations
            manual     -> user supplied keyframes (sampler_cfg.manual_poses)

        If the project already exposes real cameras, ``original`` is used
        even when another mode is requested. Synthetic modes are only used
        when no real cameras exist or when ``force_synthetic`` is True.
        """
        mode = mode.lower()

        # Prefer real cameras from the project whenever they exist
        if scene.cameras and mode != "manual":
            log.info(
                "Using %d real cameras from the active project (requested mode=%s)",
                len(scene.cameras),
                mode,
            )
            return list(scene.cameras)

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
        elif mode == "rig":
            cfg_rig = rig_cfg or RigConfig(
                distance=max(radius * 1.6, 2.5),
                num_cameras=int(getattr(cfg, "num_views", 24)),
                rings=int(getattr(cfg, "rings", 2)),
            )
            cams = CameraRigSampler(cfg_rig).sample(center, radius)
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
        _debug_report("E", "scene_adapter.py:_wrap_scene", "wrapped scene", {
            "scene_name": getattr(scene_obj, "name", None),
            "camera_count": len(cameras),
            "has_aabb": aabb is not None,
            "ply_path": str(self._read_ply_path(scene_obj)) if self._read_ply_path(scene_obj) else None,
        })
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
