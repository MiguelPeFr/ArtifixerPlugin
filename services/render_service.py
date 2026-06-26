"""Render service.

Calls the LichtFeld renderer (or a stub for headless tests) and returns
buffers as numpy arrays:

  - rgb       : (H, W, 3) uint8
  - opacity   : (H, W)    uint8 (or float32 in [0,1])
  - depth     : (H, W)    float32
  - normal    : (H, W, 3) float32 in [-1, 1]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import json
import urllib.request

from services.camera import Camera
from services.scene_adapter import Scene

log = logging.getLogger(__name__)


# #region debug-point B:render-service
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


class RenderService:
    """Dispatches render calls to the LichtFeld renderer when available."""

    def __init__(self) -> None:
        self._lichfeld_render = None  # injected in on_load
        self.debug_events: list[dict[str, Any]] = []

    # ---- Public API ---------------------------------------------------------
    def render_rgb(self, scene: Scene, cam: Camera) -> np.ndarray:
        event: dict[str, Any] = {
            "scene_name": scene.name,
            "camera_name": cam.name,
            "size": [int(cam.width), int(cam.height)],
            "path": None,
        }
        img = self._dispatch(scene, cam, pass_name="rgb")
        _debug_report("B", "render_service.py:render_rgb", "after _dispatch rgb", {
            "dispatch_hit": img is not None,
            "scene_name": scene.name,
            "camera_name": cam.name,
            "camera_size": [cam.width, cam.height],
        })
        if img is None:
            img = self._render_with_lf(scene, cam)
            if img is not None:
                event["path"] = "lf.render_at"
        _debug_report("B", "render_service.py:render_rgb", "after _render_with_lf rgb", {
            "lf_render_hit": img is not None,
            "scene_name": scene.name,
            "camera_name": cam.name,
        })
        if img is None:
            _debug_report("B", "render_service.py:render_rgb", "falling back to synthetic rgb", {
                "scene_name": scene.name,
                "camera_name": cam.name,
            })
            img = self._synthetic_rgb(scene, cam)
            event["path"] = "synthetic"
        elif event["path"] is None:
            event["path"] = "scene.renderer"
        event["shape"] = list(np.asarray(img).shape)
        self.debug_events.append(event)
        return self._ensure_uint8_rgb(img)

    def save_rgb(self, path: Any, frame_tensor: Any) -> bool:
        """Save a frame with ``lf.io.save_image`` when available."""
        try:
            import lichtfeld.io as lf_io  # type: ignore
            lf_io.save_image(str(path), frame_tensor)
            return True
        except Exception:  # noqa: BLE001
            return False

    def render_opacity(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        """Return the opacity buffer for the given camera.

        Preference order:
          1. ``scene.renderer.render_opacity`` if available
          2. Alpha channel of the RGB(A) frame from ``lf.render_at``
          3. ``None`` when the host has no training buffer to read from
        """
        img = self._dispatch(scene, cam, pass_name="opacity")
        if img is not None:
            return self._ensure_mask(img)
        alpha = self._render_alpha_with_lf(scene, cam)
        if alpha is not None:
            return self._ensure_mask(alpha)
        return None

    def render_depth(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        """Return the depth buffer for the given camera.

        ``None`` is returned when the host has no training buffer (i.e. when
        the user has not started or has stopped training). Synthetic
        gradients are intentionally not produced so the dataset reflects the
        actual scene state.
        """
        img = self._dispatch(scene, cam, pass_name="depth")
        if img is None:
            return None
        return np.asarray(img, dtype=np.float32)

    def render_normal(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        """Return the normal buffer for the given camera.

        Same policy as :meth:`render_depth`.
        """
        img = self._dispatch(scene, cam, pass_name="normal")
        if img is None:
            return None
        return np.asarray(img, dtype=np.float32)

    # ---- Internals ----------------------------------------------------------
    def _dispatch(self, scene: Scene, cam: Camera, pass_name: str) -> Optional[np.ndarray]:
        """Try the real LichtFeld renderer first, fall back to None."""
        if self._lichfeld_render is None and scene.raw is not None:
            r = getattr(scene.raw, "renderer", None)
            if r is not None:
                self._lichfeld_render = r

        r = self._lichfeld_render
        if r is None:
            _debug_report("B", "render_service.py:_dispatch", "no scene renderer available", {
                "scene_name": scene.name,
                "has_scene_raw": scene.raw is not None,
            })
            return None
        try:
            fn = getattr(r, f"render_{pass_name}", None)
            if fn is None:
                _debug_report("B", "render_service.py:_dispatch", "renderer missing pass", {
                    "pass_name": pass_name,
                    "renderer_type": type(r).__name__,
                })
                return None
            out = fn(camera=cam, scene=scene.raw)
            _debug_report("B", "render_service.py:_dispatch", "renderer pass succeeded", {
                "pass_name": pass_name,
                "renderer_type": type(r).__name__,
            })
            return np.asarray(out)
        except Exception:  # noqa: BLE001
            log.exception("LichtFeld render %s failed; using synthetic buffer", pass_name)
            _debug_report("B", "render_service.py:_dispatch", "renderer pass raised exception", {
                "pass_name": pass_name,
                "renderer_type": type(r).__name__,
            })
            return None

    def _render_with_lf(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        """Fallback for hosts that expose rendering via ``lf.render_at``."""
        try:
            import lichtfeld as lf  # type: ignore

            eye = cam.c2w[:3, 3]
            forward = cam.c2w[:3, 2]
            target = eye + forward
            up = -cam.c2w[:3, 1]
            fov = self._camera_fov_deg(cam)
            _debug_report("C", "render_service.py:_render_with_lf", "calling lf.render_at", {
                "scene_name": scene.name,
                "camera_name": cam.name,
                "eye": [float(x) for x in eye],
                "target": [float(x) for x in target],
                "up": [float(x) for x in up],
                "fov": float(fov),
                "size": [int(cam.width), int(cam.height)],
            })

            frame_tensor = lf.render_at(
                eye=tuple(float(x) for x in eye),
                target=tuple(float(x) for x in target),
                width=int(cam.width),
                height=int(cam.height),
                fov=float(fov),
                up=tuple(float(x) for x in up),
            )
            if frame_tensor is None:
                _debug_report("C", "render_service.py:_render_with_lf", "lf.render_at returned None", {
                    "scene_name": scene.name,
                    "camera_name": cam.name,
                })
                return None
            arr = self._tensor_to_numpy(frame_tensor)
            _debug_report("C", "render_service.py:_render_with_lf", "lf.render_at returned frame", {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": int(arr.min()) if arr.size else None,
                "max": int(arr.max()) if arr.size else None,
            })
            return arr
        except Exception:  # noqa: BLE001
            log.exception("lf.render_at failed; using synthetic fallback")
            _debug_report("C", "render_service.py:_render_with_lf", "lf.render_at raised exception", {
                "scene_name": scene.name,
                "camera_name": cam.name,
            })
            return None

    def _render_alpha_with_lf(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        rgba = self._render_with_lf(scene, cam)
        if rgba is None or rgba.ndim != 3 or rgba.shape[-1] < 4:
            return None
        alpha = rgba[..., 3]
        return np.asarray(alpha)

    # ---- Synthetic fallback (used by the demo) ------------------------------
    @staticmethod
    def _synthetic_rgb(scene: Scene, cam: Camera) -> np.ndarray:
        """Render a soft gradient + projected silhouette of the AABB."""
        h, w = cam.height, cam.width
        ys, xs = np.indices((h, w))
        u = (xs - cam.cx) / max(cam.fx, 1e-6)
        v = (ys - cam.cy) / max(cam.fy, 1e-6)

        # Project AABB corners as a coarse silhouette
        aabb = scene.aabb if scene.aabb is not None else np.array([[-1, -1, -1], [1, 1, 1]])
        xs = (aabb[0, 0], aabb[1, 0])
        ys = (aabb[0, 1], aabb[1, 1])
        zs = (aabb[0, 2], aabb[1, 2])
        corners = np.array(
            [[x, y, z] for x in xs for y in ys for z in zs],
            dtype=np.float64,
        )
        world = (cam.w2c @ np.concatenate([corners, np.ones((8, 1))], axis=1).T).T[:, :3]
        depth = world[:, 2]
        valid = depth > 0.01
        xpix = cam.cx + world[valid, 0] * cam.fx / np.clip(depth[valid], 1e-3, None)
        ypix = cam.cy + world[valid, 1] * cam.fy / np.clip(depth[valid], 1e-3, None)
        mask = np.zeros((h, w), dtype=np.float32)
        for x, y in zip(xpix, ypix):
            cx, cy = int(x), int(y)
            if 0 <= cx < w and 0 <= cy < h:
                rr = max(2, int(min(h, w) * 0.02))
                y0, y1 = max(0, cy - rr), min(h, cy + rr)
                x0, x1 = max(0, cx - rr), min(w, cx + rr)
                mask[y0:y1, x0:x1] = 1.0

        # Background gradient
        bg = np.stack([
            (u * 0.5 + 0.5),
            (v * 0.5 + 0.5),
            np.full_like(u, 0.7),
        ], axis=-1).astype(np.float32)
        fg = np.array([0.85, 0.45, 0.35], dtype=np.float32)

        rgb = bg * (1 - mask[..., None]) + fg * mask[..., None]
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _synthetic_opacity(scene: Scene, cam: Camera) -> np.ndarray:
        rgb = RenderService._synthetic_rgb(scene, cam)
        # Use the difference vs background as a proxy for accumulated opacity.
        bg_white = 255
        opacity = 1.0 - np.abs(rgb.astype(np.float32) - bg_white).max(-1) / 255.0
        return np.clip(opacity * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _synthetic_depth(scene: Scene, cam: Camera) -> np.ndarray:
        """Radial depth centred on the optical axis."""
        h, w = cam.height, cam.width
        ys, xs = np.indices((h, w))
        u = (xs - cam.cx) / cam.fx
        v = (ys - cam.cy) / cam.fy
        d = np.sqrt(u * u + v * v + 1.0)
        return d.astype(np.float32)

    @staticmethod
    def _synthetic_normal(scene: Scene, cam: Camera) -> np.ndarray:
        h, w = cam.height, cam.width
        ys, xs = np.indices((h, w))
        u = (xs - cam.cx) / cam.fx
        v = (ys - cam.cy) / cam.fy
        z = np.ones_like(u)
        n = np.stack([u, v, z], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        return n.astype(np.float32)

    # ---- Helpers ------------------------------------------------------------
    @staticmethod
    def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] >= 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    @staticmethod
    def _ensure_mask(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = img[..., 0]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _camera_fov_deg(cam: Camera) -> float:
        return math.degrees(2.0 * math.atan2(float(cam.width), 2.0 * max(float(cam.fx), 1e-6)))

    @staticmethod
    def _tensor_to_numpy(frame_tensor: Any) -> np.ndarray:
        if hasattr(frame_tensor, "numpy"):
            arr = frame_tensor.numpy()
        else:
            arr = np.asarray(frame_tensor)

        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
