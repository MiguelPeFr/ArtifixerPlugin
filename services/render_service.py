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

from services.camera import Camera
from services.scene_adapter import Scene

log = logging.getLogger(__name__)


class RenderService:
    """Dispatches render calls to the LichtFeld renderer when available."""

    def __init__(self) -> None:
        self._lichfeld_render = None  # injected in on_load

    # ---- Public API ---------------------------------------------------------
    def render_rgb(self, scene: Scene, cam: Camera) -> np.ndarray:
        img = self._dispatch(scene, cam, pass_name="rgb")
        if img is None:
            img = self._render_with_lf(scene, cam)
        if img is None:
            img = self._synthetic_rgb(scene, cam)
        return self._ensure_uint8_rgb(img)

    def render_opacity(self, scene: Scene, cam: Camera) -> np.ndarray:
        img = self._dispatch(scene, cam, pass_name="opacity")
        if img is None:
            img = self._render_alpha_with_lf(scene, cam)
        if img is None:
            img = self._synthetic_opacity(scene, cam)
        return self._ensure_mask(img)

    def render_depth(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        img = self._dispatch(scene, cam, pass_name="depth")
        if img is None:
            img = self._synthetic_depth(scene, cam)
        return np.asarray(img, dtype=np.float32)

    def render_normal(self, scene: Scene, cam: Camera) -> Optional[np.ndarray]:
        img = self._dispatch(scene, cam, pass_name="normal")
        if img is None:
            return self._synthetic_normal(scene, cam)
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
            return None
        try:
            fn = getattr(r, f"render_{pass_name}", None)
            if fn is None:
                return None
            out = fn(camera=cam, scene=scene.raw)
            return np.asarray(out)
        except Exception:  # noqa: BLE001
            log.exception("LichtFeld render %s failed; using synthetic buffer", pass_name)
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

            frame_tensor = lf.render_at(
                eye=tuple(float(x) for x in eye),
                target=tuple(float(x) for x in target),
                width=int(cam.width),
                height=int(cam.height),
                fov=float(fov),
                up=tuple(float(x) for x in up),
            )
            if frame_tensor is None:
                return None
            return self._tensor_to_numpy(frame_tensor)
        except Exception:  # noqa: BLE001
            log.exception("lf.render_at failed; using synthetic fallback")
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
