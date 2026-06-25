"""Synthetic camera samplers.

Each sampler returns a list of :class:`Camera` objects positioned around an
object defined by ``(center, radius)``. The default config produces a 36
viewpoint orbit, but elevation / radius / look_at can all be tweaked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from services.camera import Camera

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
@dataclass
class SamplerConfig:
    num_views: int = 36
    radius_factor: float = 1.6         # multiplier over object radius
    elevation_min_deg: float = -15.0
    elevation_max_deg: float = 60.0
    look_at: Optional[Sequence[float]] = None  # world-space target
    width: int = 1024
    height: int = 1024
    fov_deg: float = 50.0              # used when intrinsics are missing
    rng_seed: int = 0
    # Multi-ring specific
    rings: int = 3
    # Manual mode
    manual_poses: List[Tuple[Sequence[float], Sequence[float]]] = field(
        default_factory=list
    )  # list of (position, look_at)


def _fov_to_focal(fov_deg: float, size: int) -> float:
    fov_rad = math.radians(fov_deg)
    return (size / 2.0) / math.tan(fov_rad / 2.0)


def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV / COLMAP-style camera-to-world matrix (camera looks at +Z)."""
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12

    right = np.cross(forward, up)
    n = np.linalg.norm(right)
    if n < 1e-8:
        # forward parallel to up; nudge
        up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        n = np.linalg.norm(right)
    right /= n

    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up) + 1e-12

    # Camera axes in world coords: x=right, y=down, z=forward (OpenCV)
    rot = np.stack([right, -new_up, forward], axis=1)  # (3, 3)
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = eye
    return c2w


class _BaseSampler:
    def __init__(self, cfg: SamplerConfig):
        self.cfg = cfg

    def _new_camera(self, idx: int, eye: np.ndarray, target: np.ndarray) -> Camera:
        up = np.array([0.0, 0.0, 1.0])
        c2w = _look_at_matrix(eye, target, up)
        w2c = np.linalg.inv(c2w)
        focal = _fov_to_focal(self.cfg.fov_deg, self.cfg.width)
        return Camera(
            name=f"synth_{idx:04d}",
            width=self.cfg.width,
            height=self.cfg.height,
            fx=focal,
            fy=focal,
            cx=self.cfg.width / 2,
            cy=self.cfg.height / 2,
            c2w=c2w,
            w2c=w2c,
        )


# --------------------------------------------------------------------------- #
class OrbitSampler(_BaseSampler):
    """Evenly spaced cameras on a horizontal ring."""

    def sample(self, center: np.ndarray, radius: float) -> List[Camera]:
        target = np.asarray(self.cfg.look_at) if self.cfg.look_at else center
        r = radius * self.cfg.radius_factor
        cams: List[Camera] = []
        for i in range(self.cfg.num_views):
            theta = 2 * math.pi * i / self.cfg.num_views
            elev = math.radians((self.cfg.elevation_min_deg + self.cfg.elevation_max_deg) / 2)
            eye = target + r * np.array(
                [math.cos(theta) * math.cos(elev),
                 math.sin(theta) * math.cos(elev),
                 math.sin(elev)]
            )
            cams.append(self._new_camera(i, eye, target))
        return cams


# --------------------------------------------------------------------------- #
class HemisphereSampler(_BaseSampler):
    """Cameras distributed over the upper hemisphere."""

    def sample(self, center: np.ndarray, radius: float) -> List[Camera]:
        target = np.asarray(self.cfg.look_at) if self.cfg.look_at else center
        r = radius * self.cfg.radius_factor
        cams: List[Camera] = []
        n = self.cfg.num_views
        # Fibonacci sphere restricted to elevation >= 0
        golden = math.pi * (3 - math.sqrt(5))
        for i in range(n):
            y = 1 - (i / max(n - 1, 1))  # 1 .. 0
            r_xy = math.sqrt(max(0.0, 1 - y * y))
            phi = golden * i
            direction = np.array([math.cos(phi) * r_xy, math.sin(phi) * r_xy, y])
            eye = target + r * direction
            cams.append(self._new_camera(i, eye, target))
        return cams


# --------------------------------------------------------------------------- #
class MultiRingSampler(_BaseSampler):
    """Concentric rings at multiple elevations."""

    def sample(self, center: np.ndarray, radius: float) -> List[Camera]:
        target = np.asarray(self.cfg.look_at) if self.cfg.look_at else center
        r = radius * self.cfg.radius_factor
        cams: List[Camera] = []
        rings = max(1, self.cfg.rings)
        per_ring = max(4, self.cfg.num_views // rings)
        idx = 0
        for k in range(rings):
            t = (k + 1) / rings
            elev_deg = self.cfg.elevation_min_deg + t * (
                self.cfg.elevation_max_deg - self.cfg.elevation_min_deg
            )
            elev = math.radians(elev_deg)
            for j in range(per_ring):
                theta = 2 * math.pi * j / per_ring + (math.pi / rings) * k
                eye = target + r * np.array(
                    [math.cos(theta) * math.cos(elev),
                     math.sin(theta) * math.cos(elev),
                     math.sin(elev)]
                )
                cams.append(self._new_camera(idx, eye, target))
                idx += 1
        return cams


# --------------------------------------------------------------------------- #
class ManualSampler(_BaseSampler):
    """User-supplied keyframes (position, look_at)."""

    def sample(self, center: np.ndarray, radius: float) -> List[Camera]:
        if not self.cfg.manual_poses:
            log.warning("Manual mode with no poses; falling back to orbit")
            return OrbitSampler(self.cfg).sample(center, radius)

        cams: List[Camera] = []
        for i, (pos, look) in enumerate(self.cfg.manual_poses):
            cams.append(self._new_camera(
                i,
                np.asarray(pos, dtype=np.float64),
                np.asarray(look, dtype=np.float64),
            ))
        return cams