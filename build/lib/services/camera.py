"""Shared camera data class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Camera:
    """Pinhole camera with intrinsics + extrinsics."""

    name: str = "cam"
    width: int = 1024
    height: int = 1024
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 512.0
    cy: float = 512.0
    # 4x4 matrices. c2w is column-major in OpenCV / COLMAP style.
    c2w: np.ndarray = field(default_factory=lambda: np.eye(4))
    w2c: np.ndarray = field(default_factory=lambda: np.eye(4))

    def resize(self, width: int, height: int) -> None:
        sx, sy = width / self.width, height / self.height
        self.fx *= sx
        self.fy *= sy
        self.cx *= sx
        self.cy *= sy
        self.width = width
        self.height = height

    def to_metadata(self) -> dict:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": int(self.width),
            "height": int(self.height),
            "c2w": self.c2w.tolist(),
            "w2c": self.w2c.tolist(),
        }