"""LichtFeld export panel.

Lightweight, dependency-free UI. In production, LichtFeld expects a Qt
widget, but here we expose a small abstraction so the panel can be driven
from a CLI, a Tk window, or a Qt host with the same callback signature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from services.camera_sampler import SamplerConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
class ExportMode(str, Enum):
    PREVIEW = "preview"
    TRAINING = "training"
    RESEARCH = "research"


@dataclass
class ExportSettings:
    scene_id: str = ""
    output_dir: str = "./artifixer_dataset"
    resolution: tuple = (1024, 1024)
    camera_mode: str = "original"      # original / orbit / hemisphere / multi_ring / manual
    mode: ExportMode = ExportMode.TRAINING
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    coordinate_system: str = "opencv"
    force_synthetic: bool = False      # set True to ignore project cameras

    def to_json(self) -> str:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["resolution"] = list(self.resolution)
        return json.dumps(d, indent=2, default=str)


# --------------------------------------------------------------------------- #
class ExportPanel:
    """UI abstraction.

    The real LichtFeld panel would subclass a Qt widget. We expose a minimal
    surface so the same panel can be exercised from tests.
    """

    def __init__(self, run_export: Callable[[ExportSettings], Path]):
        self._run_export = run_export
        self._widget = None
        self._progress = 0
        self._status = "Idle"

    # ---- Lifecycle ---------------------------------------------------------
    def show(self) -> None:
        log.info("Export panel shown")

    def close(self) -> None:
        log.info("Export panel closed")

    # ---- Settings -----------------------------------------------------------
    def collect_settings(self) -> ExportSettings:
        """Read current settings from the widget.

        The standalone CLI fallback below is used when no Qt parent is set.
        """
        if self._widget is None:
            return self._cli_settings()
        return self._widget.collect()

    # ---- Progress -----------------------------------------------------------
    def set_progress(self, current: int, total: int, msg: str = "") -> None:
        pct = int(100 * current / max(total, 1))
        self._progress = pct
        self._status = msg or self._status
        log.info("[%s%%] %s", pct, self._status)

    # ---- Fallback -----------------------------------------------------------
    def _cli_settings(self) -> ExportSettings:
        out = input("Output dir [./artifixer_dataset]: ").strip() or "./artifixer_dataset"
        mode = input("Camera mode (original/orbit/hemisphere/multi_ring/manual) [orbit]: ").strip() or "orbit"
        preset = input("Export mode (preview/training/research) [training]: ").strip() or "training"
        try:
            res = input("Resolution WxH [1024x1024]: ").strip() or "1024x1024"
            w, h = res.lower().split("x")
            resolution = (int(w), int(h))
        except ValueError:
            resolution = (1024, 1024)

        cfg = SamplerConfig(width=resolution[0], height=resolution[1])
        return ExportSettings(
            output_dir=out,
            resolution=resolution,
            camera_mode=mode,
            mode=ExportMode(preset),
            sampler=cfg,
        )


# --------------------------------------------------------------------------- #
# Convenience: programmatic run
# --------------------------------------------------------------------------- #
def run_headless(
    run_export: Callable[[ExportSettings], Path],
    output_dir: str,
    camera_mode: str = "orbit",
    mode: str = "training",
    resolution: tuple = (1024, 1024),
    num_views: int = 36,
) -> Path:
    cfg = SamplerConfig(
        width=resolution[0], height=resolution[1], num_views=num_views
    )
    settings = ExportSettings(
        output_dir=output_dir,
        resolution=resolution,
        camera_mode=camera_mode,
        mode=ExportMode(mode),
        sampler=cfg,
    )
    return run_export(settings)