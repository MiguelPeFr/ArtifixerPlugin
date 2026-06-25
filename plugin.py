"""LichtFeld Studio plugin: ArtiFixer Dataset Export.

Exports an active LichtFeld scene (3D Gaussians / PLY) as a dataset
compatible with NVIDIA ArtiFixer:

  output/
  ├── rgb/        PNGs
  ├── opacity/    PNGs (alpha / accumulated opacity)
  ├── depth/      EXR or PNG16 (optional)
  ├── cameras.json
  ├── manifest.json
  └── rays/       NPY (postprocess)

Modes:
  - preview : RGB + cameras.json
  - training: RGB + opacity + manifest (default for ArtiFixer)
  - research: + depth / normal
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from services.dataset_writer import DatasetWriter
from services.render_service import RenderService
from services.scene_adapter import SceneAdapter
from ui.export_panel import ExportPanel, ExportSettings

log = logging.getLogger("artifixer_export")


# --------------------------------------------------------------------------- #
# Settings dataclass
# --------------------------------------------------------------------------- #
@dataclass
class ExportContext:
    settings: ExportSettings
    output_dir: Path
    scene_name: str = "scene"
    progress_cb: Optional[callable] = field(default=None, repr=False)

    def report(self, current: int, total: int, msg: str = "") -> None:
        if self.progress_cb:
            try:
                self.progress_cb(current, total, msg)
            except Exception:  # noqa: BLE001
                log.exception("progress callback failed")
        log.info("[%s/%s] %s", current, total, msg)


# --------------------------------------------------------------------------- #
# Main plugin class
# --------------------------------------------------------------------------- #
class ArtiFixerExportPlugin:
    """LichtFeld Studio plugin entry point."""

    OPERATOR_ID = "export_artifixer_dataset"

    def __init__(self) -> None:
        self.app = None
        self.panel: Optional[ExportPanel] = None
        self.adapter = SceneAdapter()
        self.renderer = RenderService()
        self.writer = DatasetWriter()

    # -- Lifecycle ------------------------------------------------------------
    def on_load(self, app) -> None:
        """Called by LichtFeld when the plugin is loaded."""
        self.app = app
        self.register_panel()
        self.register_operator()
        log.info("ArtiFixer Export plugin loaded")

    def on_unload(self) -> None:
        if self.panel is not None:
            self.panel.close()
            self.panel = None
        log.info("ArtiFixer Export plugin unloaded")

    # -- Registration ---------------------------------------------------------
    def register_panel(self) -> None:
        # In real LichtFeld:
        #     self.app.ui.register_panel(
        #         id="artifixer.export",
        #         title="ArtiFixer Export",
        #         widget=ExportPanel(self._run_export),
        #     )
        self.panel = ExportPanel(self._run_export)
        log.info("Panel registered")

    def register_operator(self) -> None:
        # In real LichtFeld:
        #     self.app.operators.register(
        #         id=self.OPERATOR_ID,
        #         label="Export ArtiFixer Dataset",
        #         handler=self._run_export,
        #     )
        log.info("Operator registered: %s", self.OPERATOR_ID)

    # -- Public operators -----------------------------------------------------
    def export_active_cameras_json(self, output_path: Path) -> Path:
        """Hello-plugin style: dump cameras.json only."""
        scene = self.adapter.get_active_scene(self.app)
        cameras = self.adapter.collect_cameras(scene)
        return self.writer.write_cameras_json(cameras, output_path)

    # -- Main export ----------------------------------------------------------
    def _run_export(self, settings: ExportSettings) -> Path:
        """Synchronous entry point invoked from the UI button."""
        out_dir = Path(settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = ExportContext(
            settings=settings,
            output_dir=out_dir,
            scene_name=settings.scene_id or "scene",
            progress_cb=(self.panel.set_progress if self.panel else None),
        )

        scene = self.adapter.get_active_scene(self.app)
        ctx.report(0, 100, f"Scene loaded: {getattr(scene, 'name', '?')}")

        cameras = self.adapter.collect_cameras(
            scene,
            mode=settings.camera_mode,
            sampler_cfg=settings.sampler,
        )
        if not cameras:
            raise RuntimeError("No cameras available for export")

        width, height = settings.resolution
        for cam in cameras:
            cam.resize(width, height)

        ctx.report(5, 100, f"{len(cameras)} camera(s) prepared @ {width}x{height}")

        manifest = self.writer.init_manifest(
            scene_id=ctx.scene_name,
            source="LichtFeld Studio",
            ply_path=getattr(scene, "ply_path", None),
            width=width,
            height=height,
        )

        n = len(cameras)
        for i, cam in enumerate(cameras, start=1):
            pct = 5 + int(80 * i / n)
            ctx.report(pct, 100, f"Rendering frame {i}/{n}")

            rgb = self.renderer.render_rgb(scene, cam)
            opacity = self.renderer.render_opacity(scene, cam)
            depth = (
                self.renderer.render_depth(scene, cam)
                if settings.mode.value in ("training", "research")
                else None
            )
            normal = (
                self.renderer.render_normal(scene, cam)
                if settings.mode.value == "research"
                else None
            )

            frame_id = f"frame_{i:04d}"
            self.writer.write_frame(
                manifest=manifest,
                frame_id=frame_id,
                rgb=rgb,
                opacity=opacity,
                depth=depth,
                normal=normal,
                camera=cam,
                out_dir=out_dir,
                mode=settings.mode.value,
            )

        manifest_path = self.writer.write_manifest(out_dir, manifest)
        self._write_debug_trace(out_dir, scene, cameras)
        ctx.report(100, 100, f"Done -> {manifest_path}")
        log.info("ArtiFixer dataset exported to %s", out_dir)
        return manifest_path

    def _write_debug_trace(self, out_dir: Path, scene, cameras) -> None:
        payload = {
            "scene_name": getattr(scene, "name", None),
            "scene_type": type(getattr(scene, "raw", None)).__name__ if getattr(scene, "raw", None) is not None else None,
            "scene_has_raw": getattr(scene, "raw", None) is not None,
            "camera_count": len(cameras),
            "first_camera": cameras[0].to_metadata() | {"name": cameras[0].name} if cameras else None,
            "render_events": list(self.renderer.debug_events),
        }
        try:
            (out_dir / "debug_trace.json").write_text(json.dumps(payload, indent=2))
        except Exception:  # noqa: BLE001
            log.exception("failed to write debug trace")


# --------------------------------------------------------------------------- #
# Entry point used by LichtFeld
# --------------------------------------------------------------------------- #
_plugin_instance: Optional[ArtiFixerExportPlugin] = None


def on_load(app) -> ArtiFixerExportPlugin:
    global _plugin_instance
    _plugin_instance = ArtiFixerExportPlugin()
    _plugin_instance.on_load(app)
    return _plugin_instance


def on_unload() -> None:
    global _plugin_instance
    if _plugin_instance is not None:
        _plugin_instance.on_unload()
        _plugin_instance = None


if __name__ == "__main__":
    # Standalone execution (for tests / CLI). Uses the demo harness
    # in tests/demo_export.py if available.
    from tests.demo_export import run_demo

    run_demo(plugin_factory=ArtiFixerExportPlugin)
