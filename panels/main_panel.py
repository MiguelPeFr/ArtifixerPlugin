"""ArtiFixer Export panel for LichtFeld Studio."""

from __future__ import annotations

import logging
import threading
from typing import List

import lichtfeld as lf

from services.camera_sampler import SamplerConfig
from ui.export_panel import ExportMode, ExportSettings

log = logging.getLogger("artifixer_export.panel")


# --------------------------------------------------------------------------- #
class ArtiFixerPanel(lf.ui.Panel):
    """Main export panel rendered as a tab inside LichtFeld."""

    id = "artifixer_export.main_panel"
    label = "ArtiFixer Export"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 250

    def __init__(self) -> None:
        self._scene_id: str = ""
        self._output_dir: str = "./artifixer_dataset"
        self._resolution_w: int = 1024
        self._resolution_h: int = 1024
        self._camera_mode: str = "orbit"
        self._export_mode: str = ExportMode.TRAINING.value
        self._num_views: int = 36
        self._radius_factor: float = 1.6
        self._status: str = "Idle"
        self._progress: int = 0
        self._busy: bool = False
        self._last_manifest: str = ""

    # ---- Imperative API used by the runner thread -------------------------
    def set_status(self, msg: str) -> None:
        self._status = msg
        lf.ui.request_redraw()

    def set_progress(self, current: int, total: int, msg: str = "") -> None:
        self._progress = int(100 * current / max(total, 1))
        if msg:
            self._status = msg
        lf.ui.request_redraw()

    @classmethod
    def poll(cls, context):
        try:
            return lf.scene.is_loaded()
        except Exception:
            return True

    def draw(self, ui) -> None:
        ui.heading("ArtiFixer Export")
        ui.text_disabled(
            "Exporta la escena activa de LichtFeld a un dataset para NVIDIA ArtiFixer."
        )

        ui.separator()
        ui.text("Scene")
        _, self._scene_id = ui.input_text_with_hint(
            "Scene ID", "object_001", self._scene_id
        )

        ui.text("Output")
        _, self._output_dir = ui.input_text_with_hint(
            "Output directory", "./artifixer_dataset", self._output_dir
        )

        ui.separator()
        ui.text("Resolution")
        _, self._resolution_w = ui.input_int("Width", self._resolution_w)
        _, self._resolution_h = ui.input_int("Height", self._resolution_h)

        ui.separator()
        ui.text("Cameras")
        _, self._camera_mode = ui.combo(
            "Camera mode",
            self._camera_mode,
            ["original", "orbit", "hemisphere", "multi_ring", "manual"],
        )
        _, self._num_views = ui.input_int("Number of views", self._num_views)
        _, self._radius_factor = ui.slider_float(
            "Radius factor", self._radius_factor, 0.5, 5.0
        )

        ui.separator()
        ui.text("Export preset")
        _, self._export_mode = ui.combo(
            "Mode",
            self._export_mode,
            [ExportMode.PREVIEW.value, ExportMode.TRAINING.value, ExportMode.RESEARCH.value],
        )

        ui.separator()
        if self._busy:
            ui.text_disabled(f"Running... {self._progress}%  {self._status}")
        else:
            if ui.button_styled("Export ArtiFixer Dataset", "primary"):
                self._launch_export()

        if self._progress:
            ui.progress_bar(self._progress / 100.0, f"{self._progress}%")
        if self._status:
            ui.text_disabled(f"Status: {self._status}")
        if self._last_manifest:
            ui.text_disabled(f"Last manifest: {self._last_manifest}")

    def _launch_export(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._progress = 0
        self._status = "Starting export..."
        lf.ui.request_redraw()

        settings = ExportSettings(
            scene_id=self._scene_id,
            output_dir=self._output_dir,
            resolution=(int(self._resolution_w), int(self._resolution_h)),
            camera_mode=self._camera_mode,
            mode=ExportMode(self._export_mode),
            sampler=SamplerConfig(
                width=int(self._resolution_w),
                height=int(self._resolution_h),
                num_views=int(self._num_views),
                radius_factor=float(self._radius_factor),
            ),
        )

        thread = threading.Thread(
            target=self._run_export_worker,
            args=(settings,),
            daemon=True,
        )
        thread.start()

    def _run_export_worker(self, settings: ExportSettings) -> None:
        try:
            from plugin import ArtiFixerExportPlugin

            plugin = ArtiFixerExportPlugin()
            plugin.on_load(app=None)

            settings.sampler.width = settings.resolution[0]
            settings.sampler.height = settings.resolution[1]

            plugin.panel.set_progress = self.set_progress  # type: ignore[attr-defined]

            manifest = plugin._run_export(settings)
            self._last_manifest = str(manifest)
            self._status = f"Done -> {manifest}"
        except Exception as exc:  # noqa: BLE001
            log.exception("ArtiFixer export failed")
            self._status = f"ERROR: {exc}"
        finally:
            self._progress = 100 if self._status.startswith("Done") else self._progress
            self._busy = False
            lf.ui.request_redraw()


_classes: List[type] = [ArtiFixerPanel]
