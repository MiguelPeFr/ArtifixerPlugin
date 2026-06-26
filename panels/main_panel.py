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
        self._resolution_w_text: str = "1024"
        self._resolution_h_text: str = "1024"
        self._camera_mode: str = "original"
        self._export_mode: str = ExportMode.TRAINING.value
        self._num_views_text: str = "36"
        self._radius_factor: float = 1.6
        self._status: str = "Idle"
        self._progress: int = 0
        self._busy: bool = False
        self._last_manifest: str = ""
        self._project_camera_count: int = 0

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
        try:
            ui.separator()
            ui.label("Scene")
            _, self._scene_id = ui.input_text_with_hint(
                "Scene ID", "object_001", self._scene_id
            )

            ui.label("Output")
            _, self._output_dir = ui.input_text_with_hint(
                "Output directory", "./artifixer_dataset", self._output_dir
            )

            ui.separator()
            ui.label("Resolution")
            _, self._resolution_w_text = ui.input_text_with_hint(
                "Width", "1024", self._resolution_w_text
            )
            _, self._resolution_h_text = ui.input_text_with_hint(
                "Height", "1024", self._resolution_h_text
            )

            ui.separator()
            ui.label("Cameras")
            self._draw_choice_buttons(
                ui,
                current=self._camera_mode,
                options=[
                    ("original", "Original"),
                    ("orbit", "Orbit"),
                    ("hemisphere", "Hemisphere"),
                    ("multi_ring", "Multi Ring"),
                    ("manual", "Manual"),
                ],
                setter=self._set_camera_mode,
            )
            _, self._num_views_text = ui.input_text_with_hint(
                "Number of views", "36", self._num_views_text
            )
            _, self._radius_factor = ui.slider_float(
                "Radius factor", self._radius_factor, 0.5, 5.0
            )

            ui.separator()
            ui.label("Export preset")
            self._draw_choice_buttons(
                ui,
                current=self._export_mode,
                options=[
                    (ExportMode.PREVIEW.value, "Preview"),
                    (ExportMode.TRAINING.value, "Training"),
                    (ExportMode.RESEARCH.value, "Research"),
                ],
                setter=self._set_export_mode,
            )
            ui.bullet_text("Preview: RGB y cameras.json")
            ui.bullet_text("Training: RGB, opacity y manifest")
            ui.bullet_text("Research: anade depth y normal")

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
        except Exception as exc:  # noqa: BLE001
            lf.log.error(f"ArtiFixer panel draw error: {exc}")
            ui.text_disabled(f"UI error: {exc}")

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
            resolution=(
                self._parse_positive_int(self._resolution_w_text, 1024),
                self._parse_positive_int(self._resolution_h_text, 1024),
            ),
            camera_mode=self._camera_mode,
            mode=ExportMode(self._export_mode),
            sampler=SamplerConfig(
                width=self._parse_positive_int(self._resolution_w_text, 1024),
                height=self._parse_positive_int(self._resolution_h_text, 1024),
                num_views=self._parse_positive_int(self._num_views_text, 36),
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

    def _set_camera_mode(self, value: str) -> None:
        self._camera_mode = value

    def _set_export_mode(self, value: str) -> None:
        self._export_mode = value

    def _draw_choice_buttons(self, ui, current: str, options, setter) -> None:
        for index, (value, label) in enumerate(options):
            style = "primary" if value == current else "secondary"
            if ui.button_styled(label, style):
                setter(value)
            if index < len(options) - 1:
                ui.same_line()

    @staticmethod
    def _parse_positive_int(raw: str, default: int) -> int:
        try:
            value = int(raw.strip())
            return value if value > 0 else default
        except Exception:
            return default


_classes: List[type] = [ArtiFixerPanel]
