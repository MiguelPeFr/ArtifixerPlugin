"""ArtiFixer Export panel for LichtFeld Studio.

Based on the 360_record plugin UI conventions:

- ``lf.ui.Panel`` subclass with ``id``, ``label``, ``space``, ``order``.
- ``draw(self, ui)`` builds the UI using the immediate-mode helpers.
- Heavy work is dispatched to a background thread and reflected through
  ``lf.ui.request_redraw()`` so the UI stays responsive.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Type

import lichtfeld as lf

from plugin import ArtiFixerExportPlugin
from services.camera_sampler import SamplerConfig
from ui.export_panel import ExportMode, ExportSettings

log = logging.getLogger("artifixer_export.panel")


# --------------------------------------------------------------------------- #
# Lazy base class resolution
# --------------------------------------------------------------------------- #
def _resolve_panel_base() -> Type[Any]:
    """Return ``lf.ui.Panel`` from the host, or ``object`` as a fallback.

    The fallback only matters in environments where LichtFeld is not
    installed (CI, tests). Inside LichtFeld the real base class is used.
    """
    try:
        import lichtfeld as _lf  # type: ignore
        return _lf.ui.Panel  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return object


PanelBase = _resolve_panel_base()


# --------------------------------------------------------------------------- #
class ArtiFixerPanel(PanelBase):
    """Main export panel rendered as a tab inside LichtFeld."""

    id = "artifixer.export_panel"
    label = "ArtiFixer Export"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 250

    # ---- UI state (kept on self across draw() calls) --------------------
    def __init__(self) -> None:
        super().__init__()
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

    # ---- Visibility predicate ---------------------------------------------
    @classmethod
    def poll(cls, context: Any) -> bool:  # noqa: D401
        """Show the panel only when a scene is loaded."""
        try:
            return lf.scene.is_loaded()
        except Exception:  # noqa: BLE001 - host API may not be available
            return True

    # ---- draw(ui) ---------------------------------------------------------
    def draw(self, ui: Any) -> None:
        ui.heading("ArtiFixer Dataset Export")
        ui.text_disabled(
            "Export the active LichtFeld scene to a dataset ready for "
            "NVIDIA ArtiFixer (RGB, opacity, depth, ray maps, manifest)."
        )

        # ---- Scene + output ------------------------------------------------
        ui.separator()
        ui.text("Scene")
        self._scene_id = ui.input_text("Scene ID", self._scene_id)

        ui.text("Output")
        self._output_dir = ui.input_text("Output directory", self._output_dir)

        # ---- Resolution ----------------------------------------------------
        ui.separator()
        ui.text("Resolution")
        self._resolution_w = ui.input_int("Width",  self._resolution_w, min=64,  max=8192)
        self._resolution_h = ui.input_int("Height", self._resolution_h, min=64,  max=8192)

        # ---- Cameras -------------------------------------------------------
        ui.separator()
        ui.text("Cameras")
        self._camera_mode = ui.combo(
            "Camera mode",
            self._camera_mode,
            ["original", "orbit", "hemisphere", "multi_ring", "manual"],
        )
        self._num_views = ui.input_int("Number of views", self._num_views, min=1, max=512)
        self._radius_factor = ui.slider_float(
            "Radius factor", self._radius_factor, 0.5, 5.0
        )

        # ---- Export preset -------------------------------------------------
        ui.separator()
        ui.text("Export preset")
        self._export_mode = ui.combo(
            "Mode",
            self._export_mode,
            [ExportMode.PREVIEW.value, ExportMode.TRAINING.value, ExportMode.RESEARCH.value],
        )

        # ---- Action button -------------------------------------------------
        ui.separator()
        if self._busy:
            ui.text_disabled(f"Running... {self._progress}%  {self._status}")
        else:
            if ui.button_styled("Export ArtiFixer Dataset", "primary"):
                self._launch_export()

        if self._progress:
            ui.progress_bar(self._progress, 0, 100)
        if self._status:
            ui.text_disabled(f"Status: {self._status}")
        if self._last_manifest:
            ui.text_disabled(f"Last manifest: {self._last_manifest}")

    # ---- Export dispatch --------------------------------------------------
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
            plugin = ArtiFixerExportPlugin()
            plugin.on_load(app=None)

            settings.sampler.width = settings.resolution[0]
            settings.sampler.height = settings.resolution[1]

            # Re-bind the progress callback so the worker updates the panel.
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


_classes: List[Type[Any]] = [ArtiFixerPanel]