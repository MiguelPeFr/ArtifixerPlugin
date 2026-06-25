"""Punto de entrada del plugin para LichtFeld.

Replica el patron de 360_record: importa ``lichtfeld`` y registra las
clases declaradas en ``panels/`` con ``lf.register_class``.
"""

from __future__ import annotations

import logging

import lichtfeld as lf

from panels.main_panel import _classes as _panel_classes

log = logging.getLogger("artifixer_export")

_classes = list(_panel_classes)


def on_load() -> None:
    """Llamado por LichtFeld cuando el plugin se activa."""
    log.info("ArtiFixer Export plugin loading")
    for cls in _classes:
        lf.register_class(cls)
    log.info("ArtiFixer Export plugin loaded (%d classes)", len(_classes))


def on_unload() -> None:
    """Llamado por LichtFeld cuando el plugin se descarga."""
    log.info("ArtiFixer Export plugin unloading")
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    log.info("ArtiFixer Export plugin unloaded")