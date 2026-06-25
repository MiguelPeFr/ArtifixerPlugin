"""Punto de entrada del plugin para LichtFeld.

La documentacion oficial exige que el directorio del plugin exponga
``on_load()`` y ``on_unload()`` desde ``__init__.py``.
"""

from __future__ import annotations

import logging

from plugin import ArtiFixerExportPlugin

log = logging.getLogger("artifixer_export")

_plugin_instance: ArtiFixerExportPlugin | None = None


def on_load() -> None:
    """Entry point compatible con el cargador v1 de LichtFeld."""
    global _plugin_instance
    _plugin_instance = ArtiFixerExportPlugin()
    _plugin_instance.on_load(app=None)
    log.info("ArtiFixer Export plugin loaded from __init__.py")


def on_unload() -> None:
    """Descarga el plugin si fue inicializado por LichtFeld."""
    global _plugin_instance
    if _plugin_instance is not None:
        _plugin_instance.on_unload()
        _plugin_instance = None
    log.info("ArtiFixer Export plugin unloaded from __init__.py")
