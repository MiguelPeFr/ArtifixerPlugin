"""Punto de entrada del plugin para LichtFeld.

LichtFeld descubre el plugin por ``pyproject.toml`` y ejecuta ``on_load()``
desde ``__init__.py``. Aqui registramos las clases ``lf.ui.Panel`` que el
host debe mostrar como pestañas en la UI.
"""

from __future__ import annotations

import logging
from typing import List, Type

try:
    import lichtfeld as lf  # type: ignore
except Exception:  # pragma: no cover - el host solo esta presente dentro de LichtFeld
    lf = None  # type: ignore[assignment]

from panels.main_panel import _classes as _panel_classes

log = logging.getLogger("artifixer_export")


def _safe_register(cls: Type) -> None:
    if lf is None:
        log.warning("lichtfeld module not available; skipping registration of %s", cls)
        return
    try:
        lf.register_class(cls)
    except Exception:  # noqa: BLE001
        log.exception("Failed to register %s", cls)


def _safe_unregister(cls: Type) -> None:
    if lf is None:
        return
    try:
        lf.unregister_class(cls)
    except Exception:  # noqa: BLE001
        log.exception("Failed to unregister %s", cls)


def on_load() -> None:
    """Llamado por LichtFeld cuando el plugin se activa."""
    log.info("ArtiFixer Export plugin loading")
    for cls in _panel_classes:
        _safe_register(cls)
    log.info("ArtiFixer Export plugin loaded (%d classes)", len(_panel_classes))


def on_unload() -> None:
    """Llamado por LichtFeld cuando el plugin se descarga."""
    log.info("ArtiFixer Export plugin unloading")
    for cls in reversed(_panel_classes):
        _safe_unregister(cls)
    log.info("ArtiFixer Export plugin unloaded")