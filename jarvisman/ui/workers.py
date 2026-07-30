

from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(str)
    token = pyqtSignal(str)


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = dict(kwargs)
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            params = inspect.signature(self.fn).parameters
            if "progress_callback" in params:
                self.kwargs["progress_callback"] = self.signals.progress.emit
            if "token_callback" in params:
                self.kwargs["token_callback"] = self.signals.token.emit
            output = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - surface everything to the UI
            self.signals.error.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        else:
            self.signals.result.emit(output)
        finally:
            self.signals.finished.emit()