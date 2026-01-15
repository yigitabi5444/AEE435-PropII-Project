from __future__ import annotations

from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class SampleInspectDialog(QtWidgets.QDialog):
    def __init__(self, figure, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sample Inspector")
        self.resize(800, 400)

        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self.setLayout(layout)
