from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_map_generation.gui_creator.app import CreatorWindow
from ai_map_generation.gui_trainer.app import TrainerWindow


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Map Generation Launcher")
        self.resize(360, 160)
        self.child_windows: list[QtWidgets.QWidget] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Choose an application:"))

        trainer_button = QtWidgets.QPushButton("Open Trainer")
        creator_button = QtWidgets.QPushButton("Open Creator")

        trainer_button.clicked.connect(self._open_trainer)
        creator_button.clicked.connect(self._open_creator)

        layout.addWidget(trainer_button)
        layout.addWidget(creator_button)

    def _open_trainer(self) -> None:
        window = TrainerWindow()
        window.show()
        self.child_windows.append(window)

    def _open_creator(self) -> None:
        window = CreatorWindow()
        window.show()
        self.child_windows.append(window)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
