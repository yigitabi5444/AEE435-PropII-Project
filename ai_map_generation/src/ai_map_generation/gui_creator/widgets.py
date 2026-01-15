from __future__ import annotations

from PySide6 import QtWidgets


def configure_table(table: QtWidgets.QTableWidget, headers: list[str]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
