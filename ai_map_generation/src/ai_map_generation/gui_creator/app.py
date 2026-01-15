from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..common.config import get_map_config
from ..common.io_formats import load_points_csv, load_points_json, save_created_map_csv, save_created_map_npz, save_latent_npz
from ..common.plotting import build_sample_figure
from ..common.utils import compute_sha256, get_device, seed_everything
from ..fitting.latent_fit import FitConfig, FitResult, fit_latent
from ..gui_common.widgets import MapInspectDialog
from ..training.export import build_model_from_artifact, load_model_artifact
from .widgets import configure_table


class FitWorker(QtCore.QThread):
    progress = QtCore.Signal(int, float)
    message = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(
        self,
        model,
        latent_dim: int,
        axis0: np.ndarray,
        axis1: np.ndarray,
        points: np.ndarray,
        targets: np.ndarray,
        config: FitConfig,
        device,
    ) -> None:
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim
        self.axis0 = axis0
        self.axis1 = axis1
        self.points = points
        self.targets = targets
        self.config = config
        self.device = device
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _is_stop_requested(self) -> bool:
        return self._stop_requested

    def run(self) -> None:
        try:
            self.model.to(self.device)
            self.model.eval()
            result = fit_latent(
                decoder=self.model.decoder,
                latent_dim=self.latent_dim,
                output_shape=self.model.input_shape,
                axis0=self.axis0,
                axis1=self.axis1,
                points=self.points,
                targets=self.targets,
                config=self.config,
                device=self.device,
                progress_callback=self.progress.emit,
                stop_requested=self._is_stop_requested,
            )
            self.message.emit("Optimization finished.")
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class CreatorWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Map Creator")
        self.resize(1000, 750)

        self.model = None
        self.model_path: Path | None = None
        self.map_type: str | None = None
        self.axis0: np.ndarray | None = None
        self.axis1: np.ndarray | None = None
        self.latent_dim: int | None = None
        self.fit_result: FitResult | None = None
        self.last_fit_config: FitConfig | None = None
        self.worker: FitWorker | None = None

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.addWidget(self._build_map_toggle())
        main_layout.addWidget(self._build_model_group())
        main_layout.addWidget(self._build_points_group())
        main_layout.addWidget(self._build_optimizer_group())
        main_layout.addWidget(self._build_run_group())
        main_layout.addWidget(self._build_residuals_group())
        main_layout.addWidget(self._build_export_group())
        main_layout.addWidget(self._build_log_group())
        self.setCentralWidget(central)

    def _build_map_toggle(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Map Type")
        layout = QtWidgets.QHBoxLayout(group)
        self.map_type_combo = QtWidgets.QComboBox()
        self.map_type_combo.addItems(["Compressor", "Turbine"])
        self.map_type_combo.currentIndexChanged.connect(self._update_point_headers)
        layout.addWidget(QtWidgets.QLabel("Mode:"))
        layout.addWidget(self.map_type_combo)
        layout.addStretch()
        return group

    def _build_model_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Load Model")
        layout = QtWidgets.QGridLayout(group)

        self.model_path_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_model)
        load_button = QtWidgets.QPushButton("Load")
        load_button.clicked.connect(self._load_model)

        self.meta_label = QtWidgets.QLabel("Model metadata: none")

        layout.addWidget(QtWidgets.QLabel("Model file"), 0, 0)
        layout.addWidget(self.model_path_edit, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        layout.addWidget(load_button, 0, 3)
        layout.addWidget(self.meta_label, 1, 0, 1, 4)
        return group

    def _build_points_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Operating Points")
        layout = QtWidgets.QVBoxLayout(group)

        button_layout = QtWidgets.QHBoxLayout()
        load_csv_button = QtWidgets.QPushButton("Load CSV")
        load_json_button = QtWidgets.QPushButton("Load JSON")
        add_row_button = QtWidgets.QPushButton("Add Row")
        remove_row_button = QtWidgets.QPushButton("Remove Row")

        load_csv_button.clicked.connect(self._load_points_csv)
        load_json_button.clicked.connect(self._load_points_json)
        add_row_button.clicked.connect(self._add_point_row)
        remove_row_button.clicked.connect(self._remove_point_row)

        button_layout.addWidget(load_csv_button)
        button_layout.addWidget(load_json_button)
        button_layout.addWidget(add_row_button)
        button_layout.addWidget(remove_row_button)
        button_layout.addStretch()

        self.points_table = QtWidgets.QTableWidget(0, 4)
        configure_table(self.points_table, self._current_point_headers())

        layout.addLayout(button_layout)
        layout.addWidget(self.points_table)
        return group

    def _build_optimizer_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Optimization Configuration")
        layout = QtWidgets.QGridLayout(group)

        self.init_combo = QtWidgets.QComboBox()
        self.init_combo.addItems(["zeros", "random"])

        self.optimizer_combo = QtWidgets.QComboBox()
        self.optimizer_combo.addItems(["Adam", "LBFGS"])

        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(300)

        self.learning_rate_spin = QtWidgets.QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(1e-2)

        self.lambda_spin = QtWidgets.QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 10.0)
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setValue(1e-3)

        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(0.0, 1.0)
        self.tolerance_spin.setDecimals(8)
        self.tolerance_spin.setValue(1e-6)

        self.gpu_checkbox = QtWidgets.QCheckBox("Use GPU")
        self.deterministic_checkbox = QtWidgets.QCheckBox("Deterministic")
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(123)

        layout.addWidget(QtWidgets.QLabel("Init"), 0, 0)
        layout.addWidget(self.init_combo, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Optimizer"), 0, 2)
        layout.addWidget(self.optimizer_combo, 0, 3)
        layout.addWidget(QtWidgets.QLabel("Iterations"), 0, 4)
        layout.addWidget(self.iterations_spin, 0, 5)

        layout.addWidget(QtWidgets.QLabel("Learning rate"), 1, 0)
        layout.addWidget(self.learning_rate_spin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Latent L2"), 1, 2)
        layout.addWidget(self.lambda_spin, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Tolerance"), 1, 4)
        layout.addWidget(self.tolerance_spin, 1, 5)

        layout.addWidget(self.gpu_checkbox, 2, 0)
        layout.addWidget(self.deterministic_checkbox, 2, 2)
        layout.addWidget(QtWidgets.QLabel("Seed"), 2, 3)
        layout.addWidget(self.seed_spin, 2, 4)
        return group

    def _build_run_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Run")
        layout = QtWidgets.QHBoxLayout(group)
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.progress_bar = QtWidgets.QProgressBar()

        self.start_button.clicked.connect(self._start_fit)
        self.stop_button.clicked.connect(self._stop_fit)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_bar)
        return group

    def _build_residuals_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Residuals")
        layout = QtWidgets.QVBoxLayout(group)
        self.residuals_table = QtWidgets.QTableWidget(0, 8)
        layout.addWidget(self.residuals_table)
        return group

    def _build_export_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Export")
        layout = QtWidgets.QHBoxLayout(group)
        map_button = QtWidgets.QPushButton("Export Map")
        inspect_button = QtWidgets.QPushButton("Inspect Map")
        latent_button = QtWidgets.QPushButton("Export Latent")
        report_button = QtWidgets.QPushButton("Export Report")

        map_button.clicked.connect(self._export_map)
        inspect_button.clicked.connect(self._inspect_map)
        latent_button.clicked.connect(self._export_latent)
        report_button.clicked.connect(self._export_report)

        layout.addWidget(map_button)
        layout.addWidget(inspect_button)
        layout.addWidget(latent_button)
        layout.addWidget(report_button)
        layout.addStretch()
        return group

    def _build_log_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Status Log")
        layout = QtWidgets.QVBoxLayout(group)
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        return group

    def _log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _current_point_headers(self) -> list[str]:
        map_type = self.map_type_combo.currentText().lower()
        if map_type == "compressor":
            return ["Nc", "mdotc", "eta", "pi"]
        return ["Nc", "pi_t", "eta", "mdotc"]

    def _update_point_headers(self) -> None:
        configure_table(self.points_table, self._current_point_headers())

    def _browse_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Model", filter="PyTorch Model (*.pt)")
        if path:
            self.model_path_edit.setText(path)

    def _load_model(self) -> None:
        path_text = self.model_path_edit.text().strip()
        if not path_text:
            self._log("Select a model file.")
            return
        try:
            artifact = load_model_artifact(path_text)
            self.model = build_model_from_artifact(artifact)
            self.model_path = Path(path_text)
            self.map_type = str(artifact["map_type"]).lower()
            grid = artifact["grid"]
            self.axis0 = np.asarray(grid["axis0"], dtype=np.float32)
            self.axis1 = np.asarray(grid["axis1"], dtype=np.float32)
            self.latent_dim = int(artifact["latent_dim"])

            map_type_text = (self.map_type or "").capitalize()
            self.map_type_combo.setCurrentText(map_type_text)
            self.meta_label.setText(
                f"Map: {self.map_type} | latent_dim: {self.latent_dim} | "
                f"grid: {self.axis0.size}x{self.axis1.size}"
            )
            self._log(f"Loaded model from {path_text}.")
        except Exception as exc:
            self._log(f"Model load error: {exc}")
            self.model = None
            self.model_path = None

    def _load_points_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load CSV", filter="CSV Files (*.csv)")
        if not path:
            return
        try:
            points = load_points_csv(path, self.map_type_combo.currentText().lower())
            self._populate_points(points.inputs, points.targets)
            self._log(f"Loaded points from {path}.")
        except Exception as exc:
            self._log(f"CSV load error: {exc}")

    def _load_points_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load JSON", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            points = load_points_json(path, self.map_type_combo.currentText().lower())
            self._populate_points(points.inputs, points.targets)
            self._log(f"Loaded points from {path}.")
        except Exception as exc:
            self._log(f"JSON load error: {exc}")

    def _populate_points(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.points_table.setRowCount(0)
        for row_index in range(inputs.shape[0]):
            self.points_table.insertRow(row_index)
            values = [inputs[row_index, 0], inputs[row_index, 1], targets[row_index, 0], targets[row_index, 1]]
            for col_index, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(f"{float(value):.6g}")
                self.points_table.setItem(row_index, col_index, item)

    def _add_point_row(self) -> None:
        row = self.points_table.rowCount()
        self.points_table.insertRow(row)

    def _remove_point_row(self) -> None:
        selected = self.points_table.currentRow()
        if selected >= 0:
            self.points_table.removeRow(selected)

    def _gather_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        rows = self.points_table.rowCount()
        if rows == 0:
            self._log("Add at least one operating point.")
            return None

        values = []
        for row_index in range(rows):
            row_values = []
            for col_index in range(4):
                item = self.points_table.item(row_index, col_index)
                if item is None or not item.text().strip():
                    self._log("All point fields must be filled.")
                    return None
                try:
                    row_values.append(float(item.text()))
                except ValueError:
                    self._log("Point values must be numeric.")
                    return None
            values.append(row_values)

        data = np.asarray(values, dtype=np.float32)
        return data[:, :2], data[:, 2:]

    def _start_fit(self) -> None:
        if self.worker:
            self._log("Optimization already running.")
            return
        if not self.model or self.axis0 is None or self.axis1 is None or self.latent_dim is None:
            self._log("Load a model before running optimization.")
            return
        if self.map_type and self.map_type != self.map_type_combo.currentText().lower():
            self._log("Map type toggle does not match loaded model.")
            return

        gathered = self._gather_points()
        if gathered is None:
            return
        inputs, targets = gathered

        seed = self.seed_spin.value()
        if self.deterministic_checkbox.isChecked():
            seed_everything(seed, deterministic=True)

        device = get_device(self.gpu_checkbox.isChecked())
        if device.type == "cpu" and self.gpu_checkbox.isChecked():
            self._log("GPU requested but not available. Using CPU.")

        config = FitConfig(
            init_strategy=self.init_combo.currentText(),
            iterations=self.iterations_spin.value(),
            learning_rate=float(self.learning_rate_spin.value()),
            latent_l2_weight=float(self.lambda_spin.value()),
            optimizer=self.optimizer_combo.currentText(),
            tolerance=float(self.tolerance_spin.value()),
        )
        self.last_fit_config = config

        self.progress_bar.setMaximum(config.iterations)
        self.progress_bar.setValue(0)

        self.worker = FitWorker(
            model=self.model,
            latent_dim=self.latent_dim,
            axis0=self.axis0,
            axis1=self.axis1,
            points=inputs,
            targets=targets,
            config=config,
            device=device,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def _stop_fit(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self._log("Stop requested...")

    def _on_progress(self, iteration: int, loss: float) -> None:
        self.progress_bar.setValue(iteration)
        self._log(f"Iter {iteration}: loss={loss:.6f}")

    def _on_error(self, message: str) -> None:
        self._log(f"Error: {message}")
        self._cleanup_worker()

    def _on_finished(self, result: FitResult) -> None:
        self.fit_result = result
        self._log(f"Optimization finished. Final loss: {result.final_loss:.6f}")
        self._update_residuals_table(result)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        self.worker = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait()
        event.accept()

    def _update_residuals_table(self, result: FitResult) -> None:
        config = get_map_config(self.map_type_combo.currentText().lower())
        headers = [
            config.axis0_name,
            config.axis1_name,
            f"target_{config.channels[0]}",
            f"target_{config.channels[1]}",
            f"pred_{config.channels[0]}",
            f"pred_{config.channels[1]}",
            f"res_{config.channels[0]}",
            f"res_{config.channels[1]}",
        ]
        configure_table(self.residuals_table, headers)
        self.residuals_table.setRowCount(result.inputs.shape[0])

        residuals = result.predicted - result.targets
        for row_index in range(result.inputs.shape[0]):
            row_values = [
                result.inputs[row_index, 0],
                result.inputs[row_index, 1],
                result.targets[row_index, 0],
                result.targets[row_index, 1],
                result.predicted[row_index, 0],
                result.predicted[row_index, 1],
                residuals[row_index, 0],
                residuals[row_index, 1],
            ]
            for col_index, value in enumerate(row_values):
                item = QtWidgets.QTableWidgetItem(f"{float(value):.6g}")
                self.residuals_table.setItem(row_index, col_index, item)

    def _build_fit_meta(self) -> dict:
        if not self.fit_result:
            return {}
        fit_meta = {
            "final_loss": self.fit_result.final_loss,
            "iterations": len(self.fit_result.loss_history),
        }
        if self.last_fit_config:
            fit_meta.update(
                {
                    "optimizer": self.last_fit_config.optimizer,
                    "learning_rate": self.last_fit_config.learning_rate,
                    "latent_l2": self.last_fit_config.latent_l2_weight,
                    "tolerance": self.last_fit_config.tolerance,
                    "init_strategy": self.last_fit_config.init_strategy,
                }
            )
        return fit_meta

    def _inspect_map(self) -> None:
        if not self.fit_result or self.axis0 is None or self.axis1 is None:
            self._log("Run optimization before inspecting the map.")
            return
        try:
            map_type = self.map_type_combo.currentText().lower()
            figure = build_sample_figure(self.fit_result.reconstructed, self.axis0, self.axis1, map_type)
            dialog = MapInspectDialog(figure, self)
            dialog.exec()
        except Exception as exc:
            self._log(f"Inspect error: {exc}")

    def _export_map(self) -> None:
        if not self.fit_result or self.axis0 is None or self.axis1 is None:
            self._log("Run optimization before exporting map.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Map", filter="NPZ (*.npz)")
        if not path:
            return
        try:
            config = get_map_config(self.map_type_combo.currentText().lower())
            save_created_map_npz(
                path,
                tensor=self.fit_result.reconstructed,
                axis0=self.axis0,
                axis1=self.axis1,
                channels=config.channels,
                map_type=config.map_type,
                latent_z=self.fit_result.latent_z,
                fit_meta=self._build_fit_meta(),
            )
            csv_path = Path(path).with_suffix(".csv")
            save_created_map_csv(csv_path, self.fit_result.reconstructed, self.axis0, self.axis1, config.map_type)
            self._log(f"Exported map to {path} and {csv_path}.")
        except Exception as exc:
            self._log(f"Export error: {exc}")

    def _export_latent(self) -> None:
        if not self.fit_result or not self.model_path or self.latent_dim is None:
            self._log("Run optimization before exporting latent.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Latent", filter="NPZ (*.npz)")
        if not path:
            return
        try:
            model_hash = compute_sha256(self.model_path)
            save_latent_npz(
                path,
                latent_z=self.fit_result.latent_z,
                map_type=self.map_type_combo.currentText().lower(),
                latent_dim=self.latent_dim,
                model_hash=model_hash,
                fit_meta=self._build_fit_meta(),
            )
            self._log(f"Exported latent to {path}.")
        except Exception as exc:
            self._log(f"Latent export error: {exc}")

    def _export_report(self) -> None:
        if not self.fit_result or not self.model_path:
            self._log("Run optimization before exporting report.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Report", filter="JSON (*.json)")
        if not path:
            return
        try:
            report = {
                "model_path": str(self.model_path),
                "map_type": self.map_type_combo.currentText().lower(),
                "latent_dim": self.latent_dim,
                "fit_meta": self._build_fit_meta(),
            }
            with open(path, "w", encoding="utf-8") as file_handle:
                json.dump(report, file_handle, indent=2)
            self._log(f"Exported report to {path}.")
        except Exception as exc:
            self._log(f"Report export error: {exc}")
