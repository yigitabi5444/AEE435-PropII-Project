from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets
from torch.utils.data import DataLoader

from ..common.plotting import build_sample_figure, save_loss_curves
from ..common.utils import get_device, parse_int_list, seed_everything
from ..models.factory import build_model_arch
from ..models.ae_mlp import MLPAutoencoder
from ..training.dataset import MapDataset
from ..training.export import build_model_artifact, save_model_artifact
from ..training.trainer import TrainingConfig, train_autoencoder
from .widgets import SampleInspectDialog


class TrainWorker(QtCore.QThread):
    progress = QtCore.Signal(int, float)
    message = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal(list)

    def __init__(
        self,
        dataset: MapDataset,
        model: MLPAutoencoder,
        config: TrainingConfig,
        device,
        model_path: Path,
        model_arch: dict,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.config = config
        self.device = device
        self.model_path = model_path
        self.model_arch = model_arch
        self.seed = seed
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _is_stop_requested(self) -> bool:
        return self._stop_requested

    def run(self) -> None:
        try:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
            )
            self.message.emit("Training started.")
            losses = train_autoencoder(
                self.model,
                dataloader,
                self.config,
                self.device,
                progress_callback=self.progress.emit,
                stop_requested=self._is_stop_requested,
            )
            if not losses:
                raise RuntimeError("Training produced no losses (was it stopped?).")

            final_loss = float(losses[-1])
            training_meta = {
                "epochs": len(losses),
                "batch_size": self.config.batch_size,
                "lr": self.config.learning_rate,
                "final_loss": final_loss,
                "seed": self.seed,
            }
            artifact = build_model_artifact(
                model=self.model,
                map_type=self.dataset.map_type,
                latent_dim=self.model.encoder[-1].out_features,
                axis0=self.dataset.axis0,
                axis1=self.dataset.axis1,
                model_arch=self.model_arch,
                training_meta=training_meta,
            )
            save_model_artifact(self.model_path, artifact)

            self.message.emit("Training completed and model saved.")
            self.finished.emit(losses)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Map Trainer")
        self.resize(900, 700)

        self.dataset: MapDataset | None = None
        self.worker: TrainWorker | None = None
        self.last_model_path: Path | None = None

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)

        main_layout.addWidget(self._build_map_toggle())
        main_layout.addWidget(self._build_dataset_group())
        main_layout.addWidget(self._build_model_group())
        main_layout.addWidget(self._build_train_group())
        main_layout.addWidget(self._build_output_group())
        main_layout.addWidget(self._build_log_group())

        self.setCentralWidget(central)

    def _build_map_toggle(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Map Type")
        layout = QtWidgets.QHBoxLayout(group)
        self.map_type_combo = QtWidgets.QComboBox()
        self.map_type_combo.addItems(["Compressor", "Turbine"])
        layout.addWidget(QtWidgets.QLabel("Mode:"))
        layout.addWidget(self.map_type_combo)
        layout.addStretch()
        return group

    def _build_dataset_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Dataset Selection")
        layout = QtWidgets.QGridLayout(group)

        self.dataset_path_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_dataset)

        load_button = QtWidgets.QPushButton("Load Dataset")
        load_button.clicked.connect(self._load_dataset)

        self.sample_count_label = QtWidgets.QLabel("Samples: 0")
        self.sample_combo = QtWidgets.QComboBox()
        self.sample_combo.setEnabled(False)
        inspect_button = QtWidgets.QPushButton("Inspect Sample")
        inspect_button.clicked.connect(self._inspect_sample)

        layout.addWidget(QtWidgets.QLabel("Folder:"), 0, 0)
        layout.addWidget(self.dataset_path_edit, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        layout.addWidget(load_button, 1, 1)
        layout.addWidget(self.sample_count_label, 1, 0)
        layout.addWidget(QtWidgets.QLabel("Sample"), 2, 0)
        layout.addWidget(self.sample_combo, 2, 1)
        layout.addWidget(inspect_button, 2, 2)
        return group

    def _build_model_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Model Configuration")
        layout = QtWidgets.QGridLayout(group)

        self.latent_dim_spin = QtWidgets.QSpinBox()
        self.latent_dim_spin.setRange(2, 512)
        self.latent_dim_spin.setValue(12)

        self.hidden_layers_edit = QtWidgets.QLineEdit("512,256")

        self.activation_combo = QtWidgets.QComboBox()
        self.activation_combo.addItems(["relu", "tanh", "gelu", "leaky_relu"])

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(200)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(16)

        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1e-3)

        self.gpu_checkbox = QtWidgets.QCheckBox("Use GPU")
        self.deterministic_checkbox = QtWidgets.QCheckBox("Deterministic")
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(123)

        layout.addWidget(QtWidgets.QLabel("Latent dim"), 0, 0)
        layout.addWidget(self.latent_dim_spin, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Hidden layers"), 0, 2)
        layout.addWidget(self.hidden_layers_edit, 0, 3)
        layout.addWidget(QtWidgets.QLabel("Activation"), 0, 4)
        layout.addWidget(self.activation_combo, 0, 5)

        layout.addWidget(QtWidgets.QLabel("Epochs"), 1, 0)
        layout.addWidget(self.epochs_spin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Batch size"), 1, 2)
        layout.addWidget(self.batch_spin, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Learning rate"), 1, 4)
        layout.addWidget(self.lr_spin, 1, 5)

        layout.addWidget(self.gpu_checkbox, 2, 0)
        layout.addWidget(self.deterministic_checkbox, 2, 2)
        layout.addWidget(QtWidgets.QLabel("Seed"), 2, 3)
        layout.addWidget(self.seed_spin, 2, 4)
        return group

    def _build_train_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Train")
        layout = QtWidgets.QHBoxLayout(group)

        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_training)
        self.stop_button.clicked.connect(self._stop_training)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_bar)
        return group

    def _build_output_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Outputs")
        layout = QtWidgets.QGridLayout(group)

        self.model_path_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_model_path)

        layout.addWidget(QtWidgets.QLabel("Model artifact"), 0, 0)
        layout.addWidget(self.model_path_edit, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        layout.addWidget(QtWidgets.QLabel("Loss curves saved next to model file."), 1, 1)
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

    def _browse_dataset(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path_edit.setText(folder)

    def _load_dataset(self) -> None:
        folder = self.dataset_path_edit.text().strip()
        if not folder:
            self._log("Select a dataset folder first.")
            return
        map_type = self.map_type_combo.currentText().lower()
        try:
            self.dataset = MapDataset(folder, map_type)
            self.sample_count_label.setText(f"Samples: {len(self.dataset)}")
            self.sample_combo.clear()
            self.sample_combo.addItems(self.dataset.sample_labels)
            self.sample_combo.setEnabled(True)
            self._log(f"Loaded dataset ({self.dataset.map_type}) from {folder}.")
        except Exception as exc:
            self._log(f"Dataset error: {exc}")
            self.dataset = None
            self.sample_combo.clear()
            self.sample_combo.setEnabled(False)

    def _inspect_sample(self) -> None:
        if not self.dataset:
            self._log("Load a dataset before inspecting.")
            return
        try:
            index = self.sample_combo.currentIndex()
            if index < 0:
                self._log("Select a sample to inspect.")
                return
            sample = self.dataset[index].numpy()
            figure = build_sample_figure(sample, self.dataset.axis0, self.dataset.axis1, self.dataset.map_type)
            dialog = SampleInspectDialog(figure, self)
            dialog.exec()
        except Exception as exc:
            self._log(f"Inspect failed: {exc}")

    def _browse_model_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Model Artifact", filter="PyTorch Model (*.pt)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _start_training(self) -> None:
        if self.worker:
            self._log("Training already running.")
            return
        if not self.dataset:
            self._log("Load a dataset before training.")
            return
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self._log("Select a model artifact output path.")
            return

        try:
            hidden_layers = parse_int_list(self.hidden_layers_edit.text())
        except ValueError as exc:
            self._log(str(exc))
            return

        latent_dim = self.latent_dim_spin.value()
        model_arch = build_model_arch(hidden_layers, self.activation_combo.currentText())
        model = MLPAutoencoder(self.dataset.input_shape, latent_dim, hidden_layers, self.activation_combo.currentText())

        config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=float(self.lr_spin.value()),
        )

        seed = self.seed_spin.value()
        if self.deterministic_checkbox.isChecked():
            seed_everything(seed, deterministic=True)
        device = get_device(self.gpu_checkbox.isChecked())
        if device.type == "cpu" and self.gpu_checkbox.isChecked():
            self._log("GPU requested but not available. Using CPU.")

        self.progress_bar.setMaximum(config.epochs)
        self.progress_bar.setValue(0)

        self.last_model_path = Path(model_path)
        self.worker = TrainWorker(
            dataset=self.dataset,
            model=model,
            config=config,
            device=device,
            model_path=self.last_model_path,
            model_arch=model_arch,
            seed=seed,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def _stop_training(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self._log("Stop requested...")

    def _on_progress(self, epoch: int, loss: float) -> None:
        self.progress_bar.setValue(epoch)
        self._log(f"Epoch {epoch}: loss={loss:.6f}")

    def _on_error(self, message: str) -> None:
        self._log(f"Error: {message}")
        self._cleanup_worker()

    def _on_finished(self, losses: list) -> None:
        self._log(f"Training finished. Final loss: {losses[-1]:.6f}")
        if self.last_model_path:
            curve_png = self.last_model_path.with_name(f"{self.last_model_path.stem}_loss.png")
            curve_csv = self.last_model_path.with_name(f"{self.last_model_path.stem}_loss.csv")
            try:
                save_loss_curves(losses, curve_png, curve_csv)
                self._log(f"Saved loss curves to {curve_png}.")
            except Exception as exc:
                self._log(f"Loss curve save error: {exc}")
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
