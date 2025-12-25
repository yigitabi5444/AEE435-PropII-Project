import csv
import os
import tkinter as tk
from tkinter import ttk, messagebox
import nidaqmx
from nidaqmx.constants import AcquisitionType, TemperatureUnits, ThermocoupleType
from nidaqmx.errors import DaqError
import math
import time


class NiDaqGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cDAQ Live Readout (NI-9212 TC + NI-9201 AI)")
        self.geometry("720x420")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_path = os.path.join(self.script_dir, "calibration.yaml")

        # DAQ tasks
        self.tc_task = None
        self.ai_task = None
        self.running = False
        self.after_id = None
        self.log_file = None
        self.log_writer = None

        # --- UI Vars ---
        # These should match what NI MAX shows. Example: "cDAQ9185-1A2B3C4DMod1"
        self.tc_module = tk.StringVar(value="cDAQ9185-20050D7Mod4")  # NI-9212
        self.ai_module = tk.StringVar(value="cDAQ9185-20050D7Mod3")  # NI-9201

        self.tc_type = tk.StringVar(value="K")      # change if needed
        self.sample_period_ms = tk.IntVar(value=200)  # 5 Hz UI update
        self.logging_enabled = tk.BooleanVar(value=False)

        # Readouts
        self.tc_vals = [tk.StringVar(value="—") for _ in range(3)]
        self.ai_vals = [tk.StringVar(value="—") for _ in range(4)]
        self.status = tk.StringVar(value="Disconnected")

        self.last_tc_raw = [None] * 3
        self.last_ai_raw = [None] * 4
        self.calibration = self._default_calibration()
        self._load_calibration()

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        # Connection settings
        conn = ttk.LabelFrame(frm, text="Connection / Channel Mapping")
        conn.pack(fill="x", **pad)

        ttk.Label(conn, text="NI-9212 module name (in MAX):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(conn, textvariable=self.tc_module, width=35).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(conn, text="NI-9201 module name (in MAX):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(conn, textvariable=self.ai_module, width=35).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(conn, text="Thermocouple type:").grid(row=0, column=2, sticky="w", **pad)
        tc_combo = ttk.Combobox(conn, textvariable=self.tc_type, values=["J", "K", "T", "E", "N", "R", "S", "B"], width=6, state="readonly")
        tc_combo.grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(conn, text="Update period (ms):").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(conn, textvariable=self.sample_period_ms, width=8).grid(row=1, column=3, sticky="w", **pad)

        # Buttons + status
        btns = ttk.Frame(frm)
        btns.pack(fill="x", **pad)

        self.btn_connect = ttk.Button(btns, text="Connect", command=self.connect)
        self.btn_connect.pack(side="left", padx=5)

        self.btn_start = ttk.Button(btns, text="Start", command=self.start, state="disabled")
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.btn_calibrate = ttk.Button(btns, text="Calibration...", command=self.open_calibration_dialog)
        self.btn_calibrate.pack(side="left", padx=5)

        self.chk_log = ttk.Checkbutton(
            btns,
            text="Log to CSV",
            variable=self.logging_enabled,
            command=self._on_logging_toggle,
        )
        self.chk_log.pack(side="left", padx=8)

        self.btn_disconnect = ttk.Button(btns, text="Disconnect", command=self.disconnect, state="disabled")
        self.btn_disconnect.pack(side="left", padx=5)

        ttk.Label(btns, textvariable=self.status).pack(side="right", padx=5)

        # Readouts
        ro = ttk.Frame(frm)
        ro.pack(fill="both", expand=True, **pad)

        tc_box = ttk.LabelFrame(ro, text="Thermocouples (NI-9212) ai0..ai2 (°C)")
        tc_box.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        for i in range(3):
            ttk.Label(tc_box, text=f"TC{i} (ai{i}):").grid(row=i, column=0, sticky="w", padx=10, pady=10)
            ttk.Label(tc_box, textvariable=self.tc_vals[i], font=("TkDefaultFont", 12, "bold")).grid(row=i, column=1, sticky="w", padx=10, pady=10)

        ai_box = ttk.LabelFrame(ro, text="Pressure / Analog Inputs (NI-9201) ai0..ai3 (V)")
        ai_box.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        for i in range(4):
            ttk.Label(ai_box, text=f"AI{i} (ai{i}):").grid(row=i, column=0, sticky="w", padx=10, pady=10)
            ttk.Label(ai_box, textvariable=self.ai_vals[i], font=("TkDefaultFont", 12, "bold")).grid(row=i, column=1, sticky="w", padx=10, pady=10)

        # Close handler
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _tc_enum(self):
        m = {
            "J": ThermocoupleType.J,
            "K": ThermocoupleType.K,
            "T": ThermocoupleType.T,
            "E": ThermocoupleType.E,
            "N": ThermocoupleType.N,
            "R": ThermocoupleType.R,
            "S": ThermocoupleType.S,
            "B": ThermocoupleType.B,
        }
        return m.get(self.tc_type.get().strip().upper(), ThermocoupleType.K)

    def _normalize_read(self, data, expected_channels):
        # Normalize NI-DAQmx return shapes to one scalar per channel.
        if isinstance(data, (list, tuple)):
            if data and isinstance(data[0], (list, tuple)):
                flat = [ch[-1] if ch else float("nan") for ch in data]
            else:
                flat = list(data)
        else:
            flat = [data]

        if len(flat) < expected_channels:
            flat.extend([float("nan")] * (expected_channels - len(flat)))
        else:
            flat = flat[:expected_channels]
        return flat

    def _format_value(self, value, fmt):
        if value is None:
            return "—"
        try:
            if isinstance(value, float) and math.isnan(value):
                return "—"
            return format(value, fmt)
        except Exception:
            return "—"

    def _channel_names(self):
        return ["TC0", "TC1", "TC2", "AI0", "AI1", "AI2", "AI3"]

    def _default_calibration(self):
        cal = {}
        for name in self._channel_names():
            cal[name] = {"raw1": 0.0, "eng1": 0.0, "raw2": 1.0, "eng2": 1.0}
        return cal

    def _parse_scalar(self, value):
        value = value.strip()
        if not value:
            return None
        low = value.lower()
        if low in ("null", "none"):
            return None
        try:
            if "." in value or "e" in low:
                return float(value)
            return int(value)
        except Exception:
            return value.strip("\"'")

    def _parse_simple_yaml(self, text):
        data = {}
        stack = [(-1, data)]
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, sep, tail = line.strip().partition(":")
            if not sep:
                continue
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1] if stack else data
            tail = tail.strip()
            if tail == "":
                new_dict = {}
                parent[key] = new_dict
                stack.append((indent, new_dict))
            else:
                parent[key] = self._parse_scalar(tail)
        return data

    def _load_calibration(self):
        if not os.path.exists(self.calibration_path):
            self._save_calibration()
            return
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception:
            return
        data = self._parse_simple_yaml(text)
        channels = {}
        if isinstance(data, dict):
            if isinstance(data.get("channels"), dict):
                channels = data["channels"]
            else:
                channels = data
        for name in self._channel_names():
            entry = channels.get(name)
            if not isinstance(entry, dict):
                continue
            for key in ("raw1", "eng1", "raw2", "eng2"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    self.calibration[name][key] = float(value)

    def _save_calibration(self):
        lines = ["version: 1", "channels:"]
        for name in self._channel_names():
            lines.append(f"  {name}:")
            for key in ("raw1", "eng1", "raw2", "eng2"):
                value = self.calibration[name].get(key, 0.0)
                lines.append(f"    {key}: {value}")
        text = "\n".join(lines) + "\n"
        with open(self.calibration_path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _apply_calibration(self, channel, value):
        if value is None:
            return value
        try:
            if isinstance(value, float) and math.isnan(value):
                return value
        except Exception:
            return value
        entry = self.calibration.get(channel)
        if not entry:
            return value
        raw1 = entry.get("raw1")
        raw2 = entry.get("raw2")
        eng1 = entry.get("eng1")
        eng2 = entry.get("eng2")
        if None in (raw1, raw2, eng1, eng2):
            return value
        if raw2 == raw1:
            return eng1
        return eng1 + (value - raw1) * (eng2 - eng1) / (raw2 - raw1)

    def _open_log(self):
        if self.log_file:
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"log_{timestamp}.csv"
            path = os.path.join(self.script_dir, filename)
            self.log_file = open(path, "w", newline="", encoding="utf-8")
            self.log_writer = csv.writer(self.log_file)
            header = ["timestamp"]
            for i in range(3):
                header.append(f"tc{i}_raw")
            for i in range(3):
                header.append(f"tc{i}_cal")
            for i in range(4):
                header.append(f"ai{i}_raw")
            for i in range(4):
                header.append(f"ai{i}_cal")
            self.log_writer.writerow(header)
            self.log_file.flush()
        except Exception as exc:
            self._close_log()
            self.logging_enabled.set(False)
            messagebox.showerror("Log failed", f"{type(exc).__name__}: {exc}")

    def _close_log(self):
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
        self.log_file = None
        self.log_writer = None

    def _on_logging_toggle(self):
        if self.running:
            if self.logging_enabled.get():
                self._open_log()
            else:
                self._close_log()

    def _get_period_ms(self):
        try:
            value = int(self.sample_period_ms.get())
        except Exception:
            return 200
        return max(50, value)

    def _configure_timing(self):
        period_ms = self._get_period_ms()
        ui_rate = 1000.0 / period_ms
        tc_rate = max(1.0, min(10.0, ui_rate * 2.0))
        ai_rate = max(10.0, min(1000.0, ui_rate * 10.0))

        self.tc_task.timing.cfg_samp_clk_timing(
            rate=tc_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=max(2, int(tc_rate * 2)),
        )
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=ai_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=max(2, int(ai_rate * 2)),
        )

    def _read_latest(self, task, expected_channels, max_samples=200):
        try:
            available = int(task.in_stream.avail_samp_per_chan)
        except Exception:
            return None

        if available <= 0:
            return None

        samples_to_read = min(available, max_samples)
        try:
            data = task.read(number_of_samples_per_channel=samples_to_read, timeout=0)
        except DaqError as exc:
            if exc.error_code == -200284:
                return None
            raise
        return self._normalize_read(data, expected_channels)

    def connect(self):
        if self.tc_task or self.ai_task:
            messagebox.showinfo("Info", "Already connected.")
            return

        tc_mod = self.tc_module.get().strip()
        ai_mod = self.ai_module.get().strip()
        if not tc_mod or not ai_mod:
            messagebox.showerror("Error", "Please enter module names as shown in NI MAX.")
            return

        try:
            # Create tasks
            self.tc_task = nidaqmx.Task(new_task_name="TC_Task")
            self.ai_task = nidaqmx.Task(new_task_name="AI_Task")

            # Add channels
            # NI-9212: ai0..ai2 thermocouple
            for i in range(3):
                ch = f"{tc_mod}/ai{i}"
                self.tc_task.ai_channels.add_ai_thrmcpl_chan(
                    physical_channel=ch,
                    thermocouple_type=self._tc_enum(),
                    units=TemperatureUnits.DEG_C
                )

            # NI-9201: ai0..ai3 voltage
            for i in range(4):
                ch = f"{ai_mod}/ai{i}"
                self.ai_task.ai_channels.add_ai_voltage_chan(ch)

            self.tc_task.in_stream.read_all_avail_samp = True
            self.ai_task.in_stream.read_all_avail_samp = True

            self.status.set("Connected (tasks created)")
            self.btn_start.config(state="normal")
            self.btn_disconnect.config(state="normal")
            self.btn_connect.config(state="disabled")

        except Exception as e:
            self._cleanup_tasks()
            messagebox.showerror("Connect failed", f"{type(e).__name__}: {e}")
            self.status.set("Disconnected")

    def start(self):
        if not self.tc_task or not self.ai_task:
            messagebox.showerror("Error", "Not connected.")
            return

        if self.running:
            return

        try:
            self._configure_timing()
            self.tc_task.start()
            self.ai_task.start()
        except Exception as e:
            messagebox.showerror("Start failed", f"{type(e).__name__}: {e}")
            self.status.set("Connected (stopped)")
            return

        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status.set("Running")
        if self.logging_enabled.get():
            self._open_log()

        # Kick off periodic read (UI thread)
        self._tick()

    def stop(self):
        if not self.running:
            return
        self.running = False
        for t in (self.tc_task, self.ai_task):
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self._close_log()
        self.btn_stop.config(state="disabled")
        self.btn_start.config(state="normal")
        self.status.set("Connected (stopped)")

    def disconnect(self):
        self.stop()
        self._cleanup_tasks()
        self.btn_connect.config(state="normal")
        self.btn_disconnect.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.status.set("Disconnected")
        for v in self.tc_vals + self.ai_vals:
            v.set("—")

    def _cleanup_tasks(self):
        for t in (self.tc_task, self.ai_task):
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
        self.tc_task = None
        self.ai_task = None

    def _tick(self):
        if not self.running:
            return

        try:
            tc = self._read_latest(self.tc_task, 3)
            ai = self._read_latest(self.ai_task, 4)

            updated = False
            if tc is not None:
                self.last_tc_raw = tc
                updated = True
            if ai is not None:
                self.last_ai_raw = ai
                updated = True

            tc_cal = [self._apply_calibration(f"TC{i}", v) for i, v in enumerate(self.last_tc_raw)]
            ai_cal = [self._apply_calibration(f"AI{i}", v) for i, v in enumerate(self.last_ai_raw)]

            if any(v is not None for v in self.last_tc_raw):
                for i in range(3):
                    self.tc_vals[i].set(self._format_value(tc_cal[i], ".2f"))
            if any(v is not None for v in self.last_ai_raw):
                for i in range(4):
                    self.ai_vals[i].set(self._format_value(ai_cal[i], ".4f"))

            if self.log_writer and updated:
                row = [time.strftime("%Y-%m-%d %H:%M:%S")]
                row.extend(self.last_tc_raw)
                row.extend(tc_cal)
                row.extend(self.last_ai_raw)
                row.extend(ai_cal)
                self.log_writer.writerow(row)
                self.log_file.flush()

        except Exception as e:
            # Stop acquisition but keep connection so user can retry
            self.running = False
            self._close_log()
            self.btn_stop.config(state="disabled")
            self.btn_start.config(state="normal")
            self.status.set("Error (stopped)")
            messagebox.showerror("Read failed", f"{type(e).__name__}: {e}")
            return

        # Schedule next tick
        period = self._get_period_ms()
        self.after_id = self.after(period, self._tick)

    def open_calibration_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("Calibration (2-point)")
        dialog.transient(self)
        dialog.grab_set()

        keys = ("raw1", "eng1", "raw2", "eng2")
        ttk.Label(dialog, text="Channel").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        for col, key in enumerate(keys, start=1):
            ttk.Label(dialog, text=key).grid(row=0, column=col, padx=6, pady=6, sticky="w")

        entries = {}
        for row, name in enumerate(self._channel_names(), start=1):
            ttk.Label(dialog, text=name).grid(row=row, column=0, padx=6, pady=4, sticky="w")
            entries[name] = {}
            for col, key in enumerate(keys, start=1):
                value = self.calibration.get(name, {}).get(key, 0.0)
                var = tk.StringVar(value=str(value))
                ttk.Entry(dialog, textvariable=var, width=10).grid(row=row, column=col, padx=4, pady=4, sticky="w")
                entries[name][key] = var

        btns = ttk.Frame(dialog)
        btns.grid(row=len(self._channel_names()) + 1, column=0, columnspan=5, pady=10, sticky="e")

        def apply_defaults():
            defaults = self._default_calibration()
            self.calibration = defaults
            self._save_calibration()
            for name in self._channel_names():
                for key in keys:
                    entries[name][key].set(str(defaults[name][key]))

        def save_and_close():
            new_cal = self._default_calibration()
            for name in self._channel_names():
                for key in keys:
                    text = entries[name][key].get().strip()
                    try:
                        value = float(text)
                    except Exception:
                        messagebox.showerror("Invalid value", f"{name} {key} must be a number.")
                        return
                    new_cal[name][key] = value
            self.calibration = new_cal
            self._save_calibration()
            dialog.destroy()

        ttk.Button(btns, text="Reset", command=apply_defaults).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)
        ttk.Button(btns, text="Save", command=save_and_close).pack(side="right", padx=5)

    def on_close(self):
        try:
            self.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = NiDaqGui()
    app.mainloop()
