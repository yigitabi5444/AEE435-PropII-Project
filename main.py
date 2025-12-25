import tkinter as tk
from tkinter import ttk, messagebox
import nidaqmx
from nidaqmx.constants import TemperatureUnits, ThermocoupleType
import math
import time


class NiDaqGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cDAQ Live Readout (NI-9212 TC + NI-9201 AI)")
        self.geometry("720x420")

        # DAQ tasks
        self.tc_task = None
        self.ai_task = None
        self.running = False
        self.after_id = None

        # --- UI Vars ---
        # These should match what NI MAX shows. Example: "cDAQ9185-1A2B3C4DMod1"
        self.tc_module = tk.StringVar(value="cDAQ9185-20050D7Mod4")  # NI-9212
        self.ai_module = tk.StringVar(value="cDAQ9185-20050D7Mod3")  # NI-9201

        self.tc_type = tk.StringVar(value="K")      # change if needed
        self.sample_period_ms = tk.IntVar(value=200)  # 5 Hz UI update

        # Readouts
        self.tc_vals = [tk.StringVar(value="—") for _ in range(3)]
        self.ai_vals = [tk.StringVar(value="—") for _ in range(4)]
        self.status = tk.StringVar(value="Disconnected")

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
                flat = [ch[0] if ch else float("nan") for ch in data]
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

    def _get_period_ms(self):
        try:
            value = int(self.sample_period_ms.get())
        except Exception:
            return 200
        return max(50, value)

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

        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status.set("Running")

        # Kick off periodic read (UI thread)
        self._tick()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
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
            # Read ONE sample per channel (raw-ish live values)
            # Returns list[float] when multiple channels
            tc = self.tc_task.read(number_of_samples_per_channel=1)
            ai = self.ai_task.read(number_of_samples_per_channel=1)

            tc = self._normalize_read(tc, 3)
            ai = self._normalize_read(ai, 4)

            for i in range(3):
                self.tc_vals[i].set(self._format_value(tc[i], ".2f"))
            for i in range(4):
                self.ai_vals[i].set(self._format_value(ai[i], ".4f"))

        except Exception as e:
            # Stop acquisition but keep connection so user can retry
            self.running = False
            self.btn_stop.config(state="disabled")
            self.btn_start.config(state="normal")
            self.status.set("Error (stopped)")
            messagebox.showerror("Read failed", f"{type(e).__name__}: {e}")
            return

        # Schedule next tick
        period = self._get_period_ms()
        self.after_id = self.after(period, self._tick)

    def on_close(self):
        try:
            self.disconnect()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = NiDaqGui()
    app.mainloop()
