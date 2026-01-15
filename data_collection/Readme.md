# Rocket Measurement GUI

This project provides a simple GUI to acquire and display live measurement data for propulsion experiments (rocket engines / micro turbojets).

Current support:
- NI cDAQ-9185 (Ethernet)
- NI-9212: 3× thermocouple channels (ai0–ai2)
- NI-9201: 4× analog input channels (ai0–ai3)

Planned extensions:
- RS232 load cell
- Data logging (CSV / TDMS)
- Live plots
- Test annotations

The GUI is intentionally lightweight and suitable for slow lab PCs.

---

## Folder Structure

```

project/
├─ main.py              # Main GUI application
├─ setup_python.bat     # One-time Python dependency installer (double-click)
├─ run_gui.bat          # Run the GUI (double-click)
└─ README.md

```

---

## System Requirements

Hardware:
- NI cDAQ-9185 chassis
- NI-9212 module
- NI-9201 module
- Ethernet connection to PC

Software:
- Windows 11 (64-bit)
- Python 3.10+ (3.11 recommended)
- NI-DAQmx driver + NI MAX

No paid NI licenses are required for data acquisition.

---

## Network Setup (PC ↔ cDAQ)

Direct Ethernet connection is supported (no router needed).

Recommended static IP setup for bench use:

PC Ethernet:
- IP address: `192.168.10.1`
- Subnet mask: `255.255.255.0`
- Gateway: *(leave empty)*

cDAQ:
- IP address: `192.168.10.2`
- Subnet mask: `255.255.255.0`

Verify connectivity:
```

ping 192.168.10.2

```

---

## NI Software Installation (One-time)

1. Install **NI Package Manager**
2. Using NI Package Manager, install:
   - **NI-DAQmx**
     - Include **NI Measurement & Automation Explorer (MAX)**
     - Include **DAQmx Runtime**

You do NOT need:
- LabVIEW
- Real-Time / FPGA
- Toolkits

Reboot if prompted.

---

## Verify Hardware in NI MAX

1. Open **NI Measurement & Automation Explorer (MAX)**
2. Navigate to:
```

Devices and Interfaces → Network Devices

```
3. Confirm:
- cDAQ-9185 is visible
- NI-9212 and NI-9201 modules are detected

4. Note the exact module names shown in MAX, for example:
- `cDAQ9185-1A2B3C4DMod1`  (NI-9212)
- `cDAQ9185-1A2B3C4DMod2`  (NI-9201)

These names are entered into the GUI.

5. Use **Test Panels** to verify live readings.

If MAX Test Panels work, the GUI will work.

---

## Python Setup (One-time)

1. Install Python 3.10+  
During install, enable:
- ✔ Add Python to PATH

2. Double-click:
```

setup_python.bat

```

This installs the required Python dependency:
- `nidaqmx`

Tkinter is included with standard Python on Windows.

---

## Running the GUI

To start the application:
```

double-click run_gui.bat

```

In the GUI:
1. Enter the NI-9212 module name (from NI MAX)
2. Enter the NI-9201 module name (from NI MAX)
3. Select thermocouple type (usually K for EGT)
4. Click:
   - **Connect**
   - **Start**

Displayed values:
- Thermocouples: °C
- Analog inputs: Volts (raw)

---

## Notes for Engine Testing

- The current GUI reads single samples for live display.
- It is intended for validation and monitoring, not high-rate logging.
- For actual test runs:
  - Use buffered continuous acquisition
  - Log data to disk to avoid sample loss
  - Treat ECU software as control-only; NI data as authoritative

These features will be added incrementally.

---

## Common Issues

**Device not visible in MAX**
- Check Ethernet LEDs
- Verify PC and cDAQ are on the same subnet
- Temporarily disable Windows Firewall for discovery testing

**Python read errors**
- Ensure MAX Test Panels work
- Ensure no other program is holding DAQ tasks open
- Check module names for typos

**Incorrect thermocouple readings**
- Wrong TC type selected
- Polarity reversed
- Open / floating thermocouple

---

## Future Extensions

Planned additions:
- RS232 load cell input
- Unified timestamping
- CSV / TDMS logging
- Live plots
- Test metadata (run ID, notes, operator)

The project is intentionally structured to grow into a full propulsion test acquisition system.
