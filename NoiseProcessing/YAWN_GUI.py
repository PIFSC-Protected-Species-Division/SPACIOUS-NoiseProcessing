# -*- coding: utf-8 -*-
r"""
YAWN_GUI.py

PyQt5 GUI wrapping NoiseApp (noise processing) and the plotting/export
helpers from noiseProcessGoogleCloud.py.

Tab 1 - Noise Processing: point at a local folder or gs:// prefix of audio
files, tweak NoiseApp parameters (all default to NoiseApp's own defaults),
and run the analysis. Scanning the audio source shows the number of files
found plus the first file's name and the date parsed from it, so users can
confirm the date is being read correctly before running a full analysis.

Tab 2 - Plotting & Export: point at HDF5 output file(s), pick a deployment
from a dropdown, choose a plot type from a dropdown, preview it, and export
metrics to CSV.

All processing/plotting/export logic is imported directly from
noiseProcessGoogleCloud.py (not reimplemented here) so future changes to that
module are picked up automatically the next time this GUI is launched.

# Run with the following command in PowerShell:
# C:/Users/kaity/anaconda3/envs/PropagationPython3_12_11/python.exe C:\Users\kaity\Documents\GitHub\SPACIOUS-NoiseProcessing\NoiseProcessing\YAWN_GUI.py
# $env:PYTHONDONTWRITEBYTECODE=1; C:/Users/kaity/anaconda3/envs/PropagationPython3_12_11/python.exe C:\Users\kaity\Documents\GitHub\SPACIOUS-NoiseProcessing\NoiseProcessing\YAWN_GUI.py
#
# Or in Command Prompt (cmd.exe) - the "$env:" syntax above is PowerShell-only
# and raises "The filename, directory name, or volume label syntax is
# incorrect." if pasted into cmd.exe:
# C:\Users\kaity\anaconda3\envs\PropagationPython3_12_11\python.exe C:\Users\kaity\Documents\GitHub\SPACIOUS-NoiseProcessing\NoiseProcessing\YAWN_GUI.py
# set PYTHONDONTWRITEBYTECODE=1 && C:\Users\kaity\anaconda3\envs\PropagationPython3_12_11\python.exe C:\Users\kaity\Documents\GitHub\SPACIOUS-NoiseProcessing\NoiseProcessing\YAWN_GUI.py
#
# If you still see "ImportError: DLL load failed while importing QtCore"
# mentioning PySide6, a stale compiled __pycache__\YAWN_GUI*.pyc from before
# this script was ported to PyQt5 is being reused - delete the __pycache__
# folder next to this file (or run with PYTHONDONTWRITEBYTECODE=1 as above)
# and rerun.
#
# To compile to an executable with PyInstaller, run the following command in
# terminal from the directory containing this script (PySide6 is excluded so
# PyInstaller doesn't bundle a second, conflicting Qt runtime alongside PyQt5's):
# (PropagationPython3_12_11) C:\Users\kaity>pyinstaller --onefile -w --exclude PySide6 C:\Users\kaity\Documents\GitHub\SPACIOUS-NoiseProcessing\NoiseProcessing\YAWN_GUI.py
"""

import contextlib
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Import NoiseApp / plotting / export helpers from noiseProcessGoogleCloud.py
# This file lives alongside noiseProcessGoogleCloud.py inside the
# NoiseProcessing package, so make sure the package's parent directory is
# importable and prefer the package-qualified import.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from NoiseProcessing.noiseProcessGoogleCloud import (  # noqa: E402
        NoiseApp,
        list_hdf5_deployments,
        plot_milidecade_statistics,
        plot_third_octave_bands,
        plot_ltsa,
        export_metric_csv,
        export_all_metrics_csv,
        _normalize_h5_inputs,
    )
except ImportError:
    # Fallback for running this file directly from within the package folder.
    from noiseProcessGoogleCloud import (  # noqa: E402
        NoiseApp,
        list_hdf5_deployments,
        plot_milidecade_statistics,
        plot_third_octave_bands,
        plot_ltsa,
        export_metric_csv,
        export_all_metrics_csv,
        _normalize_h5_inputs,
    )

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402

from PyQt5 import QtCore, QtWidgets  # noqa: E402


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _suppress_pyplot_show():
    """Prevent plotting helpers from calling plt.show() (blocking outside Qt)."""
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show


def _parse_optional_float(text):
    """Return float(text) or None if the field is blank."""
    text = (text or "").strip()
    return float(text) if text else None


def _parse_optional_str(text):
    text = (text or "").strip()
    return text if text else None


def _browse_folder(parent, line_edit):
    path = QtWidgets.QFileDialog.getExistingDirectory(parent, "Select folder", line_edit.text())
    if path:
        line_edit.setText(path)


def _browse_file(parent, line_edit, name_filter="All files (*)"):
    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, "Select file", line_edit.text(), name_filter)
    if path:
        line_edit.setText(path)


def _browse_save_file(parent, line_edit, name_filter="CSV files (*.csv)"):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Select output file", line_edit.text(), name_filter)
    if path:
        line_edit.setText(path)


def _validate_noise_inputs(kwargs):
    """Fail fast on invalid processing settings before analysis starts."""
    sound_path = (kwargs.get("soundFilePath") or "").strip()
    if not sound_path:
        raise ValueError("Enter an audio file location first.")
    if not kwargs.get("DatabaseLoc", "").strip():
        raise ValueError("Choose an output HDF5 folder first.")
    if not kwargs.get("ProjName", "").strip():
        raise ValueError("Project name cannot be blank.")
    if not kwargs.get("DepName", "").strip():
        raise ValueError("Deployment name cannot be blank.")

    fixed_lat = kwargs.get("fixed_lat")
    fixed_lon = kwargs.get("fixed_lon")
    if (fixed_lat is None) ^ (fixed_lon is None):
        raise ValueError("Provide both fixed latitude and fixed longitude together, or leave both blank.")

    if kwargs.get("location_csv") and (fixed_lat is not None or fixed_lon is not None):
        raise ValueError("Use either a location CSV or fixed latitude/longitude, not both.")

    if kwargs.get("clipFileSec", 0) < 0:
        raise ValueError("clipFileSec must be >= 0.")
    if kwargs.get("r", 0) < 0 or kwargs.get("r", 0) >= 1:
        raise ValueError("FFT overlap fraction r must be between 0 and 1.")


def _list_h5_paths(h5_input):
    """Accept a single .h5 file or a folder of them; return sorted path list."""
    h5_input = (h5_input or "").strip()
    if not h5_input:
        return []
    return _normalize_h5_inputs(h5_input)


class _EmittingStream(io.TextIOBase):
    """File-like object that forwards writes to a Qt signal (for log panels)."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def write(self, text):
        if text and text.strip():
            self._signal.emit(text)
        return len(text)

    def flush(self):
        pass


class ThreadRunner(QtCore.QObject):
    """Runs a callable on a background QThread, forwarding stdout as log lines."""

    log = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._result = None

    @property
    def result(self):
        return self._result

    @QtCore.pyqtSlot()
    def run(self):
        stream = _EmittingStream(self.log)
        try:
            with contextlib.redirect_stdout(stream):
                self._result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(True, "Done.")
        except Exception as exc:  # noqa: BLE001
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(exc))


def run_in_background(owner, fn, *args, on_log=None, on_finished=None, **kwargs):
    """Launch fn(*args, **kwargs) on a QThread. Keeps thread/worker alive on owner."""
    thread = QtCore.QThread()
    worker = ThreadRunner(fn, *args, **kwargs)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    if on_log is not None:
        worker.log.connect(on_log)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    if on_finished is not None:
        worker.finished.connect(on_finished)

    # Keep references alive on the owner so they aren't garbage collected mid-run.
    owner._active_threads = getattr(owner, "_active_threads", [])
    owner._active_threads.append((thread, worker))

    def _cleanup(*_args):
        owner._active_threads = [t for t in owner._active_threads if t[0] is not thread]

    thread.finished.connect(_cleanup)
    thread.start()
    return thread, worker


# ---------------------------------------------------------------------------
# Tab 1: Noise Processing
# ---------------------------------------------------------------------------
class NoiseProcessingTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # --- Audio source ---
        source_box = QtWidgets.QGroupBox("Audio source")
        source_layout = QtWidgets.QGridLayout(source_box)

        self.sound_path_edit = QtWidgets.QLineEdit()
        self.sound_path_edit.setPlaceholderText(r"Local folder path or gs://bucket/prefix")
        browse_btn = QtWidgets.QPushButton("Browse folder...")
        browse_btn.clicked.connect(lambda: _browse_folder(self, self.sound_path_edit))
        scan_btn = QtWidgets.QPushButton("Scan")
        scan_btn.clicked.connect(self._scan_audio_source)

        source_layout.addWidget(QtWidgets.QLabel("Audio file location:"), 0, 0)
        source_layout.addWidget(self.sound_path_edit, 0, 1)
        source_layout.addWidget(browse_btn, 0, 2)
        source_layout.addWidget(scan_btn, 0, 3)

        self.file_count_label = QtWidgets.QLabel("Files found: -")
        self.first_file_label = QtWidgets.QLabel("First file: -")
        self.first_date_label = QtWidgets.QLabel("First file date (parsed): -")
        source_layout.addWidget(self.file_count_label, 1, 0, 1, 2)
        source_layout.addWidget(self.first_file_label, 2, 0, 1, 4)
        source_layout.addWidget(self.first_date_label, 3, 0, 1, 4)

        root.addWidget(source_box)

        # --- Parameters ---
        params_box = QtWidgets.QGroupBox("Parameters")
        form = QtWidgets.QFormLayout(params_box)

        self.proj_name_edit = QtWidgets.QLineEdit("MyProject")
        self.dep_name_edit = QtWidgets.QLineEdit("Deployment1")

        self.db_loc_edit = QtWidgets.QLineEdit()
        db_loc_browse = QtWidgets.QPushButton("Browse...")
        db_loc_browse.clicked.connect(lambda: _browse_folder(self, self.db_loc_edit))
        db_loc_row = QtWidgets.QHBoxLayout()
        db_loc_row.addWidget(self.db_loc_edit)
        db_loc_row.addWidget(db_loc_browse)

        self.si_edit = QtWidgets.QLineEdit("-184")
        self.si_edit.setPlaceholderText("scalar dB re 1 V/uPa, or path to calibration CSV")
        si_browse = QtWidgets.QPushButton("Browse CSV...")
        si_browse.clicked.connect(lambda: _browse_file(self, self.si_edit, "CSV files (*.csv)"))
        si_row = QtWidgets.QHBoxLayout()
        si_row.addWidget(self.si_edit)
        si_row.addWidget(si_browse)

        self.si_units_combo = QtWidgets.QComboBox()
        self.si_units_combo.addItems(["V/\u00b5Pa", "V/Pa"])

        self.clip_sec_spin = QtWidgets.QDoubleSpinBox()
        self.clip_sec_spin.setRange(0, 3600)
        self.clip_sec_spin.setValue(0)

        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(0, 63)
        self.channel_spin.setValue(0)

        self.r_spin = QtWidgets.QDoubleSpinBox()
        self.r_spin.setRange(0.0, 0.99)
        self.r_spin.setSingleStep(0.05)
        self.r_spin.setValue(0.5)

        self.winname_combo = QtWidgets.QComboBox()
        self.winname_combo.addItems(["Hann", "Hamming", "Blackman", "None"])

        self.lcut_edit = QtWidgets.QLineEdit()
        self.lcut_edit.setPlaceholderText("auto (0 Hz)")
        self.hcut_edit = QtWidgets.QLineEdit()
        self.hcut_edit.setPlaceholderText("auto (Nyquist)")

        self.avesec_spin = QtWidgets.QDoubleSpinBox()
        self.avesec_spin.setRange(0.1, 3600)
        self.avesec_spin.setValue(60)

        self.pref_spin = QtWidgets.QDoubleSpinBox()
        self.pref_spin.setRange(0.0001, 1000)
        self.pref_spin.setDecimals(4)
        self.pref_spin.setValue(1)

        self.rmdc_check = QtWidgets.QCheckBox("Remove DC offset")
        self.rmdc_check.setChecked(True)

        self.legacy_combo = QtWidgets.QComboBox()
        self.legacy_combo.addItems(["None", "pamguide"])

        self.split_by_day_check = QtWidgets.QCheckBox("Split output HDF5 by day")
        self.split_by_day_check.setChecked(True)

        self.existing_mode_combo = QtWidgets.QComboBox()
        self.existing_mode_combo.addItems(["error", "skip", "overwrite"])

        self.tol_method_combo = QtWidgets.QComboBox()
        self.tol_method_combo.addItems(["psd_sum", "ANSI"])

        self.tol_order_spin = QtWidgets.QSpinBox()
        self.tol_order_spin.setRange(1, 10)
        self.tol_order_spin.setValue(3)

        self.fixed_lat_edit = QtWidgets.QLineEdit()
        self.fixed_lat_edit.setPlaceholderText("optional, e.g. 32.5")
        self.fixed_lon_edit = QtWidgets.QLineEdit()
        self.fixed_lon_edit.setPlaceholderText("optional, e.g. -117.2")

        self.location_csv_edit = QtWidgets.QLineEdit()
        location_csv_browse = QtWidgets.QPushButton("Browse...")
        location_csv_browse.clicked.connect(lambda: _browse_file(self, self.location_csv_edit, "CSV files (*.csv)"))
        location_csv_row = QtWidgets.QHBoxLayout()
        location_csv_row.addWidget(self.location_csv_edit)
        location_csv_row.addWidget(location_csv_browse)

        self.local_staging_edit = QtWidgets.QLineEdit()
        staging_browse = QtWidgets.QPushButton("Browse...")
        staging_browse.clicked.connect(lambda: _browse_folder(self, self.local_staging_edit))
        staging_row = QtWidgets.QHBoxLayout()
        staging_row.addWidget(self.local_staging_edit)
        staging_row.addWidget(staging_browse)

        self.sync_every_spin = QtWidgets.QSpinBox()
        self.sync_every_spin.setRange(1, 10000)
        self.sync_every_spin.setValue(25)

        form.addRow("Project name (ProjName):", self.proj_name_edit)
        form.addRow("Deployment name (DepName):", self.dep_name_edit)
        form.addRow("Output HDF5 folder (DatabaseLoc):", db_loc_row)
        form.addRow("Sensitivity (Si):", si_row)
        form.addRow("Sensitivity units (Si_units):", self.si_units_combo)
        form.addRow("Clip file start (clipFileSec):", self.clip_sec_spin)
        form.addRow("Channel:", self.channel_spin)
        form.addRow("FFT overlap fraction (r):", self.r_spin)
        form.addRow("Window (winname):", self.winname_combo)
        form.addRow("Low cutoff Hz (lcut):", self.lcut_edit)
        form.addRow("High cutoff Hz (hcut):", self.hcut_edit)
        form.addRow("Averaging window sec (aveSec):", self.avesec_spin)
        form.addRow("Reference pressure (pref):", self.pref_spin)
        form.addRow("", self.rmdc_check)
        form.addRow("Legacy mode:", self.legacy_combo)
        form.addRow("", self.split_by_day_check)
        form.addRow("Existing deployment mode:", self.existing_mode_combo)
        form.addRow("Third-octave method (tol_method):", self.tol_method_combo)
        form.addRow("Third-octave filter order (tol_order):", self.tol_order_spin)
        form.addRow("Fixed latitude:", self.fixed_lat_edit)
        form.addRow("Fixed longitude:", self.fixed_lon_edit)
        form.addRow("Location CSV (track):", location_csv_row)
        form.addRow("Local staging folder:", staging_row)
        form.addRow("Sync every N files:", self.sync_every_spin)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(params_box)
        root.addWidget(scroll, stretch=1)

        # --- Run controls ---
        run_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run analysis")
        self.run_btn.clicked.connect(self._run_analysis)
        run_row.addWidget(self.run_btn)
        run_row.addStretch(1)
        root.addLayout(run_row)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        root.addWidget(self.log_edit, stretch=1)

    # -- Scan audio source --
    def _scan_audio_source(self):
        sound_path = self.sound_path_edit.text().strip()
        if not sound_path:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Enter an audio file location first.")
            return
        self.file_count_label.setText("Files found: scanning...")
        self.first_file_label.setText("First file: -")
        self.first_date_label.setText("First file date (parsed): -")

        def _scan():
            # Use a throwaway NoiseApp instance purely to reuse its file
            # discovery and filename date-parsing logic (local or gs://).
            probe = NoiseApp(
                soundFilePath=sound_path,
                ProjName="_gui_probe",
                DepName="_gui_probe",
                DatabaseLoc=tempfile.gettempdir(),
            )
            files = probe._list_audio_inputs()
            if not files:
                return {"count": 0, "first_file": None, "first_date": None}
            first_path = files[0]
            first_name = os.path.basename(first_path)
            probe.get_datetime_format(first_name)
            _, first_dt = probe._date_key_from_name(first_path)
            return {
                "count": len(files),
                "first_file": first_name,
                "first_date": first_dt.strftime("%Y-%m-%d %H:%M:%S") if first_dt else None,
            }

        def _on_finished(ok, message):
            if not ok:
                self.file_count_label.setText("Files found: error")
                QtWidgets.QMessageBox.critical(self, "Scan failed", message)
                return
            info = worker.result
            self.file_count_label.setText(f"Files found: {info['count']}")
            self.first_file_label.setText(f"First file: {info['first_file'] or '-'}")
            self.first_date_label.setText(
                f"First file date (parsed): {info['first_date'] or 'could not be determined from filename'}"
            )

        thread, worker = run_in_background(self, _scan, on_log=self.log_edit.appendPlainText, on_finished=_on_finished)

    # -- Run analysis --
    def _gather_kwargs(self):
        si_text = self.si_edit.text().strip()
        try:
            si_value = float(si_text)
        except ValueError:
            si_value = si_text

        legacy = self.legacy_combo.currentText()
        legacy_mode = None if legacy == "None" else legacy

        fixed_lat = _parse_optional_float(self.fixed_lat_edit.text())
        fixed_lon = _parse_optional_float(self.fixed_lon_edit.text())

        return dict(
            soundFilePath=self.sound_path_edit.text().strip(),
            ProjName=self.proj_name_edit.text().strip(),
            DepName=self.dep_name_edit.text().strip(),
            DatabaseLoc=self.db_loc_edit.text().strip(),
            Si=si_value,
            Si_units=self.si_units_combo.currentText(),
            clipFileSec=self.clip_sec_spin.value(),
            channel=self.channel_spin.value(),
            r=self.r_spin.value(),
            winname=self.winname_combo.currentText(),
            lcut=_parse_optional_float(self.lcut_edit.text()),
            hcut=_parse_optional_float(self.hcut_edit.text()),
            aveSec=self.avesec_spin.value(),
            pref=self.pref_spin.value(),
            rmDC=self.rmdc_check.isChecked(),
            legacy_mode=legacy_mode,
            split_hdf5_by_day=self.split_by_day_check.isChecked(),
            existing_deployment_mode=self.existing_mode_combo.currentText(),
            tol_method=self.tol_method_combo.currentText(),
            tol_order=self.tol_order_spin.value(),
            fixed_lat=fixed_lat,
            fixed_lon=fixed_lon,
            location_csv=_parse_optional_str(self.location_csv_edit.text()),
            local_staging_dir=_parse_optional_str(self.local_staging_edit.text()),
            sync_every_n_files=self.sync_every_spin.value(),
        )

    def _run_analysis(self):
        kwargs = self._gather_kwargs()
        try:
            _validate_noise_inputs(kwargs)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        def _run():
            app = NoiseApp(**kwargs)
            app.run_analysis()
            return app.fullPath

        def _on_finished(ok, message):
            self.run_btn.setEnabled(True)
            if ok:
                QtWidgets.QMessageBox.information(self, "Analysis complete", message)
            else:
                QtWidgets.QMessageBox.critical(self, "Analysis failed", message)

        self.run_btn.setEnabled(False)
        self.log_edit.clear()
        run_in_background(self, _run, on_log=self.log_edit.appendPlainText, on_finished=_on_finished)


# ---------------------------------------------------------------------------
# Tab 2: Plotting & Export
# ---------------------------------------------------------------------------
class PlottingTab(QtWidgets.QWidget):
    PLOT_TYPES = ["Milidecade statistics", "Third-octave bands", "LTSA"]
    METRICS = ["hybrid", "third_octave", "decade", "broadband", "latitude", "longitude", "all"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        controls = QtWidgets.QVBoxLayout()

        # --- Input / output ---
        io_box = QtWidgets.QGroupBox("Input / output")
        io_form = QtWidgets.QFormLayout(io_box)

        self.h5_input_edit = QtWidgets.QLineEdit()
        h5_file_btn = QtWidgets.QPushButton("Browse file...")
        h5_file_btn.clicked.connect(lambda: _browse_file(self, self.h5_input_edit, "HDF5 files (*.h5 *.hdf5)"))
        h5_folder_btn = QtWidgets.QPushButton("Browse folder...")
        h5_folder_btn.clicked.connect(lambda: _browse_folder(self, self.h5_input_edit))
        load_btn = QtWidgets.QPushButton("Load deployments")
        load_btn.clicked.connect(self._load_deployments)
        h5_row = QtWidgets.QHBoxLayout()
        h5_row.addWidget(self.h5_input_edit)
        h5_row.addWidget(h5_file_btn)
        h5_row.addWidget(h5_folder_btn)
        h5_row.addWidget(load_btn)
        io_form.addRow("HDF5 file or folder:", h5_row)

        self.output_dir_edit = QtWidgets.QLineEdit()
        out_browse = QtWidgets.QPushButton("Browse...")
        out_browse.clicked.connect(lambda: _browse_folder(self, self.output_dir_edit))
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.output_dir_edit)
        out_row.addWidget(out_browse)
        io_form.addRow("Output folder:", out_row)

        self.deployment_combo = QtWidgets.QComboBox()
        io_form.addRow("Deployment:", self.deployment_combo)

        controls.addWidget(io_box)

        # --- Plot options ---
        plot_box = QtWidgets.QGroupBox("Plot")
        plot_form = QtWidgets.QFormLayout(plot_box)

        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(self.PLOT_TYPES)
        plot_form.addRow("Plot type:", self.plot_type_combo)

        self.pbands_edit = QtWidgets.QLineEdit("5,25,50,75,95")
        plot_form.addRow("Percentiles (pBands):", self.pbands_edit)

        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("optional")
        plot_form.addRow("Title:", self.title_edit)

        self.averaging_period_edit = QtWidgets.QLineEdit("5min")
        plot_form.addRow("LTSA averaging period:", self.averaging_period_edit)

        self.freq_scaled_check = QtWidgets.QCheckBox("Use real frequency axis (LTSA)")
        self.freq_scaled_check.setChecked(True)
        plot_form.addRow("", self.freq_scaled_check)

        self.log_freq_check = QtWidgets.QCheckBox("Log-scale frequency axis (LTSA)")
        plot_form.addRow("", self.log_freq_check)

        generate_btn = QtWidgets.QPushButton("Generate plot")
        generate_btn.clicked.connect(self._generate_plot)
        save_plot_btn = QtWidgets.QPushButton("Save plot as...")
        save_plot_btn.clicked.connect(self._save_plot)
        plot_btn_row = QtWidgets.QHBoxLayout()
        plot_btn_row.addWidget(generate_btn)
        plot_btn_row.addWidget(save_plot_btn)
        plot_form.addRow(plot_btn_row)

        controls.addWidget(plot_box)

        # --- Export options ---
        export_box = QtWidgets.QGroupBox("Export to CSV")
        export_form = QtWidgets.QFormLayout(export_box)

        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(self.METRICS)
        export_form.addRow("Metric:", self.metric_combo)

        self.export_csv_edit = QtWidgets.QLineEdit()
        export_browse = QtWidgets.QPushButton("Browse...")
        export_browse.clicked.connect(lambda: _browse_save_file(self, self.export_csv_edit))
        export_row = QtWidgets.QHBoxLayout()
        export_row.addWidget(self.export_csv_edit)
        export_row.addWidget(export_browse)
        export_form.addRow("Output CSV (single metric):", export_row)

        export_btn = QtWidgets.QPushButton("Export selected metric")
        export_btn.clicked.connect(self._export_metric)
        export_all_btn = QtWidgets.QPushButton("Export all metrics to output folder")
        export_all_btn.clicked.connect(self._export_all_metrics)
        export_btn_row = QtWidgets.QHBoxLayout()
        export_btn_row.addWidget(export_btn)
        export_btn_row.addWidget(export_all_btn)
        export_form.addRow(export_btn_row)

        controls.addWidget(export_box)
        controls.addStretch(1)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        self.log_edit.setMaximumHeight(120)
        controls.addWidget(self.log_edit)

        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(controls)
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(420)

        self._current_fig = None
        self.canvas = FigureCanvasQTAgg(plt.figure(figsize=(8, 6)))

        root.addWidget(controls_scroll)
        root.addWidget(self.canvas, stretch=1)

    # -- Helpers --
    def _selected_h5_paths(self):
        return _list_h5_paths(self.h5_input_edit.text().strip())

    def _selected_deployment(self):
        text = self.deployment_combo.currentText().strip()
        return [text] if text else []

    def _load_deployments(self):
        paths = self._selected_h5_paths()
        if not paths:
            QtWidgets.QMessageBox.warning(self, "No files", "Enter a valid HDF5 file or folder first.")
            return

        names = set()
        for p in paths:
            try:
                names.update(list_hdf5_deployments(p))
            except Exception as exc:  # noqa: BLE001
                self.log_edit.appendPlainText(f"Failed to read {p}: {exc}")

        self.deployment_combo.clear()
        self.deployment_combo.addItems(sorted(names))

        if not names:
            QtWidgets.QMessageBox.warning(self, "No deployments", "No deployment groups found in the selected file(s).")

    def _open_selected_groups(self, stack):
        """Open the selected deployment across all selected HDF5 files."""
        paths = self._selected_h5_paths()
        deployments = self._selected_deployment()
        if not paths:
            raise ValueError("Enter a valid HDF5 file or folder first.")
        if not deployments:
            raise ValueError("Select a deployment.")

        groups = []
        for p in paths:
            h5 = stack.enter_context(h5py.File(p, "r"))
            for name in deployments:
                if name in h5:
                    groups.append(h5[name])
        if not groups:
            raise ValueError("The selected deployment was not found in the selected HDF5 file(s).")
        return groups

    # -- Plot generation --
    def _generate_plot(self):
        try:
            pbands = [int(x.strip()) for x in self.pbands_edit.text().split(",") if x.strip()]
            if not pbands:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Percentiles must be a comma-separated list of integers.")
            return

        if self.plot_type_combo.currentText() == "Third-octave bands" and len(pbands) != 5:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid percentiles",
                "Third-octave plots require exactly 5 percentiles: 5,25,50,75,95.",
            )
            return

        title = _parse_optional_str(self.title_edit.text())

        with contextlib.ExitStack() as stack:
            try:
                groups = self._open_selected_groups(stack)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Cannot plot", str(exc))
                return

            plot_type = self.plot_type_combo.currentText()
            try:
                with _suppress_pyplot_show():
                    if plot_type == "Milidecade statistics":
                        fig = plot_milidecade_statistics(groups, pBands=pbands, title=title)
                    elif plot_type == "Third-octave bands":
                        fig = plot_third_octave_bands(groups, pBands=pbands, title=title)
                    else:  # LTSA
                        fig = plot_ltsa(
                            groups,
                            averaging_period=self.averaging_period_edit.text().strip() or "5min",
                            title=title,
                            freq_scaled=self.freq_scaled_check.isChecked(),
                            log_freq=self.log_freq_check.isChecked(),
                        )
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Plot failed", str(exc))
                self.log_edit.appendPlainText(traceback.format_exc())
                return

        self._replace_canvas(fig)

    def _replace_canvas(self, fig):
        old_canvas = self.canvas
        self.canvas = FigureCanvasQTAgg(fig)
        layout = self.layout()
        layout.replaceWidget(old_canvas, self.canvas)
        old_canvas.setParent(None)
        if self._current_fig is not None:
            plt.close(self._current_fig)
        self._current_fig = fig
        self.canvas.draw()

    def _save_plot(self):
        if self._current_fig is None:
            QtWidgets.QMessageBox.warning(self, "No plot", "Generate a plot first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "", "PNG image (*.png);;JPEG image (*.jpg)")
        if path:
            self._current_fig.savefig(path, dpi=300, bbox_inches="tight")
            self.log_edit.appendPlainText(f"Saved plot to {path}")

    # -- Export --
    def _export_metric(self):
        paths = self._selected_h5_paths()
        deployments = self._selected_deployment()
        if not paths or not deployments:
            QtWidgets.QMessageBox.warning(self, "Cannot export", "Load an HDF5 file/folder and select a deployment first.")
            return
        out_csv = self.export_csv_edit.text().strip()
        if not out_csv:
            QtWidgets.QMessageBox.warning(self, "Missing output", "Choose an output CSV path first.")
            return

        metric = self.metric_combo.currentText()
        try:
            if metric == "all":
                out_dir = self.output_dir_edit.text().strip() or str(Path(out_csv).parent)
                exported = export_all_metrics_csv(paths, out_dir, group_name=deployments)
                self.log_edit.appendPlainText(f"Exported: {exported}")
            else:
                result_path = export_metric_csv(paths, metric, out_csv, group_name=deployments)
                self.log_edit.appendPlainText(f"Exported to {result_path}")
            QtWidgets.QMessageBox.information(self, "Export complete", "CSV export finished.")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            self.log_edit.appendPlainText(traceback.format_exc())

    def _export_all_metrics(self):
        paths = self._selected_h5_paths()
        deployments = self._selected_deployment()
        if not paths or not deployments:
            QtWidgets.QMessageBox.warning(self, "Cannot export", "Load an HDF5 file/folder and select a deployment first.")
            return
        out_dir = self.output_dir_edit.text().strip()
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "Missing output", "Choose an output folder first.")
            return
        try:
            exported = export_all_metrics_csv(paths, out_dir, group_name=deployments)
            self.log_edit.appendPlainText(f"Exported: {exported}")
            QtWidgets.QMessageBox.information(self, "Export complete", f"Exported {len(exported)} metric CSV(s) to {out_dir}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            self.log_edit.appendPlainText(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YAWN - Yet Another Wave/Noise processor")
        self.resize(1200, 800)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(NoiseProcessingTab(), "Noise Processing")
        tabs.addTab(PlottingTab(), "Plotting && Export")
        self.setCentralWidget(tabs)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
