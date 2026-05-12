# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:51:50 2025

@author: kaity
"""

import os
import re
import glob
import math
import shutil
import tempfile
import time as time_module
from contextlib import ExitStack
from urllib.parse import urlparse
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

import soundfile as sf
import h5py
import scipy
import seaborn as sns

# Optional GCS dependency
try:
    from google.cloud import storage
    _HAS_GCS = True
except Exception:
    _HAS_GCS = False



import h5py

def print_h5_tree(h5_path, max_depth=6):
    """Print a simple tree view of groups and datasets in an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        def walk(name, obj):
            depth = name.count("/")
            if depth > max_depth:
                return
            if isinstance(obj, h5py.Dataset):
                print(f"{name}  DATASET shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"{name}  GROUP")
        f.visititems(walk)

# example:
# print_h5_tree(r"C:\path\to\one_file.h5")



def get_band_table(fft_bin_size,
                   bin1_center_frequency=0,
                   fs=64000, base=10,
                   bands_per_division=1000,
                   first_output_band_center_frequency=455,
                   use_fft_res_at_bottom=False):
    """
    Returns an array of [start, center, stop] frequencies for logarithmically
    spaced frequency bands (milli-decades / third-octaves, etc.).
    """
    band_count = 0
    max_freq = fs / 2
    low_side_multiplier = base ** (-1 / (2 * bands_per_division))
    high_side_multiplier = base ** (1 / (2 * bands_per_division))
    center_freq = 0
    linear_bin_count = 0
    log_bin_count = 0

    if use_fft_res_at_bottom:
        bin_width = 0
        while bin_width < fft_bin_size:
            band_count += 1
            center_freq = first_output_band_center_frequency * base ** (band_count / bands_per_division)
            bin_width = high_side_multiplier * center_freq - low_side_multiplier * center_freq

        center_freq = first_output_band_center_frequency * base ** (band_count / bands_per_division)
        linear_bin_count = int(np.ceil(center_freq / fft_bin_size))

        while (linear_bin_count * fft_bin_size - center_freq) > 0.0:
            band_count += 1
            linear_bin_count += 1
            center_freq = first_output_band_center_frequency * base ** (band_count / bands_per_division)

        if fft_bin_size * linear_bin_count > max_freq:
            linear_bin_count = int(max_freq / fft_bin_size) + 1
    else:
        linear_bin_count = 0

    log_band1 = band_count

    # Count the log-space frequencies
    while max_freq > center_freq:
        band_count += 1
        log_bin_count += 1
        center_freq = first_output_band_center_frequency * base ** (band_count / bands_per_division)

    # Initialize the bands array
    bands = np.zeros((linear_bin_count + log_bin_count, 3))

    # Generate the linear frequencies
    for i in range(linear_bin_count):
        center_freq = bin1_center_frequency + i * fft_bin_size
        bands[i, 1] = center_freq
        bands[i, 0] = center_freq - fft_bin_size / 2
        bands[i, 2] = center_freq + fft_bin_size / 2

    # Generate the log-spaced bands
    for i in range(log_bin_count):
        out_band_number = linear_bin_count + i
        m_dec_number = log_band1 + i + 1
        center_freq = first_output_band_center_frequency * base ** ((m_dec_number - 1) / bands_per_division)
        bands[out_band_number, 1] = center_freq
        bands[out_band_number, 0] = center_freq * low_side_multiplier
        bands[out_band_number, 2] = center_freq * high_side_multiplier

    # Adjust the upper bound of the last band
    if log_bin_count > 0:
        bands[out_band_number, 2] = max_freq

    return bands


def buffer(data, duration, dataOverlap):
    """Split a 1-D array into overlapping fixed-length segments with zero padding."""
    numberOfSegments = int(math.ceil((len(data) - dataOverlap) / (duration - dataOverlap)))
    tempBuf = [data[i:i + duration] for i in range(0, len(data), (duration - int(dataOverlap)))]
    tempBuf[numberOfSegments - 1] = np.pad(tempBuf[numberOfSegments - 1],
                                           (0, duration - tempBuf[numberOfSegments - 1].shape[0]),
                                           'constant')
    tempBuf2 = np.vstack(tempBuf[0:numberOfSegments])
    return tempBuf2


# Regex patterns and datetime formats for filename parsing
DateTimeformats = {
    'yyyymmdd_HHMMSS_fff': r'\d{8}_\d{6}_\d{3}',
    'yyyymmdd_HHMMSS': r'\d{8}_\d{6}',
    'AMAR': r'\d{8}T\d{6}',
    'SoundTrap': r'\d+\.(\d{12})',  # capture YYMMDDHHMMSS
    'yymmdd-HHMMSS.fff': r'\d{6}-\d{6}\.\d{3}',
    
}

DATE_FORMATS = {
    r'\d{8}_\d{6}_\d{3}': "%Y%m%d_%H%M%S_%f",
    r'\d{8}_\d{6}': "%Y%m%d_%H%M%S",
    r'\d{8}T\d{6}': "%Y%m%dT%H%M%S",
    # Soundtraps
    r'\d+\.(\d{12})': "%y%m%d%H%M%S",
    r'\d{6}-\d{6}\.\d{3}': "%y%m%d-%H%M%S.%f",
    r'\d{6}_\d{6}': "%y%m%d_%H%M%S",
}


class NoiseApp:
    """
    Run long-term noise analyses from local files or Google Cloud Storage audio.

    Parameters
    ----------
    soundFilePath : str
        Input location for audio files.
        - Local mode: path to a folder containing audio files.
        - GCS mode: a gs:// URI prefix (for example gs://bucket/path/prefix).
        Supported file types are discovered by extension.
    ProjName : str
        Project name used to build output HDF5 filename(s).
    DepName : str
        Deployment/instrument group name written as the top-level HDF5 group.
    DatabaseLoc : str
        Output directory where HDF5 file(s) are created.
    Si : float or str, default -184
        Hydrophone sensitivity input.
        - float: scalar sensitivity in dB re 1 V/uPa.
        - str: path to a CSV where first column is frequency (Hz) and second
          column is sensitivity in dB (interpreted using Si_units).
    clipFileSec : float, default 0
        Seconds to trim from the start of each audio file before analysis. This is useful for soundtraps that include 
        three seconds of calibration tones at the beginning of each file. Set to 0 to disable clipping.
    channel : int, default 0
        Zero-based channel index to analyze for multichannel audio.
    r : float, default 0.5
        Fractional overlap between adjacent FFT windows. Typical range is
        0 <= r < 1.
    winname : str, default 'Hann'
        Window label recorded in metadata. Current PSD path uses a Hann window
        for spectrogram computation.
    lcut : float or None, default None
        Low-frequency analysis limit in Hz. If None, defaults to 0.0 Hz.
    hcut : float or None, default None
        High-frequency analysis limit in Hz. If None, defaults to Nyquist
        (fs / 2).
    aveSec : float, default 60
        Time-bin duration in seconds used to average PSD frames before writing
        metrics.
    pref : float, default 1
        Reference pressure used in dB calculations (typically 1 uPa).
    rmDC : bool, default True
        If True, remove mean value from each streamed block before spectral
        analysis.
    legacy_mode : str or None, default None
        Optional legacy behavior.
        - None: standard processing.
        - 'pamguide': apply PAMGuide-compatible normalization path where used.
    Si_units : str, default 'V/µPa'
        Units for sensitivity input Si.
        Accepted values:
        - 'V/µPa', 'V/μPa', 'V/uPa', 'V per µPa', 'V per μPa'
        - 'V/Pa', 'V per Pa'
    split_hdf5_by_day : bool, default True
        If True, create one HDF5 file per date inferred from filenames.
        If False, write all data to one HDF5 file.
    existing_deployment_mode : {'error', 'skip', 'overwrite'}, default 'error'
        Behavior when DepName already exists in an output file and contains
        datasets.
        - 'error': raise ValueError.
        - 'skip': skip analysis for that target file.
        - 'overwrite': delete and recreate the deployment group.
    calibration_info : dict or Any, default None
        Optional calibration metadata saved into HDF5 parameter attributes.
        Dict keys are sanitized for attribute names.
    fixed_lat : float or None, default None
        Fixed latitude assigned to every output time bin.
    fixed_lon : float or None, default None
        Fixed longitude assigned to every output time bin.
        Note: fixed_lat and fixed_lon must be provided together.
    location_csv : str or None, default None
        Optional CSV path with timestamped positions for nearest-time lookup.
        Mutually exclusive with fixed_lat/fixed_lon.
    location_time_col : str, default 'date'
        Timestamp column name in location_csv.
    location_lat_col : str, default 'latitude'
        Latitude column name in location_csv.
    location_lon_col : str, default 'longitude'
        Longitude column name in location_csv.
    tol_method : {'psd_sum', 'ANSI', 'legacy', 'ansi_filterbank'}, default 'psd_sum'
        One-third-octave calculation method.
        - 'psd_sum' or 'legacy': integrate PSD over band bins (internal
          normalized label: 'psd_sum').
        - 'ANSI' or 'ansi_filterbank': ANSI edge-based integration (internal
          normalized label: 'ANSI').
    tol_order : int, default 3
        Third-octave filter order metadata value retained for compatibility and
        provenance.

    Notes
    -----
    - Date/time parsing is inferred from supported filename patterns and used
      for output timestamping and optional per-day file splitting.
    - If sensitivity is frequency-dependent (CSV), sensitivity is interpolated
      onto the analysis frequency grid before converting from V^2/Hz to
      uPa^2/Hz.
    """

    def __init__(self, soundFilePath, ProjName, DepName, DatabaseLoc,
                 Si=-184, clipFileSec=0, channel=0, r=0.5,
                 winname='Hann', lcut=None, hcut=None, aveSec=60,
                 pref=1, rmDC=True, legacy_mode = None, Si_units='V/µPa',
                 split_hdf5_by_day=True,
                 existing_deployment_mode='error',
                 calibration_info=None,
                 fixed_lat=None,
                 fixed_lon=None,
                 location_csv=None,
                 location_time_col='date',
                 location_lat_col='latitude',
                 location_lon_col='longitude',
                 tol_method="psd_sum", # "psd_sum" (current) or "ANSI"
                 tol_order=3):
        """
        Create long-term noise metrics from audio files (local folder or GCS).
        """
        # Inputs
        self.soundFilePath = soundFilePath
        self.ProjName = ProjName
        self.DepName = DepName
        self.DatabaseLoc = DatabaseLoc
        
        # Make it act like PAMGuide by calculating the xxxx
        self.legacy_mode = legacy_mode  # None or "pamguide"
        
        # Different ways of calculating third octave levels
        self.tol_method = self._normalize_tol_method(tol_method)
        self.tol_order  = int(tol_order)  # 3 matches PAMGuide’s default approach

        # Calibration can be scalar dB re 1 V/µPa or a CSV path with [Hz, dB]
        self.Si_source = Si
        self.Si = Si
        self.Si_units = Si_units
        if isinstance(self.Si, str):
            self.Si = pd.read_csv(self.Si)
        self.calibration_info = calibration_info

        # Optional geolocation inputs: fixed point OR timestamped CSV
        self.fixed_lat = float(fixed_lat) if fixed_lat is not None else None
        self.fixed_lon = float(fixed_lon) if fixed_lon is not None else None
        self.location_csv = location_csv
        self.location_time_col = location_time_col
        self.location_lat_col = location_lat_col
        self.location_lon_col = location_lon_col
        if (self.fixed_lat is None) ^ (self.fixed_lon is None):
            raise ValueError("Provide both fixed_lat and fixed_lon together, or neither.")
        if self.location_csv and self.fixed_lat is not None:
            raise ValueError("Provide either fixed_lat/fixed_lon OR location_csv, not both.")
        self._location_df = self._load_location_csv() if self.location_csv else None

        # Analysis settings
        self.clipFileSec = clipFileSec
        self.channel = channel
        self.r = r
        self.winname = winname
        self.lcut = lcut
        self.hcut = hcut
        self.aveSec = aveSec
        self.pref = pref
        self.rmDC = rmDC
        self.split_hdf5_by_day = bool(split_hdf5_by_day)
        valid_existing_modes = {'error', 'skip', 'overwrite'}
        if existing_deployment_mode not in valid_existing_modes:
            raise ValueError(
                f"existing_deployment_mode must be one of {sorted(valid_existing_modes)}; "
                f"got '{existing_deployment_mode}'"
            )
        self.existing_deployment_mode = existing_deployment_mode

        # Derived/initialized later
        self.fs = None
        self.N = None
        self.overlap = 0
        self.step = 0
        self.welch = None
        self.f = None
        self.M_uPa = None
        self.freqCal = None
        self.flowInd = 0
        self.fhighInd = 0
        self.DatePattern = None
        self.DateFormat = None
        self.audiofiles = None

        # Precomputed params containers
        self.decPrms = None
        self.TolPrms = None
        self.HbrdMlDec = None

        # HDF5 bookkeeping
        self.fullPath = None
        self.DateRun = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        # Temp and GCS
        self.temp_dir = tempfile.mkdtemp(prefix="noiseapp_")
        self._tmp_paths_to_delete = []
        self._gcs_storage_client = None  # cached Client
        
        
        
        
       # PAMGUide helpers 
    def _normalize_tol_method(self, method):
        """Normalize user-facing third-octave method labels to internal names."""
        if method is None:
            return "psd_sum"

        normalized = str(method).strip().lower()
        aliases = {
            "psd_sum": "psd_sum",
            "legacy": "psd_sum",
            "ansi": "ANSI",
            "ansi_filterbank": "ANSI",
        }

        if normalized not in aliases:
            raise ValueError(
                "tol_method must be one of {'psd_sum', 'ANSI'} "
                "(aliases: 'legacy', 'ansi_filterbank'). "
                f"Got '{method}'."
            )

        return aliases[normalized]

    def _pamguide_alpha_for_window(self):
        """Return the coherent gain term used by PAMGuide for the chosen window."""
        w = str(self.winname).lower()
        if w in ("hann", "hanning"):
            return 0.5
        if w == "hamming":
            return 0.54
        if w == "blackman":
            return 0.42
        if w in ("none", "rect", "rectangular"):
            return 1.0
        raise ValueError(f"Unsupported window for PAMGuide legacy: {self.winname}")
    
    def _pamguide_B(self, window):
        """Compute PAMGuide's window normalization factor for PSD correction."""
        alpha = self._pamguide_alpha_for_window()
        ww = window.astype(np.float64) / alpha
        return (1.0 / float(self.N)) * np.sum(ww * ww)

    def _match_column_name(self, df, preferred, aliases=None):
        """Resolve a dataframe column by exact/case-insensitive/normalized matching."""
        aliases = aliases or []
        candidates = [preferred] + list(aliases)

        for cand in candidates:
            if cand in df.columns:
                return cand

        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for cand in candidates:
            key = str(cand).strip().lower()
            if key in lower_map:
                return lower_map[key]

        normalize = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
        norm_map = {normalize(c): c for c in df.columns}
        for cand in candidates:
            key = normalize(cand)
            if key in norm_map:
                return norm_map[key]

        raise ValueError(
            f"Could not find column '{preferred}' in location CSV. Available columns: {list(df.columns)}"
        )

    def _load_location_csv(self):
        """Load timestamped lat/lon CSV and normalize to UTC timestamps."""
        df = pd.read_csv(self.location_csv)
        if df.empty:
            raise ValueError(f"Location CSV is empty: {self.location_csv}")

        time_col = self._match_column_name(
            df,
            self.location_time_col,
            aliases=['datetime', 'timestamp', 'time', 'DateTime', 'DATE']
        )
        lat_col = self._match_column_name(
            df,
            self.location_lat_col,
            aliases=['lat', 'Latitude', 'LAT', 'latitiude']
        )
        lon_col = self._match_column_name(
            df,
            self.location_lon_col,
            aliases=['lon', 'long', 'Longitude', 'LON', 'lng']
        )

        loc = pd.DataFrame({
            'timestamp': pd.to_datetime(df[time_col], errors='coerce', utc=True),
            'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
            'longitude': pd.to_numeric(df[lon_col], errors='coerce'),
        })
        loc = loc.dropna(subset=['timestamp', 'latitude', 'longitude']).sort_values('timestamp')
        if loc.empty:
            raise ValueError(
                f"No valid rows found in location CSV after parsing time/lat/lon: {self.location_csv}"
            )
        return loc.reset_index(drop=True)

    def _resolve_lat_lon_for_times(self, dt_bins):
        """Return (lat, lon) Nx1 arrays for bins from fixed point or nearest CSV row."""
        if not dt_bins:
            return None, None

        n = len(dt_bins)
        if self.fixed_lat is not None and self.fixed_lon is not None:
            return (
                np.full((n, 1), float(self.fixed_lat), dtype=np.float64),
                np.full((n, 1), float(self.fixed_lon), dtype=np.float64),
            )

        if self._location_df is None:
            return None, None

        q = pd.DataFrame({
            'orig_idx': np.arange(n, dtype=int),
            'timestamp': pd.to_datetime(pd.Series(dt_bins), utc=True),
        }).sort_values('timestamp')

        merged = pd.merge_asof(
            q,
            self._location_df,
            on='timestamp',
            direction='nearest'
        ).sort_values('orig_idx')

        lat = merged['latitude'].to_numpy(dtype=np.float64).reshape(-1, 1)
        lon = merged['longitude'].to_numpy(dtype=np.float64).reshape(-1, 1)
        return lat, lon


    # ---------- GCS helpers (single, canonical set) ----------
    def _get_gcs_client(self):
        """Return a cached google.cloud.storage.Client, creating it once."""
        if not _HAS_GCS:
            raise ImportError("google-cloud-storage not installed. pip install google-cloud-storage")
        if self._gcs_storage_client is None:
            self._gcs_storage_client = storage.Client()
        return self._gcs_storage_client

    def _is_gcs_path(self, path: str) -> bool:
        """Return True when the provided path uses the gs:// URI scheme."""
        return isinstance(path, str) and path.startswith("gs://")

    def _parse_gs_uri(self, uri: str):
        """Split a gs:// URI into bucket name and object key."""
        # urlparse('gs://bucket/prefix/file.wav') -> ('bucket', 'prefix/file.wav')
        p = urlparse(uri)
        return p.netloc, p.path.lstrip('/')

    def _list_audio_inputs(self):
        """Return a sorted list of local paths OR gs:// URIs to process."""
        exts = {'.wav', '.aif', '.aiff', '.flac', '.ogg', '.caf'}
        if self._is_gcs_path(self.soundFilePath):
            bucket, prefix = self._parse_gs_uri(self.soundFilePath)
            client = self._get_gcs_client()
            blobs = client.list_blobs(bucket, prefix=prefix)
            uris = [f"gs://{bucket}/{b.name}"
                    for b in blobs
                    if os.path.splitext(b.name)[1].lower() in exts and not b.name.endswith('/')]
            return sorted(uris)
        else:
            if not os.path.isdir(self.soundFilePath):
                raise FileNotFoundError(f"Local folder not found: {self.soundFilePath}")
            entries = [f for f in os.listdir(self.soundFilePath)
                       if os.path.isfile(os.path.join(self.soundFilePath, f))]
            if not entries:
                return []
            files = [
                os.path.join(self.soundFilePath, entry)
                for entry in entries
                if os.path.splitext(entry)[1].lower() in exts
            ]
            return sorted(files)

    def _download_to_temp(self, uri_or_path: str, tmpdir: str) -> str:
        """If gs://, download to tmpdir and return local path; else return as-is."""
        if not self._is_gcs_path(uri_or_path):
            return uri_or_path
        bucket, key = self._parse_gs_uri(uri_or_path)
        local = os.path.join(tmpdir, os.path.basename(key))
        client = self._get_gcs_client()
        client.bucket(bucket).blob(key).download_to_filename(local)
        return local

    def _download_if_gcs(self, uri: str) -> str:
        """If gs://, download to temp and return local path; else return original."""
        if not self._is_gcs_path(uri):
            return uri
        bucket, key = self._parse_gs_uri(uri)
        client = self._get_gcs_client()
        blob = client.bucket(bucket).blob(key)
        local = os.path.join(self.temp_dir, os.path.basename(key))
        blob.download_to_filename(local)
        self._tmp_paths_to_delete.append(local)
        return local
    # ---------------------------------------------------------

    def _date_key_from_name(self, name: str): #XXX TEMP
            """Extract a file date and datetime from the filename or fall back to mtime."""
            base = os.path.basename(name)
        
            if self.DatePattern and self.DateFormat:
                m = re.search(self.DatePattern, base)
                if m:
                    # Use first capture group if it exists, otherwise full match
                    dt_str = m.group(1) if m.lastindex else m.group(0)
        
                    try:
                        dt = datetime.strptime(dt_str, self.DateFormat)
                        return dt.date(), dt
                    except ValueError as e:
                        print(f"Datetime parse failed for '{dt_str}' with format '{self.DateFormat}': {e}")
        
            # Fallback: local file mtime (UTC date)
            try:
                ts = os.path.getmtime(name)
                dt = datetime.utcfromtimestamp(ts)
                return dt.date(), dt
            except Exception:
                return None, None

    def _start_new_hdf5_for_date(self, day: datetime.date):
        """Create a fresh HDF5 for a day or for the full run."""
        if self.split_hdf5_by_day:
            projName = f"{self.ProjName}_{day.strftime('%Y%m%d')}.h5"
        else:
            projName = f"{self.ProjName}.h5"
        fullPath = os.path.join(self.DatabaseLoc, projName)
        self.fullPath = fullPath
        return self.initilize_HDF5(fullPath, projName)

    def get_datetime_format(self, filename):
        """Detect the first supported datetime pattern present in a filename."""
        for date_pattern, date_format in DATE_FORMATS.items():
            match = re.search(date_pattern, filename)
            if match:
                self.DatePattern = date_pattern
                self.DateFormat = date_format
                return date_pattern, date_format
        return None, None

    def initilize_HDF5(self, fullPath, projName):
        """Create or append a deployment group and store run/parameter metadata."""
        # Ensure output directory exists
        os.makedirs(self.DatabaseLoc, exist_ok=True)

        self.fullPath = fullPath
        if isinstance(self.Si, pd.DataFrame):
            calibration_input_type = "frequency_curve_csv"
            calibration_points = int(len(self.Si))
            calibration_value_db = np.nan
        else:
            calibration_input_type = "scalar_db"
            calibration_points = 0
            calibration_value_db = float(self.Si)

        if self.fixed_lat is not None and self.fixed_lon is not None:
            location_mode = "fixed_point"
        elif self._location_df is not None:
            location_mode = "csv_track"
        else:
            location_mode = "none"

        metaVals = {
            "channel": self.channel,
            "r": self.r,
            "fs": self.fs,
            "N": self.N,
            "winname": self.winname,
            "lcut": self.lcut,
            "hcut": self.hcut,
            "overlap": self.overlap,
            "step": self.step,
            "Channel": self.channel,
            "aveSec": self.aveSec,
            "welch": self.welch,
            'rmDCoffset': self.rmDC,
            "DateRun": self.DateRun,
            "calibration_input_type": calibration_input_type,
            "calibration_units": str(self.Si_units),
            "calibration_source": str(self.Si_source),
            "calibration_points": calibration_points,
            "calibration_value_db": calibration_value_db,
            "location_mode": location_mode,
            "location_csv": str(self.location_csv) if self.location_csv else "",
            "location_time_col": str(self.location_time_col),
            "location_lat_col": str(self.location_lat_col),
            "location_lon_col": str(self.location_lon_col),
            "fixed_lat": np.nan if self.fixed_lat is None else float(self.fixed_lat),
            "fixed_lon": np.nan if self.fixed_lon is None else float(self.fixed_lon),
            "third_octave_method": str(self.tol_method),
            "third_octave_filter_order": int(self.tol_order),
        }

        if isinstance(self.calibration_info, dict):
            for raw_key, raw_value in self.calibration_info.items():
                safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(raw_key)).strip("_").lower()
                if not safe:
                    continue
                metaVals[f"calibration_info_{safe}"] = str(raw_value)
        elif self.calibration_info is not None:
            metaVals["calibration_info_note"] = str(self.calibration_info)
        print('Preparing HDF5 File %s' % projName)
        with self._open_hdf5_with_retry(fullPath, "a") as f:
            if self.DepName in f:
                instrument_group = f[self.DepName]
                existing_datasets = [name for name in instrument_group.keys() if name != "Parameters"]
                if existing_datasets:
                    if self.existing_deployment_mode == 'skip':
                        print(
                            f"Deployment group '{self.DepName}' already exists in {projName}. Skipping analysis."
                        )
                        return False
                    if self.existing_deployment_mode == 'overwrite':
                        del f[self.DepName]
                        instrument_group = f.create_group(self.DepName)
                        params_group = instrument_group.create_group("Parameters")
                    else:
                        raise ValueError(
                            f"Deployment group '{self.DepName}' already exists in {projName}. "
                            "Set existing_deployment_mode to 'skip' or 'overwrite' to change this behavior."
                        )
                else:
                    params_group = instrument_group.require_group("Parameters")
            else:
                instrument_group = f.create_group(self.DepName)
                params_group = instrument_group.create_group("Parameters")

            for k, v in metaVals.items():
                params_group.attrs[k] = v
        return True

    def _open_hdf5_with_retry(self, file_path, mode, retries=5, delay_sec=0.5):
        """Open an HDF5 file with a few retries for transient network-path failures."""
        last_exc = None
        for attempt in range(retries):
            try:
                return h5py.File(file_path, mode)
            except (FileNotFoundError, FileExistsError, OSError) as exc:
                last_exc = exc
                if attempt == retries - 1:
                    break
                time_module.sleep(delay_sec * (attempt + 1))

        raise OSError(
            f"Unable to open HDF5 file after {retries} attempts: {file_path}"
        ) from last_exc

    def _iter_blocks(self, y, block_sec: float):
        """Yield (y_block, start_sample) chunks of y with length ≈ block_sec."""
        if block_sec is None or block_sec <= 0:
            yield y, 0
            return
        L = len(y)
        B = int(block_sec * self.fs)
        if B <= 0:
            B = len(y)
        for start in range(0, L, B):
            yield y[start:start + B], start

    def _interp_sensitivity_db_uPa(self):
        """
        Returns sensitivity in dB re 1 V/µPa at self.f (Hz).
        If self.Si is a DataFrame: first col=Hz, second col=sens_dB (re 1 V/µPa).
        If self.Si is numeric: broadcast across freq.
        """
        if isinstance(self.Si, pd.DataFrame):
            f_col = self.Si.columns[0]
            s_col = self.Si.columns[1]
            sens_db = np.interp(
                self.f,
                np.concatenate(([0.0], self.Si[f_col].values, [self.fs / 2])),
                np.concatenate(([self.Si[s_col].iloc[0]], self.Si[s_col].values, [self.Si[s_col].iloc[-1]]))
            )
        else:
            sens_db = np.full_like(self.f, float(self.Si), dtype=float)
        return sens_db

    def _build_M_uPa(self):
        """Convert sensitivity inputs into a linear transfer function in V/µPa."""
        sens_db = self._interp_sensitivity_db_uPa()
        u = str(self.Si_units).lower().replace('u', 'µ')
        if u in ('v/µpa', 'v/μpa', 'v per µpa', 'v per μpa'):
            return 10 ** (sens_db / 20.0)           # already V/µPa
        elif u in ('v/pa', 'v per pa'):
            return (10 ** (sens_db / 20.0)) / 1e6   # V/Pa → V/µPa
        else:
            raise ValueError(f"Unknown Si_units='{self.Si_units}'. Use 'V/µPa' or 'V/Pa'.")

    def _read_blocks_from_file(self, path, block_sec: float = 30.0, max_block_bytes: int = 64 * 1024 ** 2):
        """Stream one audio file in channel-selected float32 blocks sized for analysis."""
        with sf.SoundFile(path, 'r') as f:
            fs = int(f.samplerate)
            ch = int(f.channels)
            bps = np.dtype('float32').itemsize

            max_frames_by_mem = max(1, max_block_bytes // (bps * ch))
            frames_per_block = max(self.N, min(int(block_sec * fs), int(max_frames_by_mem)))

            total_frames = len(f)
            start = 0
            while start < total_frames:
                frames = min(frames_per_block, total_frames - start)
                f.seek(start)
                yb2d = f.read(frames=frames, dtype='float32', always_2d=True)
                yb = yb2d[:, self.channel] if yb2d.shape[1] > 1 else yb2d[:, 0]
                yield yb, start
                start += frames

    def prep_audio(self):
        """Discover inputs, infer time parsing, and initialize analysis parameters."""
        # List inputs (local or GCS)
        inputs = self._list_audio_inputs()
        self.audiofiles = inputs

        if not inputs:
            print(f"No audio files found under {self.soundFilePath}. Nothing to process.")
            return False
    
        # Determine date pattern from the first filename (basename only!)
        first_name = os.path.basename(inputs[0])
        self.DatePattern, self.DateFormat = self.get_datetime_format(first_name)
    
        # Probe samplerate
        with tempfile.TemporaryDirectory() as td:
            probe_path = self._download_to_temp(inputs[0], td)
            info = sf.info(probe_path)
            self.fs = int(info.samplerate)
    
        # ----- Analysis band defaults FIRST -----
        if self.lcut is None:
            self.lcut = 0.0
        if self.hcut is None:
            self.hcut = self.fs / 2.0
    
        # ----- FFT / spectrogram params (match local) -----
        # If hcut==fs/2 → N = fs → ~1 Hz bins
        self.N = min(self.fs, int(self.hcut * 2))
        self.overlap = int(np.ceil(self.N * self.r))
        self.step = self.N - self.overlap
    
        # Welch compress factor
        self.welch = self.aveSec * (self.fs / self.N) / (1 - self.r)
    
        # rFFT grid + calibration
        self.f = np.fft.rfftfreq(self.N, d=1.0 / self.fs)
        self.M_uPa = self._build_M_uPa()
    
        self.flowInd = np.searchsorted(self.f, self.lcut, side='left')
        self.fhighInd = np.searchsorted(self.f, min(self.hcut, self.fs / 2), side='right') - 1
    
        return True


    def run_analysis(self):
        """
        Stream inputs (local or GCS), compute PSD, average into aveSec bins,
        compute metrics, and write into HDF5 files, optionally split by date.
        """
        if not self.prep_audio():
            return
        
        # XXXX FIX THIS TO USER SPECIFIED WINDOW XXXX
        window = np.hanning(self.N).astype(np.float32)
        

        # Determine if PAMGUide version
        B_pg = None
        if self.legacy_mode == "pamguide":
            B_pg = self._pamguide_B(window)
            # Optional: store for metadata/debug
            self._B_pg = float(B_pg)


        current_date_key = None   # YYYYMMDD
        data_start = 0            # row cursor within active HDF5

        with tempfile.TemporaryDirectory() as tmproot:
            for inp in self.audiofiles:
                local_path = None
                try:
                    # Download if GCS
                    local_path = self._download_to_temp(inp, tmproot)

                    # Figure out the date for file rotation
                    file_date, file_ts_dt = self._date_key_from_name(local_path)
                    date_key = file_date.strftime("%Y%m%d") if file_date else "unknown"

                    # Rotate HDF5 when date changes, or initialize once for a single output file
                    should_rotate = self.fullPath is None
                    if self.split_hdf5_by_day:
                        should_rotate = should_rotate or (date_key != current_date_key)

                    if should_rotate:
                        current_date_key = date_key
                        initialized = self._start_new_hdf5_for_date(file_date or datetime.utcnow().date())
                        if not initialized:
                            return
                        data_start = 0 if self.split_hdf5_by_day else data_start
                        print(f"Writing to HDF5: {os.path.basename(self.fullPath)}")

                    print(os.path.basename(local_path))

                    all_t = []
                    all_psd = []

                    # ---- stream blocks (~30 s or memory-capped) ----
                    for yb, start_samp in self._read_blocks_from_file(
                            local_path, block_sec=30.0, max_block_bytes=32 * 1024 ** 2):

                        # optional clip at file start
                        extra_offset = 0.0
                        if self.clipFileSec and start_samp == 0:
                            clip_frames = int(self.clipFileSec * self.fs)
                            if clip_frames < len(yb):
                                yb = yb[clip_frames:]
                                extra_offset = self.clipFileSec
                            else:
                                continue

                        if self.rmDC:
                            yb = yb - np.mean(yb)

                        if len(yb) < self.N:
                            continue  # ensure full window

                        f, t, Sxx = scipy.signal.spectrogram(
                            yb, fs=self.fs, window=window, nperseg=self.N,
                            noverlap=self.overlap, nfft=self.N,
                            detrend=False, scaling='density', mode='psd'
                        )

                        # seconds from file start
                        t_abs = t + (start_samp / self.fs) + extra_offset

                        # Keep calibration grid aligned
                        if (self.f is None) or (len(self.f) != len(f)) or (not np.allclose(self.f, f)):
                            self.f = f.copy()
                            self.M_uPa = self._build_M_uPa()

                        # V²/Hz -> µPa²/Hz
                        newPss_V2Hz = Sxx.T.astype(np.float64, copy=False)   # (T,F)
                        M = self.M_uPa[None, :]
                        newPss_cal = newPss_V2Hz / (M ** 2)

                        if self.legacy_mode == "pamguide":
                            # Apply PAMGuide-legacy normalization consistently across all metrics
                            newPss_cal = newPss_cal * B_pg

                        all_t.append(t_abs)
                        all_psd.append(newPss_cal)

                    if not all_t:
                        continue

                    # ---- concatenate, bin to aveSec ----
                    Tsec = np.concatenate(all_t)       # (Ncols,)
                    PSD = np.vstack(all_psd)           # (Ncols, F)

                    delf = (self.f[1] - self.f[0]) if len(self.f) > 1 else (self.fs / self.N)
                    t0_sec = Tsec.min()
                    t_anchor = t0_sec - (t0_sec % self.aveSec)
                    bin_idx = ((Tsec - t_anchor) // self.aveSec).astype(int)

                    uniq = np.unique(bin_idx)
                    PSD_bin = np.zeros((len(uniq), PSD.shape[1]), dtype=float)
                    dt_bins = []
                    for j, b in enumerate(uniq):
                        m = (bin_idx == b)
                        PSD_bin[j, :] = np.nanmean(PSD[m, :], axis=0)
                        tc = (b + 0.5) * self.aveSec + t_anchor
                        if file_ts_dt is None:
                            dt_bins.append(datetime.utcfromtimestamp(0) + timedelta(seconds=float(tc)))
                        else:
                            # Anchor to the file's parsed start datetime so LTSA bins preserve
                            # the real time-of-day instead of collapsing each file to midnight.
                            dt_bins.append(file_ts_dt + timedelta(seconds=float(tc)))

                    ttISO = np.array([dt.strftime('%Y%m%dT%H%M%S') for dt in dt_bins])
                    lat_bins, lon_bins = self._resolve_lat_lon_for_times(dt_bins)

                    # ---- metrics ----
                    apsd_60 = 10.0 * np.log10(np.maximum(PSD_bin, 1e-30) / (self.pref ** 2))
                    milidec = np.round(self.calcHybridMilidecades(apsd_60), 2)
                    Broadband = np.round(self.calcBroadband(PSD_bin, delf), 2)
                    if self.tol_method == "ANSI":
                        TOL = np.round(self.calc13OctaveANSI(PSD_bin, B=1.0), 2)
                    else:
                        TOL = np.round(self.calc13Octave(PSD_bin, B=1.0), 2)
                    decadeBands = np.round(self.calcDecadeband(PSD_bin), 2)

                    # ---- write ----
                    self.writeDatatoHDF5(ttISO, 'DateTime', data_start=data_start, storage_mode='str')
                    self.writeDatatoHDF5(milidec, 'hybridMiliDecLevels', data_start=data_start)
                    self.writeDatatoHDF5(Broadband, 'broadband', data_start=data_start)
                    self.writeDatatoHDF5(TOL, 'thirdoct', data_start=data_start)
                    self.writeDatatoHDF5(decadeBands, 'decadeLevels', data_start=data_start)
                    if lat_bins is not None and lon_bins is not None:
                        self.writeDatatoHDF5(lat_bins, 'latitude', data_start=data_start)
                        self.writeDatatoHDF5(lon_bins, 'longitude', data_start=data_start)

                    if data_start == 0:
                        self.writeDatatoHDF5(self.HbrdMlDec['freqLims'], 'hybridDecFreqHz',
                                             data_start=0, max_rows=len(self.HbrdMlDec['freqLims']))
                        self.writeDatatoHDF5(self.TolPrms['fc'], 'thirdOctFreqHz',
                                             data_start=0, max_rows=len(self.TolPrms['fc']))
                        self.writeDatatoHDF5(self.decPrms['decade_edges'], 'decadeFreqHz',
                                             data_start=0, max_rows=len(self.decPrms['decade_edges']))

                    data_start += len(dt_bins)
                finally:
                    if local_path and self._is_gcs_path(inp) and os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass

        return

    def writeDatatoHDF5(self, new_data, data_type, data_start=0,
                        max_rows=None, storage_mode="float64", fill_value=np.nan):
        """
        Append-or-create dataset and auto-resize as needed.
        - Strings: 1-D variable length UTF-8
        - Numerics (vector): stored as 2-D (rows, 1)
        - Numerics (matrix): stored as 2-D (rows, cols)
        """
        if new_data is None:
            raise ValueError("new_data cannot be None.")

        arr = np.asarray(new_data)
        if arr.ndim == 0:
            arr = arr[None]

        if not self.fullPath:
            raise ValueError("HDF5 output path is not initialized. Run initilize_HDF5 before writing data.")

        output_dir = os.path.dirname(self.fullPath)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as exc:
                raise OSError(
                    f"Unable to create or access HDF5 output directory: {output_dir}"
                ) from exc

        is_string = (storage_mode == "str")
        is_vector = (arr.ndim == 1)
        nrows = int(arr.shape[0])
        ncols = 1 if is_vector else int(arr.shape[1])

        with self._open_hdf5_with_retry(self.fullPath, "a") as hdf:
            grp = hdf.require_group(self.DepName)

            # --- create if missing ---
            if data_type not in grp:
                init_rows = int(max((max_rows or 0), data_start + nrows))

                if is_string:
                    dt = h5py.string_dtype(encoding="utf-8")
                    chunk_len = max(1024, min(16384, nrows))
                    dset = grp.create_dataset(
                        data_type,
                        shape=(init_rows,),
                        maxshape=(None,),
                        chunks=(chunk_len,),
                        dtype=dt,
                        fillvalue="0000-00-00 00:00:00",
                    )
                else:
                    # numeric -> always 2-D
                    chunk_rows = max(64, min(4096, nrows))
                    dset = grp.create_dataset(
                        data_type,
                        shape=(init_rows, ncols),
                        maxshape=(None, ncols),
                        chunks=(chunk_rows, ncols),
                        dtype=storage_mode,
                        fillvalue=fill_value,
                    )
            else:
                dset = grp[data_type]
                # sanity: fixed column count must match
                if not is_string and (dset.ndim != 2 or int(dset.shape[1]) != ncols):
                    raise ValueError(
                        f"Column mismatch for '{data_type}': incoming {ncols}, dataset shape {dset.shape}"
                    )

            # --- grow rows if needed ---
            need_rows = int(data_start + nrows)
            if dset.shape[0] < need_rows:
                if dset.maxshape[0] is not None and need_rows > dset.maxshape[0]:
                    raise ValueError(
                        f"Dataset '{data_type}' not resizable. Existing {dset.shape}, need rows {need_rows}."
                    )
                # resize rows (1-D or 2-D)
                if dset.ndim == 1:
                    dset.resize((need_rows,))
                else:
                    dset.resize((need_rows, dset.shape[1]))

            # --- write ---
            if dset.ndim == 1:
                dset[data_start:data_start + nrows] = arr.astype(str)
            else:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                dset[data_start:data_start + nrows, :] = arr

    def calcBroadband(self, PssCropped, delf):
        """
        Broadband SPL from calibrated PSD (µPa²/Hz). PssCropped: shape (T, F).
        """
        delf = float(delf)
        total_power = np.sum(PssCropped, axis=1) * delf        # µPa² per time-bin
        rms = np.sqrt(np.maximum(total_power, 0.0))            # µPa
        return 20.0 * np.log10(np.maximum(rms, 1e-30) / self.pref)

    def calcDecadeband(self, PssCropped):
        """
        Calculate decade band levels.
        """
        if self.decPrms is None:
            decade_edges = np.logspace(
                np.floor(np.log10(self.lcut + 1)),
                np.ceil(np.log10(self.hcut) - 1),
                num=int(np.ceil(np.log10(self.hcut)) - np.floor(np.log10(self.lcut + 1))),
            )
            idxVals = np.zeros([len(decade_edges), 2])
            for ii in range(len(decade_edges)):
                idxVals[ii, 0] = np.searchsorted(self.f, decade_edges[ii], side='left')
                idxVals[ii, 1] = np.searchsorted(self.f, decade_edges[ii] * 10, side='right')

            self.decPrms = dict()
            self.decPrms['decade_edges'] = decade_edges
            self.decPrms['idxVals'] = idxVals

        decade_bands = np.zeros([self.decPrms['idxVals'].shape[0], PssCropped.shape[0]])
        for ii in range(self.decPrms['idxVals'].shape[0]):
            band_sum = np.sum(PssCropped[:, int(self.decPrms['idxVals'][ii, 0]):int(self.decPrms['idxVals'][ii, 1])],
                              axis=1)
            decade_bands[ii, :] = (10 * np.log10(band_sum / (self.pref ** 2)))
        decade_bands = decade_bands.T
        return decade_bands

    def calc13Octave(self, PssCropped, B):
        """
        Calculate third octave levels.
        """
        if self.TolPrms is None:
            low13band = max(25, self.lcut)
            lobandf = np.floor(np.log10(low13band))
            hibandf = np.ceil(np.log10(self.hcut))
            nband = int(10 * (hibandf - lobandf) + 1)

            fc = np.zeros(nband)
            fc[0] = 10 ** lobandf
            for i in range(1, nband):
                fc[i] = fc[i - 1] * (10 ** 0.1)

            fc = fc[(fc >= low13band) & (fc <= self.hcut)]
            nfc = len(fc)

            fb = fc * (10 ** -0.05)
            fb = np.append(fb, fc[-1] * (10 ** 0.05))

            if fb[-1] > self.hcut:
                fc = fc[:-1]
                fb = fb[:-1]
                nfc = len(fc)

            fli = np.zeros(len(fc))
            fui = np.zeros(len(fc))
            for i in range(nfc):
                fli[i] = np.searchsorted(self.f, fb[i], side='left')
                fui[i] = np.searchsorted(self.f, fb[i + 1], side='right') - 1

            self.TolPrms = dict()
            self.TolPrms['nfc'] = nfc
            self.TolPrms['fli'] = fli
            self.TolPrms['fui'] = fui
            self.TolPrms['fc'] = fc

        P13 = np.zeros((PssCropped.shape[0], self.TolPrms['nfc']))
        for i in range(self.TolPrms['nfc']):
            if self.TolPrms['fui'][i] >= self.TolPrms['fli'][i]:
                P13[:, i] = np.sum(PssCropped[:, int(self.TolPrms['fli'][i]):int(self.TolPrms['fui'][i]) + 1], axis=1)

        a13 = 10 * np.log10((1 / B) * P13 / (self.pref ** 2))
        return a13

    def calc13OctaveANSI(self, PssCropped, B=1.0):
        """
        Calculate ANSI one-third-octave levels from PSD by integrating exact
        ANSI base-10 band edges (f_c * 10^(+/-1/20)) over the FFT-bin support.
        """
        if len(self.f) < 2:
            raise ValueError("Frequency vector self.f must have at least two bins for ANSI third-octave.")

        if (self.TolPrms is None) or (self.TolPrms.get('method') != 'ANSI'):
            low13band = max(25.0, float(self.lcut))
            G10 = 10.0 ** 0.1
            edge_ratio = 10.0 ** 0.05

            # ANSI exact base-10 one-third-octave centers relative to 1000 Hz.
            k_min = int(np.floor(10.0 * np.log10(low13band / 1000.0))) - 2
            k_max = int(np.ceil(10.0 * np.log10(float(self.hcut) / 1000.0))) + 2
            k = np.arange(k_min, k_max + 1, dtype=int)
            fc = 1000.0 * (G10 ** k)
            fc = fc[(fc >= low13band) & (fc <= float(self.hcut))]

            if fc.size == 0:
                raise ValueError(
                    f"No ANSI third-octave centers found inside analysis band [{low13band}, {self.hcut}] Hz."
                )

            flo = fc / edge_ratio
            fhi = fc * edge_ratio

            # Clip upper edges to analysis Nyquist bound.
            fhi = np.minimum(fhi, float(self.hcut))

            self.TolPrms = {
                'method': 'ANSI',
                'nfc': int(fc.size),
                'fc': fc,
                'flo': flo,
                'fhi': fhi,
            }

        df = float(self.f[1] - self.f[0])
        bin_lo = np.maximum(0.0, self.f - 0.5 * df)
        bin_hi = self.f + 0.5 * df

        n_t = PssCropped.shape[0]
        n_b = self.TolPrms['nfc']
        P13 = np.zeros((n_t, n_b), dtype=float)

        for i in range(n_b):
            blo = float(self.TolPrms['flo'][i])
            bhi = float(self.TolPrms['fhi'][i])

            idx = np.where((bin_hi > blo) & (bin_lo < bhi))[0]
            if idx.size == 0:
                continue

            overlap = np.minimum(bin_hi[idx], bhi) - np.maximum(bin_lo[idx], blo)
            overlap = np.maximum(overlap, 0.0)
            if not np.any(overlap > 0.0):
                continue

            # Integrate PSD (uPa^2/Hz) over each ANSI band using overlap weights (Hz).
            P13[:, i] = np.sum(PssCropped[:, idx] * overlap[None, :], axis=1)

        a13 = 10.0 * np.log10(np.maximum((1.0 / float(B)) * P13 / (self.pref ** 2), 1e-30))
        return a13

    def calcHybridMilidecades(self, apsd, fcross=435.0):
        """
        apsd: (T, F) in dB re 1 µPa²/Hz on the SAME freq grid as self.f.
        Hybrid convention (erratum): use 1 Hz bands up to 434 Hz,
        then millidecade bands beginning at 435 Hz.
        Returns: (T, nBands) band-averaged spectral density (dB re 1 µPa²/Hz).
        """
        df = (self.f[1] - self.f[0]) if len(self.f) > 1 else (self.fs / self.N)
    
        if self.HbrdMlDec is None:
            # ----- linear (per-FFT-bin) below fcross -----
            k_cross = int(np.searchsorted(self.f, fcross, side='left'))
            k_cross = max(1, k_cross)
    
            low = np.zeros((k_cross, 3), dtype=float)
            low[:, 1] = self.f[:k_cross]
            low[:, 0] = np.maximum(0.0, low[:, 1] - 0.5 * df)
            low[:, 2] = low[:, 1] + 0.5 * df
    
            # ----- log-spaced milli-decades above fcross -----
            logbands = get_band_table(
                fft_bin_size=df,
                bin1_center_frequency=0,
                fs=int(min(self.fs, int(self.hcut * 2))),
                base=10,
                bands_per_division=1000,
                first_output_band_center_frequency=fcross,
                use_fft_res_at_bottom=False
            )
    
            # Keep log bands from the transition center upward.
            logbands = logbands[logbands[:, 1] >= fcross]
    
            bands = np.vstack([low, logbands]) if logbands.size > 0 else low
            self.HbrdMlDec = {'freqLims': bands}
    
        bands = self.HbrdMlDec['freqLims']
        T = apsd.shape[0]
        out = np.full((T, bands.shape[0]), np.nan, dtype=float)

        # Treat each FFT line as a finite-width bin so band edges can include
        # proportional contributions from boundary bins.
        bin_lo = self.f - 0.5 * df
        bin_hi = self.f + 0.5 * df
    
        for i, (flo, fcen, fhi) in enumerate(bands):
            overlap_hz = np.minimum(bin_hi, fhi) - np.maximum(bin_lo, flo)
            overlap_hz = np.maximum(overlap_hz, 0.0)
            idx = np.where(overlap_hz > 0.0)[0]

            if idx.size == 0:
                k = int(np.clip(np.searchsorted(self.f, fcen, side='left'), 0, len(self.f) - 1))
                out[:, i] = apsd[:, k]
            elif idx.size == 1:
                out[:, i] = apsd[:, idx[0]]
            else:
                w = overlap_hz[idx]
                p_lin = np.nansum((10.0 ** (apsd[:, idx] / 10.0)) * w[None, :], axis=1)
                bw = max(float(np.sum(w)), df)
                avg_density = p_lin / bw                                       # µPa²/Hz
                out[:, i] = 10.0 * np.log10(np.maximum(avg_density, 1e-30) / (self.pref ** 2))
    
        return np.round(out, 2)


    def welchIt(self, PssCropped, tt):
        """
        Welch compress as per Merchant paper.
        """
        rA, cA = map(int, PssCropped.shape)
        lout = int(np.ceil(rA / self.welch))
        AWelch = np.zeros([lout, cA])
        AWelch[0, :] = PssCropped[0, :]
        tint = ((1 - self.r) * self.N / self.fs)
        tcompressed = np.linspace(tt[0], tt[-1], num=lout)
        for ii in range(lout):
            stt = tt[0] + (ii * tint * self.welch)
            ett = stt + (self.welch * tint)
            tidxs = np.where(np.logical_and(tt >= stt, tt < ett))
            nowA = np.mean(PssCropped[tidxs, :], axis=1)
            AWelch[ii, ] = nowA
            tcompressed[ii] = stt + self.welch * tint / 2
        return [AWelch, tcompressed]

    def load_data(self, file_path):
        """
        Load audio data from a file and handle multi-channel selection.
        """
        try:
            data, fs = sf.read(file_path)
            if data.ndim > 1:
                data = data[:, self.channel]
            if data.dtype.kind == 'i':
                max_val = np.iinfo(data.dtype).max
                data = data / max_val
            self.soundFilePath = file_path
            return data, fs
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None, None

    def selftest_calibration(self, file_index: int = 0,
                             max_duration_sec: float = None,
                             waveform_is_pressure: bool = False,
                             expect_tolerance_db: float = 2.0,
                             synthetic_sens_db: float = 6.0,
                             autoguess_units: bool = True):
        """
        Memory-safe calibration sanity test on a single file.
        """
        # discover one file (supports GCS)
        if self.audiofiles is None:
            self.audiofiles = self._list_audio_inputs()

        if not self.audiofiles:
            print(f"No audio files found under {self.soundFilePath}. Self-test skipped.")
            return None

        test_uri = self.audiofiles[file_index]

        # read capped duration as float32 to avoid huge allocs
        with tempfile.TemporaryDirectory() as td:
            local_path = self._download_to_temp(test_uri, td)
            with sf.SoundFile(local_path, 'r') as f:
                fs0 = f.samplerate
                if max_duration_sec is None:
                    max_duration_sec = min(60.0, len(f) / fs0)
                frames_to_read = int(max_duration_sec * fs0)
                data = f.read(frames=frames_to_read, dtype='float32', always_2d=True)

        # channel select
        yy = data[:, self.channel] if data.ndim > 1 else data
        if self.rmDC:
            yy = yy - np.mean(yy)

        # analysis params
        if self.fs is None:
            self.fs = fs0
            self.N = min(self.fs, 2 ** 15)
            self.overlap = int(np.ceil(self.N * self.r))
            if (self.lcut is None) or (self.hcut is None):
                self.lcut = 0.0
                self.hcut = self.fs / 2.0
            self.welch = self.aveSec * (self.fs / self.N) / (1 - self.r)
            self.f = np.fft.rfftfreq(self.N, d=1.0 / self.fs)

            # Build sensitivity M(f)
            if isinstance(self.Si, pd.DataFrame):
                f_col = self.Si.columns[0]
                s_col = self.Si.columns[1]
                sens_all_db = self.Si[s_col].astype(float).values
                mean_db = float(np.nanmean(sens_all_db))
                if autoguess_units:
                    si_is_v_per_uPa = (mean_db < -120.0)
                else:
                    si_is_v_per_uPa = True
                sens_db = np.interp(self.f,
                                    np.concatenate(([0.0], self.Si[f_col].values, [self.fs / 2])),
                                    np.concatenate(([self.Si[s_col].iloc[0]], self.Si[s_col].values, [self.Si[s_col].iloc[-1]])))
                if si_is_v_per_uPa:
                    M_uPa = 10.0 ** (sens_db / 20.0)
                    csv_units = "V/µPa"
                else:
                    M_uPa = (10.0 ** (sens_db / 20.0)) / 1e6
                    csv_units = "V/Pa"
            else:
                sens_db = float(self.Si)
                M_uPa = np.full_like(self.f, 10.0 ** (sens_db / 20.0), dtype=float)
                csv_units = "V/µPa (scalar)"

            self.M_uPa = M_uPa
            self.flowInd = np.searchsorted(self.f, self.lcut, side='left')
            self.fhighInd = np.searchsorted(self.f, min(self.hcut, self.fs / 2), side='right') - 1
        else:
            csv_units = "cached"

        # spectrogram (V²/Hz) and Welch compress
        window = np.hanning(self.N).astype(np.float32)
        f_spec, t_spec, Sxx = scipy.signal.spectrogram(
            yy, fs=self.fs, window=window, nperseg=self.N, noverlap=self.overlap,
            nfft=self.N, detrend=False, scaling='density', mode='psd'
        )
        delf = (f_spec[1] - f_spec[0]) if len(f_spec) > 1 else (self.fs / self.N)
        tt = np.linspace(0, len(yy) / self.fs, num=Sxx.shape[1], dtype=float)

        newPss_V2Hz, newtt = self.welchIt(Sxx.T, tt)  # (T,F)

        # align sensitivity to spectrogram bins
        M_uPa_aligned = np.interp(f_spec, self.f, self.M_uPa,
                                  left=self.M_uPa[0], right=self.M_uPa[-1])

        # calibration: V²/Hz → µPa²/Hz
        newPss_cal = newPss_V2Hz / (M_uPa_aligned[None, :] ** 2)

        # PSD-integrated broadband SPL
        total_power_t = np.sum(newPss_cal, axis=1) * delf
        rms_psd = float(np.sqrt(np.mean(total_power_t)))
        Lp_psd_db = 20.0 * np.log10(max(rms_psd, 1e-30) / self.pref)

        # TD SPL if waveform already pressure
        if waveform_is_pressure:
            rms_td = float(np.sqrt(np.mean(yy ** 2)))
            Lp_time_db = 20.0 * np.log10(max(rms_td, 1e-30) / self.pref)
            delta_db = Lp_time_db - Lp_psd_db
        else:
            Lp_time_db = np.nan
            delta_db = np.nan

        # synthetic sensitivity check (expect ~ -synthetic_sens_db)
        M_gain = M_uPa_aligned * (10.0 ** (synthetic_sens_db / 20.0))
        newPss_cal_gain = newPss_V2Hz / (M_gain[None, :] ** 2)
        total_power_t_gain = np.sum(newPss_cal_gain, axis=1) * delf
        rms_psd_gain = float(np.sqrt(np.mean(total_power_t_gain)))
        Lp_psd_gain_db = 20.0 * np.log10(max(rms_psd_gain, 1e-30) / self.pref)
        shift_db = Lp_psd_gain_db - Lp_psd_db

        # report
        print("\n=== Calibration Self-Test ===")
        print(f"File: {os.path.basename(str(test_uri))}")
        print(f"CSV units assumed: {csv_units}")
        print(f"fs={self.fs:.1f} Hz, N={self.N}, delf={delf:.6f} Hz, "
              f"T_welch={newPss_cal.shape[0]}, F={newPss_cal.shape[1]}")
        if waveform_is_pressure:
            print(f"Time-domain SPL (µPa):             {Lp_time_db:.2f} dB re 1 µPa")
            print(f"PSD-integrated SPL (µPa):          {Lp_psd_db:.2f} dB re 1 µPa")
            print(f"Δ(TD - PSD):                       {delta_db:+.2f} dB "
                  f"(tol ±{expect_tolerance_db:.1f} dB)")
            if abs(delta_db) > expect_tolerance_db:
                print("WARNING: |TD - PSD| exceeds tolerance. Check windowing/averaging or units.")
        else:
            print("Waveform treated as VOLTAGE. Skipping TD vs PSD comparison.")
            print(f"PSD-integrated SPL (µPa):          {Lp_psd_db:.2f} dB re 1 µPa")

        print(f"Sensitivity +{synthetic_sens_db:.1f} dB → SPL shift {shift_db:+.2f} dB "
              f"(expect ≈ -{synthetic_sens_db:.1f} dB).")
        if abs(shift_db + synthetic_sens_db) > 0.6:
            print("WARNING: Synthetic shift deviates >0.6 dB. Check sensitivity units (V/µPa vs V/Pa).")

        sens_db_aligned = 20.0 * np.log10(M_uPa_aligned)
        print(f"Median sensitivity over band: {np.nanmedian(sens_db_aligned):.1f} dB re 1 V/µPa")

        return dict(
            file=str(test_uri), fs=int(self.fs), N=int(self.N), delf=float(delf),
            csv_units=csv_units, waveform_is_pressure=bool(waveform_is_pressure),
            Lp_time_db=float(Lp_time_db) if np.isfinite(Lp_time_db) else np.nan,
            Lp_psd_db=float(Lp_psd_db),
            delta_db=float(delta_db) if np.isfinite(delta_db) else np.nan,
            Lp_psd_gain_db=float(Lp_psd_gain_db),
            shift_db=float(shift_db)
        )


# -------------------- Plotting helpers --------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _extract_milidecade_psd(instrument_group):
    """
    Extract PSD and frequency for a single instrument_group, trimming
    to the valid time range based on DateTime.
    """
    # Read time stamps and find the usable data range
    time_stamps = instrument_group['DateTime'][:].astype(str)

    if "0000-00-00 00:00:00" in time_stamps:
        cutoff_idx = np.argmax(time_stamps == "0000-00-00 00:00:00")
        max_data_len = min(cutoff_idx, len(time_stamps))
    else:
        max_data_len = len(time_stamps)

    # Read frequency and data
    ff = instrument_group['hybridDecFreqHz'][:, 1]    # (F,)
    PSD = instrument_group['hybridMiliDecLevels'][0:max_data_len, :]  # (T, F)

    return PSD, ff

def plot_milidecade_statistics(instrument_group_or_list, pBands=[5, 25, 50, 75, 95],
                               title=None, save_path=None, dpi=300):
    """
    Plot milidecade statistics (SPD, percentiles, RMS) for one or more days.

    Parameters
    ----------
    instrument_group_or_list : h5py.Group or list of h5py.Group
        - Single HDF5 group (one file/day), OR
        - List/tuple of groups for multiple files/days.
    pBands : list of int
        Percentiles to plot.
    title : str, optional
        Figure title. If None, defaults to 'Empirical Probability Density (SPD)'.
    save_path : str or Path, optional
        Full path (including filename and extension, e.g. 'DS01_plot.png') to
        save the figure. PNG and JPG are both supported. If None, the figure
        is not saved.
    dpi : int
        Resolution in dots per inch when saving. Default 300.
    """
    # Normalize input to a list of groups
    if isinstance(instrument_group_or_list, (list, tuple)):
        groups = instrument_group_or_list
    else:
        groups = [instrument_group_or_list]

    if len(groups) == 0:
        raise ValueError("No instrument groups provided to plot_milidecade_statistics.")

    # --- 1. Collect PSDs and check frequency consistency across groups ---
    PSD_list = []
    ff_ref = None

    for g in groups:
        PSD_g, ff_g = _extract_milidecade_psd(g)
        PSD_list.append(PSD_g)

        if ff_ref is None:
            ff_ref = ff_g
        else:
            # sanity check: all groups must have same freq bins
            if not np.allclose(ff_ref, ff_g):
                raise ValueError("Frequency vectors differ between instrument groups; "
                                 "cannot safely concatenate PSDs.")

    # Concatenate along time axis: (T_total, F)
    PSD_all = np.vstack(PSD_list)
    ff = ff_ref

    # --- 2. Compute stats on concatenated PSD ---
    # RMS level (across time)
    RMSlevel = 10 * np.log10(np.mean(10 ** (PSD_all / 10), axis=0))

    # Percentiles across time
    p = np.percentile(PSD_all, pBands, axis=0)

    # Min/Max dB levels
    mindB = np.floor(np.min(PSD_all) / 10) * 10
    maxdB = np.ceil(np.max(PSD_all) / 10) * 10

    # --- 3. Empirical Probability Density (SPD) ---
    hind = 0.1
    dbvec = np.arange(mindB, maxdB + hind, hind)
    M = PSD_all.shape[0] - 1  # number of "intervals" in time

    d = np.zeros((len(dbvec) - 1, PSD_all.shape[1]))
    for i in range(PSD_all.shape[1]):
        d[:, i] = np.histogram(PSD_all[:, i], bins=dbvec, density=False)[0]

    # Scale to density per dB per "time"
    d /= (hind * M)
    d[d == 0] = np.nan

    # --- 4. Plot ---
    X, Y = np.meshgrid(ff + 1, dbvec[:-1])
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

#    c = ax0.pcolor(X, Y, d, shading='auto')
    c = ax0.pcolor(
        X, Y, d,
        shading='auto',
        norm=LogNorm(vmin=np.nanmax(d) * 1e-4, vmax=np.nanmax(d))
    )

    ax0.set_xscale('log')
    plt.colorbar(c, ax=ax0, label='Empirical Probability Density')

    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel('PSD (dB re 1 µPa²/Hz)')
    ax0.set_title(title if title is not None else 'Empirical Probability Density (SPD)')

    # Percentile curves
    cvals = [
        [0, 0, 0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ]
    for i, p_band in enumerate(pBands):
        ax0.semilogx(ff + 1, p[i, :], label=f'L{p_band}', color=cvals[i])

    # RMS curve
    ax0.semilogx(ff + 1, RMSlevel, label='RMS Level', color='m', linewidth=2)

    ax0.set_xlim((ff + 1).min(), (ff + 1).max())
    ax0.set_ylim(Y.min(), Y.max())
    ax0.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(str(save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_third_octave_bands(instrument_group_or_list, pBands=[5, 25, 50, 75, 95],
                            title=None, save_path=None, dpi=300):
    """
    Box-and-whisker plot of third-octave levels stored in HDF5 files.

    Parameters
    ----------
    instrument_group_or_list : h5py.Group or list of h5py.Group
        Single HDF5 deployment group, or a list/tuple of groups (e.g. multiple
        daily files) whose data will be concatenated before plotting.
    pBands : list of int
        Percentiles used to draw the whiskers and box edges.  Must have exactly
        5 values interpreted as [lower-whisker, Q1, median, Q3, upper-whisker].
        Default [5, 25, 50, 75, 95].
    title : str, optional
        Figure title.  Defaults to 'Third-Octave Band Levels' when None.
    save_path : str or Path, optional
        Full file path (including extension, e.g. 'DS01_tob.png') to save the
        figure at high resolution.  Supports PNG and JPG.  When None the figure
        is displayed interactively instead.
    dpi : int
        Resolution when saving.  Default 300.
    """
    if len(pBands) != 5:
        raise ValueError("pBands must have exactly 5 values: "
                         "[lower_whisker, Q1, median, Q3, upper_whisker].")

    # --- 1. Normalise input ---
    if isinstance(instrument_group_or_list, (list, tuple)):
        groups = instrument_group_or_list
    else:
        groups = [instrument_group_or_list]

    if not groups:
        raise ValueError("No instrument groups provided to plot_third_octave_bands.")

    # --- 2. Collect data, validate frequency consistency ---
    tol_list = []
    fc_ref = None

    for g in groups:
        # Trim sentinel rows using DateTime (same logic as milidecade extractor)
        raw_ts = g['DateTime'][:].astype(str)
        sentinel_idxs = np.where(raw_ts == "0000-00-00 00:00:00")[0]
        n_valid = int(sentinel_idxs[0]) if len(sentinel_idxs) > 0 else len(raw_ts)

        tol_g = g['thirdoct'][:n_valid, :]          # (T, F)

        # frequency centers stored as 1-D or (F,1) — flatten to 1-D
        fc_raw = g['thirdOctFreqHz'][:]
        fc_g = fc_raw.ravel()

        if fc_ref is None:
            fc_ref = fc_g
        else:
            if not np.allclose(fc_ref, fc_g):
                raise ValueError("Third-octave frequency vectors differ between "
                                 "instrument groups; cannot concatenate.")

        tol_list.append(tol_g)

    TOL_all = np.vstack(tol_list)   # (T_total, F)
    fc = fc_ref                      # (F,)
    n_bands = len(fc)

    # --- 3. Compute per-band statistics ---
    p_lo, p_q1, p_med, p_q3, p_hi = [
        np.percentile(TOL_all, pb, axis=0) for pb in pBands
    ]
    rms_levels = 10.0 * np.log10(np.mean(10.0 ** (TOL_all / 10.0), axis=0))

    # --- 4. Build uniform box widths in log space ---
    # Each box spans ±half the log-spacing of adjacent centre frequencies so
    # all boxes look the same width on a log-frequency axis.
    log_fc = np.log10(fc)
    if n_bands > 1:
        log_spacing = np.diff(log_fc)
        half_w = np.empty(n_bands)
        half_w[0]    = log_spacing[0]  / 2.0
        half_w[-1]   = log_spacing[-1] / 2.0
        half_w[1:-1] = (log_spacing[:-1] + log_spacing[1:]) / 4.0
        # Convert back to linear so patches land correctly on log axis
        x_lo = 10 ** (log_fc - half_w * 0.7)
        x_hi = 10 ** (log_fc + half_w * 0.7)
    else:
        x_lo = fc * 0.7
        x_hi = fc * 1.3

    # --- 5. Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    box_color = '#4878CF'
    med_color = '#444444'
    whisk_color = '#666666'
    rms_color = '#cc2f2f'

    for i in range(n_bands):
        xl, xr, xc = float(x_lo[i]), float(x_hi[i]), float(fc[i])

        # Box outline
        rect_x = [xl, xr, xr, xl, xl]
        rect_y = [p_q1[i], p_q1[i], p_q3[i], p_q3[i], p_q1[i]]
        ax.plot(rect_x, rect_y, color=box_color, linewidth=0.8, zorder=3)

        # Median line
        ax.plot([xl, xr], [p_med[i], p_med[i]],
                color=med_color, linewidth=1.2, linestyle='--', zorder=4)

        # Whiskers (vertical lines from box edge to whisker tip)
        ax.plot([xc, xc], [p_lo[i], p_q1[i]],
                color=whisk_color, linewidth=0.8, linestyle='--', zorder=2)
        ax.plot([xc, xc], [p_q3[i], p_hi[i]],
                color=whisk_color, linewidth=0.8, linestyle='--', zorder=2)

        # Whisker caps
        cap_w = (xr - xl) * 0.3
        for tip in [p_lo[i], p_hi[i]]:
            ax.plot([xc - cap_w, xc + cap_w], [tip, tip],
                    color=whisk_color, linewidth=0.8, zorder=2)

    ax.plot(fc, rms_levels, color=rms_color, linewidth=1.5, zorder=5, label='RMS Level')

    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('SPL (dB re 1 µPa²/Hz)')
    ax.set_title(title if title is not None else 'Third-Octave Band Levels')
    ax.set_xlim(float(x_lo[0]) * 0.9, float(x_hi[-1]) * 1.1)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='none', edgecolor=box_color,
              label=f'L{pBands[1]}–L{pBands[3]} (IQR)'),
        Line2D([0], [0], color=med_color, linewidth=1.2, linestyle='--',
               label=f'L{pBands[2]} (median)'),
        Line2D([0], [0], color=whisk_color, linewidth=0.8, linestyle='--',
               label=f'L{pBands[0]}–L{pBands[4]} (whiskers)'),
        Line2D([0], [0], color=rms_color, linewidth=1.5,
               label='RMS Level'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(str(save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    return fig


def _extract_ltsa_data(instrument_group):
    """
    Extract DateTime, PSD (hybridMiliDecLevels) and frequency vector
    from a single instrument_group, trimming any trailing '0000-00-00 00:00:00'.
    Returns:
        times : pandas.DatetimeIndex
        PSD   : 2D array (T, F)
        freq  : 1D array (F,)
    """
    # Raw timestamps as strings
    raw_ts = instrument_group['DateTime'][:].astype(str)

    # Trim at first sentinel, if present
    sentinel = "0000-00-00 00:00:00"
    sentinel_idx = np.where(raw_ts == sentinel)[0]
    if len(sentinel_idx) > 0:
        raw_ts = raw_ts[:sentinel_idx[0]]

    # Convert to datetime and drop NaT
    times = pd.to_datetime(raw_ts, errors='coerce')
    valid_mask = ~times.isna()
    times = times[valid_mask]

    # PSD and frequency
    PSD_full = instrument_group['hybridMiliDecLevels'][:len(raw_ts), :]
    PSD = PSD_full[valid_mask, :]
    freq = instrument_group['hybridDecFreqHz'][:, 1]

    return times, PSD, freq

def plot_ltsa(instrument_group_or_list,
              averaging_period='5min',
              title=None,
              save_path=None,
              dpi=300,
              freq_scaled=True,
              log_freq=False):
    """
    LTSA plot over one or more instrument groups (e.g., multiple days).

    Parameters
    ----------
    instrument_group_or_list : h5py.Group or list/tuple of h5py.Group
        One group for a single file/day or a list of groups for multiple files.
    averaging_period : str
        Pandas offset alias for time-averaging (e.g., '5min', '1min', '1H').
    title : str, optional
        Figure title. If None, no title is displayed.
    save_path : str or Path, optional
        Full file path (including extension, e.g. 'ltsa_plot.png') to save the
        figure at high resolution. Supports PNG and JPG. If None, the figure
        is displayed interactively instead.
    dpi : int
        Resolution in dots per inch when saving. Default 300.
    freq_scaled : bool
        If True, use actual frequency values as the y-coordinate (pcolormesh).
        If False, use an index-based y-axis with frequency labels only.
    log_freq : bool
        If True and freq_scaled is True, use a log scale for the frequency axis.
    """
    # Normalize input to list
    if isinstance(instrument_group_or_list, (list, tuple)):
        groups = instrument_group_or_list
    else:
        groups = [instrument_group_or_list]

    if not groups:
        raise ValueError("No instrument groups provided to plot_ltsa.")

    # ---- 1. Collect and concatenate data across all groups ----
    time_list = []
    PSD_list = []
    freq_ref = None

    for g in groups:
        times_g, PSD_g, freq_g = _extract_ltsa_data(g)
        if len(times_g) == 0:
            continue

        time_list.append(times_g)
        PSD_list.append(PSD_g)

        if freq_ref is None:
            freq_ref = freq_g
        else:
            if not np.allclose(freq_ref, freq_g):
                raise ValueError("Frequency vectors differ between instrument groups; "
                                 "cannot safely combine LTSA.")

    if not time_list:
        raise ValueError("No valid data found in supplied groups.")

    # Concatenate and sort by time
    times_all = pd.DatetimeIndex(np.concatenate([t.values for t in time_list]))
    PSD_all = np.vstack(PSD_list)

    order = np.argsort(times_all.values)
    times_all = times_all[order]
    PSD_all = PSD_all[order, :]

    freq = freq_ref
    n_freq = PSD_all.shape[1]

    # ---- 2. Build averaging bins ----
    start = times_all[0].floor(freq=averaging_period)
    end = times_all[-1].ceil(freq=averaging_period)
    time_edges = pd.date_range(start=start, end=end, freq=averaging_period)

    if len(time_edges) < 2:
        raise ValueError("Not enough data to form LTSA bins with "
                         f"averaging_period='{averaging_period}'.")

    # NT x NF internal grid
    n_bins = len(time_edges) - 1
    nlVals = np.full((n_freq, n_bins), np.nan)

    for i in range(n_bins):
        t0, t1 = time_edges[i], time_edges[i + 1]
        mask = (times_all >= t0) & (times_all < t1)
        if not mask.any():
            continue

        data_chunk = PSD_all[mask, :]  # (n_chunk, n_freq)
        # mean in linear space, then back to dB
        med_vals = 10 * np.log10(np.mean(10 ** (data_chunk / 10.0), axis=0))
        nlVals[:, i] = med_vals

    # Drop columns with all NaNs
    valid_cols = ~np.isnan(nlVals).all(axis=0)
    nlVals = nlVals[:, valid_cols]
    time_edges = time_edges[:-1][valid_cols]

    # Bin centers for plotting
    dt = (time_edges[1] - time_edges[0]) if len(time_edges) > 1 else pd.Timedelta(0)
    time_centers = time_edges + dt / 2

    # ---- 3. Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))

    if freq_scaled:

        # Use real frequency as the y coordinate
        t_num = mdates.date2num(time_centers)
        T_grid, F_grid = np.meshgrid(t_num, freq)

        pcm = ax.pcolormesh(T_grid, F_grid, nlVals,
                            shading='auto', cmap='cubehelix')

        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))

        # ----- Log-scaled frequency axis (base-10) -----
        if log_freq:
            # Matplotlib requires strictly positive values
            freq_pos = freq[freq > 0]
            if freq_pos.size == 0:
                raise ValueError("No positive frequency bins; cannot use log scale.")

            ax.set_yscale("log")
            ax.set_ylim(freq_pos.min(), freq.max())
        else:
            # Linear frequency axis
            ax.set_ylim(freq.min(), freq.max())


    else:
        # Index-based y axis with frequency labels only
        # Lowest frequency at bottom by using origin='lower'
        pcm = ax.imshow(nlVals,
                        aspect='auto',
                        origin='lower',
                        cmap='cubehelix')

        ax.set_xlabel("Time bin index")
        ax.set_ylabel("Frequency (Hz)")

        # Map indices to frequency labels
        n_rows = nlVals.shape[0]
        yticks = np.linspace(0, n_rows - 1, num=6)
        yfreqs = np.linspace(freq.min(), freq.max(), num=6)
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.round(yfreqs).astype(int))

        # For index-based mode, you might want simple integer ticks on x
        ax.set_xticks(np.linspace(0, nlVals.shape[1] - 1, num=10))

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r'RMS SPL (dB re 1 $\mu$Pa)')

    if title is not None:
        ax.set_title(title)

    ax.grid(False)
    ax.tick_params(direction='out', top=False, right=False)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(str(save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    return fig



from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Literal


BandType = Literal["third_octave", "decade"]
MetricType = Literal[
    "hybrid",
    "third_octave",
    "decade",
    "broadband",
    "latitude",
    "longitude",
]


def _resolve_h5_group_names(
    h5: h5py.File,
    group_name: Optional[Union[str, Iterable[str]]] = None,
    required_datasets: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Resolve deployment-group names from an HDF5 file.

    If group_name is None, return all top-level groups that contain the
    required datasets. If group_name is provided, validate that each named
    group exists and contains the requested datasets.
    """
    required = list(required_datasets or [])

    if group_name is None:
        names = [name for name in h5.keys() if isinstance(h5[name], h5py.Group)]
    elif isinstance(group_name, str):
        names = [group_name]
    else:
        names = list(group_name)

    resolved = []
    missing = []
    incomplete = []

    for name in names:
        if name not in h5:
            missing.append(name)
            continue

        group = h5[name]
        if not isinstance(group, h5py.Group):
            missing.append(name)
            continue

        missing_datasets = [dataset_name for dataset_name in required if dataset_name not in group]
        if missing_datasets:
            incomplete.append((name, missing_datasets))
            continue

        resolved.append(name)

    if group_name is not None:
        if missing:
            raise KeyError(f"Group(s) not found in HDF5: {missing}")
        if incomplete:
            missing_text = "; ".join(
                f"{name} missing {dataset_names}" for name, dataset_names in incomplete
            )
            raise ValueError(f"Requested group(s) missing required datasets: {missing_text}")

    if not resolved:
        if group_name is None:
            raise ValueError("No deployment groups with the required datasets were found in the HDF5 file.")
        raise ValueError("No valid deployment groups were resolved from the HDF5 file.")

    return resolved


def extract_bands_df(
    h5_path_or_paths: Union[str, Path, Iterable[Union[str, Path]]],
    band: BandType = "third_octave",
    averaging_period: str = "5min",
    group_name: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Extract third-octave or decade-band levels from YAWN/NoiseApp HDF5(s),
    average over user-specified time bins, and return a DataFrame with:
      index = time (DatetimeIndex)
      columns = frequency in Hz (float)
      values = band level (dB), averaged in linear space then converted to dB.

    Parameters
    ----------
    h5_path_or_paths
        A single .h5/.hdf5 file path, a folder path (all *.h5, *.hdf5 inside),
        or an iterable of file paths.
    band
        "third_octave" uses datasets: thirdOctFreqHz + thirdoct
        "decade"       uses datasets: decadeFreqHz   + decadeLevels
    averaging_period
        Pandas offset alias (e.g., "5min", "30min", "1H", "4H").
    group_name
        Deployment group name, iterable of names, or None to auto-read all
        deployment groups in each HDF5 file.

    Returns
    -------
    pd.DataFrame
    """
    paths = _normalize_h5_inputs(h5_path_or_paths)
    if not paths:
        raise ValueError("No HDF5 files found/provided.")

    if band == "third_octave":
        freq_key, data_key = "thirdOctFreqHz", "thirdoct"
    elif band == "decade":
        freq_key, data_key = "decadeFreqHz", "decadeLevels"
    else:
        raise ValueError("band must be 'third_octave' or 'decade'.")

    time_list: List[pd.DatetimeIndex] = []
    data_list: List[np.ndarray] = []
    freq_ref: Optional[np.ndarray] = None

    for p in paths:
        with h5py.File(p, "r") as h5:
            group_names = _resolve_h5_group_names(
                h5,
                group_name=group_name,
                required_datasets=["DateTime", freq_key, data_key],
            )

            for resolved_group_name in group_names:
                g = h5[resolved_group_name]

                # Time
                t = pd.to_datetime(g["DateTime"][()].astype(str), errors="raise")

                # Freqs (flatten (N,1)->(N,))
                f = np.asarray(g[freq_key][()], dtype=float).reshape(-1)

                # Levels (Ntime, Nfreq)
                X = np.asarray(g[data_key][()], dtype=float)

                if X.shape[0] != len(t):
                    raise ValueError(
                        f"{p} [{resolved_group_name}]: {data_key} rows {X.shape[0]} != DateTime {len(t)}"
                    )

                if X.shape[1] != len(f):
                    raise ValueError(
                        f"{p} [{resolved_group_name}]: {data_key} cols {X.shape[1]} != {freq_key} {len(f)}"
                    )

                if freq_ref is None:
                    freq_ref = f
                else:
                    if not np.allclose(freq_ref, f, atol=1e-10, rtol=0):
                        raise ValueError(
                            f"{p} [{resolved_group_name}]: frequency vector differs; cannot combine safely."
                        )

                time_list.append(pd.DatetimeIndex(t))
                data_list.append(X)

    if not time_list:
        raise ValueError("No deployment data found in the provided HDF5 file(s).")

    # Concatenate + sort
    times_all = pd.DatetimeIndex(np.concatenate([t.values for t in time_list]))
    X_all = np.vstack(data_list)

    order = np.argsort(times_all.values)
    times_all = times_all[order]
    X_all = X_all[order, :]

    # Build native-resolution df
    freqs = freq_ref.astype(float)  # type: ignore[union-attr]
    df = pd.DataFrame(X_all, index=times_all, columns=freqs)

    # Average in linear domain then convert back to dB
    lin = 10.0 ** (df / 10.0)
    lin_mean = lin.resample(averaging_period).mean()
    out = 10.0 * np.log10(lin_mean)

    # Drop empty bins
    out = out.dropna(how="all")

    # Sort columns by frequency
    out = out.reindex(sorted(out.columns), axis=1)

    return out


def _normalize_h5_inputs(
    h5_path_or_paths: Union[str, Path, Iterable[Union[str, Path]]]
) -> List[str]:
    """
    Accept:
      - file path
      - folder path (collect *.h5, *.hdf5)
      - iterable of file paths
    Return sorted list of file paths as strings.
    """
    if isinstance(h5_path_or_paths, (str, Path)):
        p = Path(h5_path_or_paths)
        if p.is_dir():
            files = list(p.glob("*.h5")) + list(p.glob("*.hdf5"))
            return sorted([str(x) for x in files])
        else:
            return [str(p)]
    else:
        return sorted([str(Path(x)) for x in h5_path_or_paths])


def export_metric_csv(
    h5_path_or_paths: Union[str, Path, Iterable[Union[str, Path]]],
    metric: MetricType,
    output_csv: Union[str, Path],
    group_name: Optional[Union[str, Iterable[str]]] = None,
) -> str:
    """
    Export one metric from one or more NoiseApp HDF5 files to a long-form CSV.

    Output columns always include:
      - datetime
      - source_file
      - deployment
      - metric
      - value

    Frequency columns are included where available:
      - frequency_hz
      - frequency_low_hz
      - frequency_high_hz
    """
    metric_map = {
        "hybrid": ("hybridMiliDecLevels", "hybridDecFreqHz"),
        "third_octave": ("thirdoct", "thirdOctFreqHz"),
        "decade": ("decadeLevels", "decadeFreqHz"),
        "broadband": ("broadband", None),
        "latitude": ("latitude", None),
        "longitude": ("longitude", None),
    }

    metric_key = str(metric).strip().lower()
    if metric_key not in metric_map:
        raise ValueError(f"Unsupported metric '{metric}'. Choose one of {sorted(metric_map.keys())}.")

    data_key, freq_key = metric_map[metric_key]
    paths = _normalize_h5_inputs(h5_path_or_paths)
    if not paths:
        raise ValueError("No HDF5 files found/provided.")

    frames = []
    for p in paths:
        with h5py.File(p, "r") as h5:
            required = ["DateTime", data_key]
            if freq_key is not None:
                required.append(freq_key)

            group_names = _resolve_h5_group_names(
                h5,
                group_name=group_name,
                required_datasets=required,
            )

            for resolved_group_name in group_names:
                g = h5[resolved_group_name]

                raw_ts = g["DateTime"][()].astype(str)
                sentinel_idxs = np.where(raw_ts == "0000-00-00 00:00:00")[0]
                n_valid = int(sentinel_idxs[0]) if len(sentinel_idxs) > 0 else len(raw_ts)

                times = pd.to_datetime(raw_ts[:n_valid], errors="coerce")
                valid_mask = ~times.isna()
                times = pd.DatetimeIndex(times[valid_mask])

                X = np.asarray(g[data_key][()], dtype=float)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X = X[:n_valid, :]
                X = X[np.asarray(valid_mask), :]

                if X.shape[0] != len(times):
                    raise ValueError(
                        f"{p} [{resolved_group_name}]: {data_key} rows {X.shape[0]} != valid DateTime {len(times)}"
                    )

                source_file = str(Path(p).name)
                n_time, n_freq = X.shape

                base = pd.DataFrame({
                    "datetime": np.repeat(times.astype(str).to_numpy(), n_freq),
                    "source_file": source_file,
                    "deployment": str(resolved_group_name),
                    "metric": metric_key,
                    "value": X.reshape(-1),
                })

                if freq_key is None:
                    base["frequency_hz"] = np.nan
                    base["frequency_low_hz"] = np.nan
                    base["frequency_high_hz"] = np.nan
                else:
                    farr = np.asarray(g[freq_key][()], dtype=float)

                    if farr.ndim == 2 and farr.shape[1] == 3:
                        flow = farr[:, 0]
                        fcenter = farr[:, 1]
                        fhigh = farr[:, 2]
                    else:
                        fcenter = farr.reshape(-1)
                        flow = np.full_like(fcenter, np.nan, dtype=float)
                        fhigh = np.full_like(fcenter, np.nan, dtype=float)

                    if len(fcenter) != n_freq:
                        raise ValueError(
                            f"{p} [{resolved_group_name}]: frequency bins {len(fcenter)} != {data_key} columns {n_freq}"
                        )

                    base["frequency_hz"] = np.tile(fcenter, n_time)
                    base["frequency_low_hz"] = np.tile(flow, n_time)
                    base["frequency_high_hz"] = np.tile(fhigh, n_time)

                frames.append(base)

    if not frames:
        raise ValueError("No data rows found to export.")

    out_df = pd.concat(frames, axis=0, ignore_index=True)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return str(out_path)


def export_all_metrics_csv(
    h5_path_or_paths: Union[str, Path, Iterable[Union[str, Path]]],
    output_dir: Union[str, Path],
    group_name: Optional[Union[str, Iterable[str]]] = None,
    metrics: Optional[Iterable[MetricType]] = None,
    file_prefix: str = "",
    skip_missing: bool = True,
) -> Dict[str, str]:
    """
    Export multiple metrics to separate CSV files.

    Returns a dictionary mapping metric name to output CSV path.
    """
    if metrics is None:
        selected_metrics: List[str] = [
            "hybrid",
            "third_octave",
            "decade",
            "broadband",
            "latitude",
            "longitude",
        ]
    else:
        selected_metrics = [str(m).strip().lower() for m in metrics]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: Dict[str, str] = {}
    for metric_name in selected_metrics:
        out_name = f"{file_prefix}{metric_name}.csv" if file_prefix else f"{metric_name}.csv"
        out_csv = out_dir / out_name

        try:
            exported_path = export_metric_csv(
                h5_path_or_paths=h5_path_or_paths,
                metric=metric_name,
                output_csv=out_csv,
                group_name=group_name,
            )
            exported[metric_name] = exported_path
        except (KeyError, ValueError):
            if skip_missing:
                continue
            raise

    if not exported:
        raise ValueError(
            "No metric CSVs were exported. Verify metric names, deployments, and dataset availability."
        )

    return exported


# -------------------- Script entry --------------------
if __name__ == "__main__":

    gsCloudLoc = "gs://swfsc-1/2024_CalCurCEAS/glider/audio_flac/sg680_CalCurCEAS_Sep2024"
    out_dir = r"C:\Users\pam_user\Documents\HybridMilliDaily"
    calib_csv = 'C:\\Users\\pam_user\\Downloads\\sg680_CalCurCEAS_Sep2024_sensitivity_2025-07-29.csv'
    app = NoiseApp(
        Si=calib_csv,
        soundFilePath=gsCloudLoc,
        ProjName='sg680_CalCurCEAS_Apr2022',
        DepName='SG680',
        DatabaseLoc=out_dir,
        rmDC=True,
        Si_units='V/µPa',
        split_hdf5_by_day=False
    )

    app.run_analysis()

    # Example plotting for either one combined HDF5 or many daily HDF5 files.
    if app.split_hdf5_by_day:
        plot_paths = sorted(Path(out_dir).glob(f"{app.ProjName}_*.h5"))
    else:
        plot_paths = [Path(app.fullPath)] if app.fullPath else []

    if not plot_paths:
        raise FileNotFoundError(f"No HDF5 files found for plotting in {out_dir}")

    with ExitStack() as stack:
        instrument_groups = []
        for plot_path in plot_paths:
            hdf_file = stack.enter_context(h5py.File(plot_path, 'r'))
            group_names = _resolve_h5_group_names(
                hdf_file,
                required_datasets=['DateTime', 'hybridMiliDecLevels', 'hybridDecFreqHz']
            )
            instrument_groups.extend(hdf_file[group_name] for group_name in group_names)

        plot_milidecade_statistics(instrument_groups)
        plot_ltsa(instrument_groups)
