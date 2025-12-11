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


def get_band_table(fft_bin_size,
                   bin1_center_frequency=0,
                   fs=64000, base=10,
                   bands_per_division=1000,
                   first_output_band_center_frequency=435,
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
    'SoundTrap_1': r'\d{9}\.\d{12}',
    'SoundTrap_2': r'\d{4}\.\d{12}',
    'yymmdd-HHMMSS.fff': r'\d{6}-\d{6}\.\d{3}',
    
}

DATE_FORMATS = {
    r'\d{8}_\d{6}_\d{3}': "%Y%m%d_%H%M%S_%f",
    r'\d{8}_\d{6}': "%Y%m%d_%H%M%S",
    r'\d{8}T\d{6}': "%Y%m%dT%H%M%S",
    r'\d{9}\.\d{12}': "%Y%m%d%H%M%S%f",
    r'\d{4}\.\d{12}': "%Y%m%d%H%M%S%f",
    r'\d{6}-\d{6}\.\d{3}': "%y%m%d-%H%M%S.%f",
    r'\d{6}_\d{6}': "%y%m%d_%H%M%S",
}


class NoiseApp:
    def __init__(self, soundFilePath, ProjName, DepName, DatabaseLoc,
                 Si=-184, clipFileSec=0, channel=0, r=0.5,
                 winname='Hann', lcut=None, hcut=None, aveSec=60,
                 pref=1, rmDC=True, Si_units='V/µPa'):
        """
        Create long-term noise metrics from audio files (local folder or GCS).
        """
        # Inputs
        self.soundFilePath = soundFilePath
        self.ProjName = ProjName
        self.DepName = DepName
        self.DatabaseLoc = DatabaseLoc

        # Calibration can be scalar dB re 1 V/µPa or a CSV path with [Hz, dB]
        self.Si = Si
        self.Si_units = Si_units
        if isinstance(self.Si, str):
            self.Si = pd.read_csv(self.Si)

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

    # ---------- GCS helpers (single, canonical set) ----------
    def _get_gcs_client(self):
        """Return a cached google.cloud.storage.Client, creating it once."""
        if not _HAS_GCS:
            raise ImportError("google-cloud-storage not installed. pip install google-cloud-storage")
        if self._gcs_storage_client is None:
            self._gcs_storage_client = storage.Client()
        return self._gcs_storage_client

    def _is_gcs_path(self, path: str) -> bool:
        return isinstance(path, str) and path.startswith("gs://")

    def _parse_gs_uri(self, uri: str):
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
            if not uris:
                raise FileNotFoundError(f"No audio files found under {self.soundFilePath}")
            return sorted(uris)
        else:
            if not os.path.isdir(self.soundFilePath):
                raise FileNotFoundError(f"Local folder not found: {self.soundFilePath}")
            entries = [f for f in os.listdir(self.soundFilePath)
                       if os.path.isfile(os.path.join(self.soundFilePath, f))]
            if not entries:
                raise FileNotFoundError(f"No files in {self.soundFilePath}")
            _, ext = os.path.splitext(entries[0])
            files = glob.glob(os.path.join(self.soundFilePath, f"*{ext}"))
            if not files:
                raise FileNotFoundError(f"No *{ext} files in {self.soundFilePath}")
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

    def _date_key_from_name(self, name: str):
        """
        Extract YYYYMMDD from the filename using the known regex/format.
        Falls back to file mtime date if no match.
        """
        base = os.path.basename(name)
        if self.DatePattern and self.DateFormat:
            m = re.search(self.DatePattern, base)
            if m:
                dt = datetime.strptime(m.group(0), self.DateFormat)
                return dt.date(), dt
        # Fallback: local file mtime (UTC date)
        try:
            ts = os.path.getmtime(name)
            dt = datetime.utcfromtimestamp(ts)
            return dt.date(), dt
        except Exception:
            return None, None

    def _start_new_hdf5_for_date(self, day: datetime.date):
        """Create a fresh HDF5 per calendar day."""
        projName = f"{self.ProjName}_{day.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}.h5"
        fullPath = os.path.join(self.DatabaseLoc, projName)
        self.fullPath = fullPath
        self.initilize_HDF5(fullPath, projName)
        return fullPath

    def get_datetime_format(self, filename):
        for date_pattern, date_format in DATE_FORMATS.items():
            match = re.search(date_pattern, filename)
            if match:
                self.DatePattern = date_pattern
                self.DateFormat = date_format
                return date_pattern, date_format
        return None, None

    def initilize_HDF5(self, fullPath, projName):
        """Create/overwrite the HDF5 file and store run/parameter metadata."""
        # Ensure output directory exists
        os.makedirs(self.DatabaseLoc, exist_ok=True)

        self.fullPath = fullPath
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
            "rmDc": self.rmDC,
            "Channel": self.channel,
            "aveSec": self.aveSec,
            "welch": self.welch,
            'rmDCoffset': self.rmDC,
            "DateRun": self.DateRun
        }
        print('Creating HDF5 File %s' % projName)
        with h5py.File(fullPath, "w") as f:
            instrument_group = f.create_group(self.DepName)
            params_group = instrument_group.create_group("Parameters")
            for k, v in metaVals.items():
                params_group.attrs[k] = v
        return None

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
        sens_db = self._interp_sensitivity_db_uPa()
        u = str(self.Si_units).lower().replace('u', 'µ')
        if u in ('v/µpa', 'v/μpa', 'v per µpa', 'v per μpa'):
            return 10 ** (sens_db / 20.0)           # already V/µPa
        elif u in ('v/pa', 'v per pa'):
            return (10 ** (sens_db / 20.0)) / 1e6   # V/Pa → V/µPa
        else:
            raise ValueError(f"Unknown Si_units='{self.Si_units}'. Use 'V/µPa' or 'V/Pa'.")

    def _read_blocks_from_file(self, path, block_sec: float = 30.0, max_block_bytes: int = 64 * 1024 ** 2):
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
        # List inputs (local or GCS)
        inputs = self._list_audio_inputs()
        self.audiofiles = inputs
    
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
    
        return None


    def run_analysis(self):
        """
        Stream inputs (local or GCS), compute PSD, average into aveSec bins,
        compute metrics, and write into an HDF5 file per *date* encountered.
        """
        _ = self.prep_audio()
        window = np.hanning(self.N).astype(np.float32)

        current_date_key = None   # YYYYMMDD
        data_start = 0            # row cursor within current day's HDF5

        with tempfile.TemporaryDirectory() as tmproot:
            for inp in self.audiofiles:
                # Download if GCS
                local_path = self._download_to_temp(inp, tmproot)

                # Figure out the date for file rotation
                file_date, file_ts_dt = self._date_key_from_name(local_path)
                date_key = file_date.strftime("%Y%m%d") if file_date else "unknown"

                # Rotate HDF5 when date changes
                if date_key != current_date_key:
                    current_date_key = date_key
                    projName = f"{self.ProjName}_{date_key}.h5"
                    self.fullPath = os.path.join(self.DatabaseLoc, projName)
                    self.initilize_HDF5(self.fullPath, projName)  # write metadata
                    data_start = 0
                    print(f"Switched to new HDF5: {projName}")

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
                        # anchor to the file’s parsed timestamp date (time-of-day preserved)
                        day_start = datetime.combine(file_ts_dt.date(), time())
                        dt_bins.append(day_start + timedelta(seconds=float(tc)))

                ttISO = np.array([dt.strftime('%Y%m%dT%H%M%S') for dt in dt_bins])

                # ---- metrics ----
                apsd_60 = 10.0 * np.log10(np.maximum(PSD_bin, 1e-30) / (self.pref ** 2))
                milidec = np.round(self.calcHybridMilidecades(apsd_60), 2)
                Broadband = np.round(self.calcBroadband(PSD_bin, delf), 2)
                TOL = np.round(self.calc13Octave(PSD_bin, B=1.0), 2)
                decadeBands = np.round(self.calcDecadeband(PSD_bin), 2)

                # ---- write ----
                self.writeDatatoHDF5(ttISO, 'DateTime', data_start=data_start, storage_mode='str')
                self.writeDatatoHDF5(milidec, 'hybridMiliDecLevels', data_start=data_start)
                self.writeDatatoHDF5(Broadband, 'broadband', data_start=data_start)
                self.writeDatatoHDF5(TOL, 'thirdoct', data_start=data_start)
                self.writeDatatoHDF5(decadeBands, 'decadeLevels', data_start=data_start)

                if data_start == 0:
                    self.writeDatatoHDF5(self.HbrdMlDec['freqLims'], 'hybridDecFreqHz',
                                         data_start=0, max_rows=len(self.HbrdMlDec['freqLims']))
                    self.writeDatatoHDF5(self.TolPrms['fc'], 'thirdOctFreqHz',
                                         data_start=0, max_rows=len(self.TolPrms['fc']))
                    self.writeDatatoHDF5(self.decPrms['decade_edges'], 'decadeFreqHz',
                                         data_start=0, max_rows=len(self.decPrms['decade_edges']))

                data_start += len(dt_bins)

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

        is_string = (storage_mode == "str")
        is_vector = (arr.ndim == 1)
        nrows = int(arr.shape[0])
        ncols = 1 if is_vector else int(arr.shape[1])

        with h5py.File(self.fullPath, "a") as hdf:
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

    def calcHybridMilidecades(self, apsd, fcross=435.0):
        """
        apsd: (T, F) in dB re 1 µPa²/Hz on the SAME freq grid as self.f.
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
    
            # keep only bands whose UPPER EDGE exceeds fcross to ensure continuity
            logbands = logbands[logbands[:, 2] > fcross]
            if logbands.size > 0:
                # avoid a gap right at the crossover
                logbands[0, 0] = max(fcross, logbands[0, 0])
    
            bands = np.vstack([low, logbands]) if logbands.size > 0 else low
            self.HbrdMlDec = {'freqLims': bands}
    
        bands = self.HbrdMlDec['freqLims']
        T = apsd.shape[0]
        out = np.full((T, bands.shape[0]), np.nan, dtype=float)
    
        for i, (flo, fcen, fhi) in enumerate(bands):
            # half-open except make the very last band inclusive on the right
            if i == bands.shape[0] - 1:
                idx = np.where((self.f >= flo) & (self.f <= fhi))[0]
            else:
                idx = np.where((self.f >= flo) & (self.f <  fhi))[0]
    
            if idx.size == 0:
                k = int(np.clip(np.searchsorted(self.f, fcen, side='left'), 0, len(self.f) - 1))
                out[:, i] = apsd[:, k]
            elif idx.size == 1:
                out[:, i] = apsd[:, idx[0]]
            else:
                p_lin = np.nansum(10.0 ** (apsd[:, idx] / 10.0), axis=1) * df  # µPa² over band
                bw = max(fhi - flo, df)
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

def plot_milidecade_statistics(instrument_group_or_list, pBands=[5, 25, 50, 75, 95]):
    """
    Plot milidecade statistics (SPD, percentiles, RMS) for one or more days.

    Parameters
    ----------
    instrument_group_or_list : h5py.Group or list of h5py.Group
        - Single HDF5 group (one file/day), OR
        - List/tuple of groups for multiple files/days.
    pBands : list of int
        Percentiles to plot.
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
    ax0.set_title('Empirical Probability Density (SPD)')

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

    ax0.set_xlim(ff.min(), ff.max())
    ax0.set_ylim(Y.min(), Y.max())
    ax0.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

    return plt

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
              titleText="",
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
    titleText : str
        Figure title (optional).
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

    if titleText:
        ax.set_title(titleText)

    ax.grid(False)
    ax.tick_params(direction='out', top=False, right=False)

    plt.tight_layout()
    return fig

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
        Si_units='V/µPa'
    )

    app.run_analysis()

    #Example for plotting (uncomment and point to an HDF5 from out_dir)
    h5_path = r"X:\\\\Kaitlin_Palmer\\\\CalCursea_680_Noise\\\\sg680_CalCurCEAS_Sep2024_20241001.h5"
    h5_path = app.fullPath
    with h5py.File(h5_path, 'r') as hdf_file:
        instrument_group = hdf_file['SG680']
        plot_milidecade_statistics(instrument_group)
        plot_ltsa(instrument_group)
