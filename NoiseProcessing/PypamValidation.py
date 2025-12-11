# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:16:27 2025

@author: pam_user
"""
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Paths and basic parameters for a SINGLE LOCAL FILE
# ------------------------------------------------------------------
# Point this to ONE audio file you want to compare on.
local_wav = r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-NoiseProcessing\ExampleAudio\glider_sg639_MHI_Apr2022_recordings_wav_sg639_MHI_Apr2022_220406-100723.803.wav"
local_wav = r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-NoiseProcessing\ExampleAudio\PAMGuide\WhiteNoise_10s_48kHz_+-0.5.wav"


# Folder containing ONLY that file (or accept that both tools
# will process any other files present here as well).
local_dir = os.path.dirname(local_wav)

# Probe sampling rate from the file (no assumptions)
info = sf.info(local_wav)
fs_local = int(info.samplerate)

# Band and nfft as per PyPAM hybrid millidecade example
band = [0, fs_local / 2.0]
nfft = int(band[1] * 2)
binsize = 60.0  # seconds; use same as aveSec below for fair comparison
binsize = 5.0  # seconds; 5 seconds for merchand 10sec noise

# ------------------------------------------------------------------
# 2) PyPAM hybrid millidecade bands
# ------------------------------------------------------------------
import pyhydrophone as pyhy
import pypam
import xarray as xr

# TODO: Replace this hydrophone definition with your actual system.
# For example, if you have a SoundTrap, use the appropriate class
# and your real sensitivity / serial number.
#
# See: https://lifewatch-pypam.readthedocs.io generated "Hybrid Millidecade bands" example. :contentReference[oaicite:0]{index=0}
#

import pyhydrophone as pyhy

hydrophone = pyhy.hydrophone.Hydrophone(
    name="MY_DEPLOYMENT",
    model="UNKNOWN",
    serial_number=0,        # or whatever you want to use
    sensitivity=0.0,        # dB re 1 V/µPa
    preamp_gain=0.0,        # dB
    Vpp=2.0,                # ADC full-scale (V peak-to-peak)
    string_format="%Y%m%d_%H%M%S",  # <<< set this to match your filenames
    calibration_file=None   # or a real CSV if you want freq-dep calibration
)

# Acoustic Survey
asa = pypam.ASA(
    hydrophone=hydrophone,
    folder_path=local_dir,
    binsize=binsize,
    nfft=nfft,
    fft_overlap=0.5,      # matches r=0.5 style overlap
    timezone="UTC",
    include_dirs=False,
    zipped=False,
    dc_subtract=True,
    channel=0,
    calibration=None,     # you can change this if you use pyhydrophone calibration
)

# Compute PyPAM hybrid millidecade bands (PSD, in dB re µPa²/Hz)
milli_ds = asa.hybrid_millidecade_bands(
    db=True,
    method="density",
    band=band,
    percentiles=None,
)

# Extract data and frequency vector
milli_da = milli_ds["millidecade_bands"]
milli_vals = milli_da.values                         # shape (T_pam, F_pam)
freq_pam = milli_da.coords["frequency_bins"].values  # Hz

# Mean spectrum across time in linear domain, back to dB
spec_pam = 10.0 * np.log10(
    np.nanmean(10.0 ** (milli_vals / 10.0), axis=0)
)

# ------------------------------------------------------------------
# 3) Your NoiseApp hybrid milidecade bands on the SAME folder
# ------------------------------------------------------------------
from noiseProcessGoogleCloud import NoiseApp
import h5py

out_dir = r"C:\Temp\HybridMilliDailyCompare"   # <<< EDIT as desired
os.makedirs(out_dir, exist_ok=True)

app = NoiseApp(
    Si=0, # This isn an offset (linear)
    soundFilePath = r"C:\Users\pam_user\Documents\GitHub\SPACIOUS-NoiseProcessing\ExampleAudio\PAMGuide",
    ProjName='PyPAM_Compare',
    DepName='DEP1',
    DatabaseLoc=out_dir,
    rmDC=True,
    Si_units='V/µPa',
    aveSec=binsize,   # match PyPAM binsize
)

# This will process all files with the same extension in local_dir.
# If you want a strict one-to-one comparison, keep only local_wav there.
app.run_analysis()

# Open the resulting HDF5 and extract your hybrid milidecade data
h5_path = app.fullPath
with h5py.File(h5_path, 'r') as hdf_file:
    instrument_group = hdf_file[app.DepName]

    # Hybrid milidecade levels (time x band) and centre frequencies
    milidec_app = instrument_group['hybridMiliDecLevels'][:]   # (T_app, F_app)
    freq_app = instrument_group['hybridDecFreqHz'][:, 1]       # Hz (centre freq)

# Mean spectrum across time in linear domain, back to dB
spec_app = 10.0 * np.log10(
    np.nanmean(10.0 ** (milidec_app / 10.0), axis=0)
)

# ------------------------------------------------------------------
# 4) Plot PyPAM vs NoiseApp hybrid milidecade mean spectra
# ------------------------------------------------------------------
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False

plt.figure(figsize=(8, 5))
plt.semilogx(freq_pam, spec_pam, label="PyPAM hybrid millidecade", linewidth=2)
plt.semilogx(freq_app, spec_app, label="NoiseApp hybrid millidecade", linewidth=2, linestyle="--")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean PSD (dB re 1 µPa²/Hz)")
plt.title("Hybrid millidecade bands: PyPAM vs NoiseApp")
plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# 5) Check for disparities
# -----------------------------------------------------------------

# Interpolate your NoiseApp curve onto the PyPAM frequency grid
spec_app_interp = np.interp(freq_pam, freq_app, spec_app)

# Difference: NoiseApp - PyPAM (dB) at each frequency
delta = spec_app_interp - spec_pam

print("Median offset (NoiseApp - PyPAM):", np.nanmedian(delta), "dB")
print("5th–95th percentile of offset:", np.nanpercentile(delta, [5, 95]), "dB")


# # ------------------------------------------------------------------
# # 6) Third-octave band comparison + LaTeX-ready table
# # ------------------------------------------------------------------
# import pandas as pd
# from scipy.signal import welch
# from pypam import utils as pam_utils

# # ---- 6.1 Read audio and compute Welch PSD (density, uPa^2/Hz) ----
# data, fs_check = sf.read(local_wav)
# assert fs_check == fs_local, "Sampling rate mismatch between sf.info and sf.read"

# # Use first channel if stereo
# if data.ndim > 1:
#     data = data[:, 0]

# # Welch PSD with same nfft / 50% overlap style as before
# f_psd, Pxx = welch(
#     data,
#     fs=fs_local,
#     nperseg=nfft,
#     noverlap=nfft // 2,
#     scaling="density",
# )

# # ---- 6.2 Define 1/3-octave bands safely below Nyquist ----
# # Use pypam.utils.oct_fbands to be consistent with PyPAM
# min_freq = 20.0  # Hz (adjust if you want to start higher)
# max_freq = 0.8 * (fs_local / 2.0)  # safety margin below Nyquist

# bands_idx, fc_all = pam_utils.oct_fbands(int(min_freq), int(max_freq), fraction=3)
# fc = fc_all  # centre frequencies of 1/3-octave bands

# # Standard 1/3-oct band edges:
# # bandwidth = 1/3 octave => factor 2^(1/3) between centres
# # edges are ±1/6 octave from centre: flo = fc / 2^(1/6), fhi = fc * 2^(1/6)
# edge_factor = 2.0 ** (1.0 / 6.0)
# flo = fc / edge_factor
# fhi = fc * edge_factor

# # ---- 6.3 Re-bin Welch PSD into 1/3-octave bands (PyPAM-style) ----
# third_pam_lin = np.full_like(fc, np.nan, dtype=float)

# for i, (fl, fh) in enumerate(zip(flo, fhi)):
#     # select Welch bins inside this band
#     band_mask = (f_psd >= fl) & (f_psd < fh)
#     if np.any(band_mask):
#         third_pam_lin[i] = np.nanmean(Pxx[band_mask])

# # Convert to dB re 1 µPa²/Hz
# third_pam_db = 10.0 * np.log10(third_pam_lin)

# # ---- 6.4 Extract NoiseApp 1/3-octave data from HDF5 ----
# # Try to auto-detect the dataset names; adjust if needed.


# # --- 6.1 Load NoiseApp 1/3-octave data from the same HDF5 file ---
# with h5py.File(app.fullPath, "r") as hf:
#     instrument_group = hf[app.DepName]

#     print("Available datasets in NoiseApp instrument group:",
#           list(instrument_group.keys()))

#     # Your implementation: time x band, in dB re 1 µPa²/Hz
#     third_app = instrument_group["thirdoct"][:]          # (T_app, F_app)
#     freq_app_3 = instrument_group["thirdOctFreqHz"][:]   # (F_app,) centre freqs in Hz

# # Average across time in linear domain, then back to dB
# spec_app_3 = 10.0 * np.log10(
#     np.nanmean(10.0 ** (third_app / 10.0), axis=0)
# )



# hf =  h5py.File(app.fullPath, 'r') 
# instrument_group =  hf[app.DepName] 


# print("Available datasets in NoiseApp instrument group:", list(instrument_group.keys()))


# third_app = instrument_group['thirdoct'][:]  # expect shape (T_app, F_app)
# instrument_group['thirdOctFreqHz'][:]

# # Guess frequency dataset name for third-octave bands
# freq_keys = [k for k in instrument_group['thirdOctFreqHz']
#              if "freq" in k.lower() and "third" in k.lower()]

# if not freq_keys:
#     # Fall back to any freq-like key; you may need to tighten this if it misfires
#     freq_keys = [k for k in instrument_group.keys() if "freq" in k.lower()]

# if not freq_keys:
#     raise RuntimeError(
#         "Could not find a frequency vector for third-octave bands in NoiseApp HDF5."
#     )

# freq_name = freq_keys[0]
# freq_app_third = instrument_group[freq_name][:]

# # If freq dataset has extra columns (e.g. [low, centre, high]),
# # take the centre frequency column (index 1) like hybridDecFreqHz.
# if freq_app_third.ndim == 2 and freq_app_third.shape[1] >= 2:
#     freq_app_third = freq_app_third[:, 1]

# # NoiseApp levels: time x band, in dB
# if third_app.ndim == 2:
#     third_app_db = 10.0 * np.log10(
#         np.nanmean(10.0 ** (third_app / 10.0), axis=0)
#     )
# elif third_app.ndim == 1:
#     third_app_db = third_app.astype(float)
# else:
#     raise RuntimeError(
#         f"Unexpected shape for NoiseApp third-octave array: {third_app.shape}"
#     )

# # ---- 6.5 Interpolate NoiseApp third-octave onto the PyPAM band centres ----
# # Restrict to overlapping frequency range to avoid extrapolation nonsense
# common_mask = (fc >= freq_app_third.min()) & (fc <= freq_app_third.max())

# fc_common = fc[common_mask]
# pam_common = third_pam_db[common_mask]

# third_app_interp = np.interp(fc_common, freq_app_third, third_app_db)

# delta_third = third_app_interp - pam_common  # NoiseApp - PyPAM (dB)

# print("Third-octave comparison (NoiseApp - PyPAM):")
# print("  Median offset:", np.nanmedian(delta_third), "dB")
# print("  5th–95th percentile:", np.nanpercentile(delta_third, [5, 95]), "dB")

# # ---- 6.6 Build a LaTeX-ready summary table ----
# df_third = pd.DataFrame({
#     "fc_Hz": fc_common,
#     "PyPAM_dB": pam_common,
#     "NoiseApp_dB": third_app_interp,
#     "Delta_dB": delta_third,
# })

# # Optional: round for prettier output
# df_third_rounded = df_third.copy()
# df_third_rounded["fc_Hz"] = df_third_rounded["fc_Hz"].round(1)
# df_third_rounded["PyPAM_dB"] = df_third_rounded["PyPAM_dB"].round(2)
# df_third_rounded["NoiseApp_dB"] = df_third_rounded["NoiseApp_dB"].round(2)
# df_third_rounded["Delta_dB"] = df_third_rounded["Delta_dB"].round(2)

# print("\nLaTeX table for 1/3-octave comparison:\n")
# print(df_third_rounded.to_latex(index=False, float_format="%.2f"))






