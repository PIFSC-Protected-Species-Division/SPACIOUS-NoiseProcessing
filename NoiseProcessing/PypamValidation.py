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
    soundFilePath=local_dir,
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

