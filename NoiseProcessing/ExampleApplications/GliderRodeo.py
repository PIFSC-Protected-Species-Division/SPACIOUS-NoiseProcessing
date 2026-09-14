# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:26:22 2025

@author: pam_user
"""
from noiseProcessGoogleCloud import NoiseApp, print_h5_tree
import os
import numpy as np



# One entry per glider deployment, keeping all related settings together
# so they can't accidentally get out of sync with each other.

calib_csv_Whispr = 'C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\sg680_CalCurCEAS_Sep2024_sensitivity_2025-07-29.csv'
SeaExplorer_calib = "C:\\Users\\pam_user\\Downloads\SEA117-M026_20260128_hpSensitivity_ch1_ESTIMATE.csv"


deployments = [
    {
        "mission_id": "sg607_20260128",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/sg607_20260128_WHICEAS/Recordings_CENSOR/flac",
        "channel": 1, # run channel 1
        "hyd_sensitivity": calib_csv_Whispr,  # HTI calibration
    },
    {
        "mission_id": "sg274_20260128",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/sg274_20260128_WHICEAS/Recordings_CENSOR/flac",
        "channel": 1,
        "hyd_sensitivity": calib_csv_Whispr,  # HTI calibration
    },
    
    {
        "mission_id": "risso-20260128",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/risso-20260128/Recordings_CENSOR/wav_2kHz",
        "channel": 1,
        "hyd_sensitivity": -203,
     },
    
    {
        "mission_id": "stenella-20260128",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/stenella-20260128/Recordings_CENSOR/flac",
        "channel": 1,
        "hyd_sensitivity": -165,  # HTI calibration
    },    
    
    {
        "mission_id": "capex987_20260128",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/capex987_20260128/Recordings_CENSOR/wav_512kHz/OBS-1195.17.512000.M36-V35-100",
        "channel": 1,
        "hyd_sensitivity": -165.11,  # HTI calibration
    }, 
    
    {
        "mission_id": "belladonna_20260128",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/belladonna_20260128/Recordings_CENSOR/flac/200kHz",
        "channel": 1,
        "hyd_sensitivity":  -176,
        },
    
    {
        "mission_id": "SEA117-M026_20260128_30sec",
        "gs_path": "gs://nmfs-collaborative/2026_GliderRodeo/SEA117-M026_20260128/Recordings_CENSOR/wav_30s",
        "channel": 1,
        "hyd_sensitivity": SeaExplorer_calib  
        },
    
    
]

# Define where you want to store the data
out_dir_base = r"X:\Kaitlin_Palmer\GliderRodeo"

# Local scratch drive for staging HDF5 writes before syncing to the shared X: drive
local_staging_base = r"C:\GliderRodeoScratch"

for deployment in deployments:

    print(deployment["gs_path"])

    out_dir = os.path.join(out_dir_base, deployment["mission_id"])
    staging_dir = os.path.join(local_staging_base, deployment["mission_id"])

    

    # Declare the noise app object and give it a project name and
    # a deployment name. Note that you can store multiple
    # deployments within a project
    app = NoiseApp(
        soundFilePath=deployment["gs_path"],
        ProjName=deployment["mission_id"],
        DepName='GliderRodeo',
        channel=deployment["channel"],
        Si=deployment["hyd_sensitivity"],
        DatabaseLoc=out_dir,
        split_hdf5_by_day=False,
        rmDC=True, # Remove the DC offset from each audio file
        Si_units='V/µPa',
        existing_deployment_mode='skip',
        local_staging_dir=staging_dir,
        sync_every_n_files=25)
    
    # Confirm there's audio to process before kicking off the analysis
    audio_files = app._list_audio_inputs()
    print(f"Found {len(audio_files)} audio file(s) in {deployment['gs_path']}")
    if not audio_files:
        print(f"Skipping {deployment['mission_id']}: no audio files found.")
        continue
    
    

    # Go do the thing!
    app.run_analysis()
    
    
#%% Create PSD plots

import h5py
import glob
from pathlib import Path
from noiseProcessGoogleCloud import plot_milidecade_statistics, plot_ltsa, plot_third_octave_bands
import os
import matplotlib.pyplot as plt

#Example for plotting (uncomment and point to an HDF5 from out_dir)





# Where to store the figures
figDir = r"X:\\Kaitlin_Palmer\\GliderRodeo\\TestFigures\\"


# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\SEA117-M026_20260128_30sec\\SEA117-M026_20260128_30sec.h5"
# Oh no! I forget the structure of the HDF5!
print_h5_tree(h5_path)




hdf_file = h5py.File(h5_path, 'r')
Glider_id = "M026_20260128_30sec"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")

print_h5_tree(h5_path)

fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)



save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)


plt.close(fig)



# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\belladonna_20260128\\belladonna_20260128.h5"
hdf_file = h5py.File(h5_path, 'r')
Glider_id = "belladonna_20260128"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")
fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)
plt.close(fig)




# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\capex987_20260128\\capex987_20260128.h5"
hdf_file = h5py.File(h5_path, 'r')
Glider_id = "capex987_20260128"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")
fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)
plt.close(fig)




# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\risso-20260128\\risso-20260128.h5"
hdf_file = h5py.File(h5_path, 'r')
Glider_id = "risso-20260128"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")
fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)
plt.close(fig)



# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\sg274_20260128\\sg274_20260128.h5"
hdf_file = h5py.File(h5_path, 'r')
Glider_id = "sg274_20260128"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")
fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)
plt.close(fig)




# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\sg607_20260128\\sg607_20260128.h5"
hdf_file = h5py.File(h5_path, 'r')
Glider_id = "sg607_20260128"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")
fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)
plt.close(fig)




# Explore the hdf5 file a bit
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\stenella-20260128\\stenella-20260128.h5"
hdf_file = h5py.File(h5_path, 'r')
Glider_id = "stenella-20260128"
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.png")
fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, 
                                 save_path=save_file)  # This takes a while
plt.close(fig)
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], title=Glider_id, save_path=save_LTSA,
                averaging_period='5min',
                freq_scaled=True,   # real frequency on y
                log_freq=True)
plt.close(fig)





#%% Create PSD plots



