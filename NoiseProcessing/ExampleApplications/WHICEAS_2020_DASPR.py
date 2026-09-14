# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:26:22 2025

@author: pam_user
"""
from noiseProcessGoogleCloud import NoiseApp, print_h5_tree
import os
import numpy as np

# Define the location of the data on google cloud
gsCloudLoc = [
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS1/ST-1/1208795167",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS2/ST-2/1208766495",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS3/ST-15/470290496",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS4/ST-3/1208520735",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS5/ST-4/1208504351",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS6/ST-5/1208754207",# bad data
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS7/ST-7/1208496160",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS8/ST-8/1208774686",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS9/ST-9/1208487968",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS11/ST-11/470081600",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS12/ST-12/671125543",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS13/ST-13/470077504",
    "gs://pifsc-1/drifting_recorder/2020_WHICEAS_2001/DS14/ST-2/1208766495"]




hydSen__Ch1 =[-164.9,
-155.0,
-155.2,
-155.0,
-155.0,
-155.0,
-155.4,
-155.0,
-164.9,
-164.9,
-155.0,
-155.4,
-155.0,
-155.4]

ht1SericalNumber_Ch1 = [856081,
856091,
856116,
856091,
856087,
856091,
856086,
856087,
856081,
856081,
856087,
856086,
856087,
856086]



# Define where you want to store the data
out_dir = r"X:\Kaitlin_Palmer\WHICEASE2020_soundscape"

# 9,10,13,
for ii in range(0,len(gsCloudLoc)):
    print(gsCloudLoc[ii]) 
    path = gsCloudLoc[ii]
    ds_part = [p for p in path.split('/') if p.startswith('DS')][0]
    ds_num = int(ds_part[2:])
    ds_str = f"DS{ds_num:02d}"
    
    depId = ds_str+"_ch01_hti_"+str(ht1SericalNumber_Ch1[ii])
    print(depId) 
    
    # Declare the noise app object and give it a project name and 
    # a deployment name. Note that you can store multiple 
    # deployments within a project
    app = NoiseApp(
        soundFilePath=gsCloudLoc[ii],
        ProjName='WHICEAS2020_ch01',
        DepName=depId,
        channel = 0,
        Si = hydSen__Ch1[ii], # HTI ccalibration
        DatabaseLoc=out_dir,
        split_hdf5_by_day = False,
        rmDC=True, # Remove the DC offset from each audio file
        Si_units='V/µPa',
        existing_deployment_mode='overwrite')

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
h5_path = r"X:\Kaitlin_Palmer\WHICEASE2020_soundscape\WHICEAS2020_ch01.h5"

# Where to store the figures
figDir = r"X:\Kaitlin_Palmer\WHICEASE2020_soundscape\figures\\"

# Explore the hdf5 file a bit
hdf_file = h5py.File(h5_path, 'r')

# This should be the project name
deploymentNames = list(hdf_file.keys())


# With the included plotting function, make a plot

# With the included plotting function, make a plot

with h5py.File(h5_path, 'r') as hdf_file:
    for ii in range(len(deploymentNames)):
        dep_name = deploymentNames[ii]
        Project = hdf_file[dep_name]
        
        DASBR_id = dep_name.split('_',1)[0]
        
        save_file = os.path.join(figDir, f"{DASBR_id}_milidecade_SPD.png")
        fig = plot_milidecade_statistics(Project, 
                                         title=DASBR_id, 
                                         save_path=save_file)  # This takes a while
        plt.close(fig)
        save_tob = os.path.join(figDir, f"{DASBR_id}_third_octave.png")
        fig = plot_third_octave_bands(Project, 
                                      title=DASBR_id, 
                                      save_path=save_tob)
        plt.close(fig)
        save_LTSA = os.path.join(figDir, f"{DASBR_id}_5min_ltsa.png")
        fig = plot_ltsa(Project, title=DASBR_id, save_path=save_LTSA,
                        averaging_period='5min',
                        freq_scaled=True,   # real frequency on y
                        log_freq=False)
        plt.close(fig)
