# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:26:22 2025

@author: pam_user
"""
from noiseProcessGoogleCloud import NoiseApp


# Define the location of the data on google cloud
gsCloudLoc = "gs://swfsc-1/2024_CalCurCEAS/glider/audio_flac/sg680_CalCurCEAS_Sep2024"


# Define where you want to store the data
out_dir = r"C:\Users\pam_user\Documents\HybridMilliDaily"

# Loccation of the calibration csv, first column should be frequency in Hz 
# Second column should be end-to-end calibration in dB
calib_csv = 'C:\\Users\\pam_user\\Downloads\\sg680_CalCurCEAS_Sep2024_sensitivity_2025-07-29.csv'


# Declare the noise app object and give it a project name and 
# a deployment name. Note that you can store multiple 
# deployments within a project
app = NoiseApp(
    Si=calib_csv,
    soundFilePath=gsCloudLoc,
    ProjName='sg680_CalCurCEAS_Apr2022',
    DepName='SG680',
    DatabaseLoc=out_dir,
    rmDC=True, # Remove the DC offset from each audio file
    Si_units='V/µPa'
)

# Go do the thing!
app.run_analysis()


#%% Plot Some of the days

import h5py
from noiseProcessGoogleCloud import plot_milidecade_statistics, plot_ltsa

#Example for plotting (uncomment and point to an HDF5 from out_dir)
h5_path = r"X:\\\\Kaitlin_Palmer\\\\CalCursea_680_Noise\\\\sg680_CalCurCEAS_Sep2024_20241001.h5"

# Explore the hdf5 file a bit
hdf_file = h5py.File(h5_path, 'r')

# This should be the project name
projectName = list(hdf_file.keys())

# Use this to see the deployments within the project
hdf_file[projectName[0]].keys()

# This shows you the various metrics  including datetime stamp, broadband
# decadd, third octave, and hybridmilidecade band levels 


# Lets look at the first  ten decade levels and their frequencies
hdf_file[projectName[0]]['decadeLevels'][0:9] # Values
hdf_file[projectName[0]]['decadeFreqHz'][0:9] # Lower frequency range

# Lets look at the first  ten decade levels and their frequencies
hdf_file[projectName[0]]['hybridMiliDecLevels'][0:10] # Values
hdf_file[projectName[0]]['hybridDecFreqHz'][0:10] # Lower frequency range


# With the included plotting function, make a plot
with h5py.File(h5_path, 'r') as hdf_file:
    Project = hdf_file['CalCurCEAS_2024']
    plot_milidecade_statistics(Project) # This takes a while
    plot_ltsa(Project) # This is still in development
