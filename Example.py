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

app.run_analysis()

#Example for plotting (uncomment and point to an HDF5 from out_dir)
h5_path = r"X:\\\\Kaitlin_Palmer\\\\CalCursea_680_Noise\\\\sg680_CalCurCEAS_Sep2024_20241001.h5"
h5_path = app.fullPath
with h5py.File(h5_path, 'r') as hdf_file:
    instrument_group = hdf_file['SG650']
    plot_milidecade_statistics(instrument_group)
    plot_ltsa(instrument_group)
