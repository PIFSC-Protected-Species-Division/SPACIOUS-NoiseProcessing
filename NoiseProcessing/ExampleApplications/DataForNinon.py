# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:26:22 2025

@author: pam_user
"""


#%% Plot example day

import h5py
import glob
from pathlib import Path
from noiseProcessGoogleCloud import plot_milidecade_statistics, plot_ltsa, plot_third_octave_bands, export_metric_csv

#Example for plotting (uncomment and point to an HDF5 from out_dir)
h5_path = r"C:/Users\\pam_user\\Downloads\\JavaInlet.h5"


# Explore the hdf5 file a bit
hdf_file = h5py.File(h5_path, 'r')

# This should be the project name
projectNames = list(hdf_file.keys())

# Use this to see the deployments within the project
hdf_file[projectNames[0]].keys()



# With the included plotting function, make a plot
for proj in projectNames:
    with h5py.File(h5_path, 'r') as hdf_file:
        Project = hdf_file[proj]
        #plot_milidecade_statistics(Project, title=proj) # This takes a while
        #plot_third_octave_bands(hdf_file[proj])
        export_metric_csv(h5_path, metric = 'broadband', group_name = proj, output_csv= proj+'broadband.csv' )
    
    
    


#%% Data exploration


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

