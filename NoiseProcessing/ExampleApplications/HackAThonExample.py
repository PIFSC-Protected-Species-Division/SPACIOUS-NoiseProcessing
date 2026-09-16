
import h5py
import glob
from pathlib import Path
from noiseProcessGoogleCloud import (print_h5_tree, plot_milidecade_statistics, 
                                     plot_ltsa, plot_third_octave_bands,
                                     list_hdf5_deployments, summarize_hdf5_file,
                                     plot_third_octave_bands,
                                     export_metric_csv)
import os
import matplotlib.pyplot as plt



# Imagine you've been running the noise analysis for a few days and thought
# quite carefully about how you set it up but now you can't recall. The following
# functions are designed to display some of the basics of the noise file. 

# Where you stored your favoirite glider file
h5_path = r"X:\Kaitlin_Palmer\GliderRodeo\sg607_20260128\\sg607_20260128.h5"
figDir = r"X:\\Kaitlin_Palmer\\GliderRodeo\\TestFigures\\"


summarize_hdf5_file(h5_path) # The basics of the data and dataset names (e.g. OH CRAP I FORGET)


# I only want info on one dataset in the HDF5
summarize_hdf5_file(h5_path, group_name='GliderRodeo')

# I only need the deployment names (because I forget just those)
list_hdf5_deployments(h5_path)

# Or if you just want to know what the run parameters were set at
# hdf_file['GliderRodeo']['Parameters'].attrs.keys()


# Plots are available for one dataset (i.e. deployment at a time) so you
# should know the deployment id or use list_hdf5_deployments to recover them
# if you want to string multiple together


hdf_file = h5py.File(h5_path, 'r')
Glider_id = "sg607_20260128"

# Try loading the data (file['deployment']['metric'][0:10])
hdf_file['GliderRodeo']['broadband']



# Output path and name comprised of several parts, for simplicity you could
# use just one string
save_file = os.path.join(figDir, f"{Glider_id}_milidecade_SPD.pfd")

fig = plot_milidecade_statistics(hdf_file['GliderRodeo'], 
                                 title=Glider_id, # If you want a custom title
                                 save_path=save_file,   
                                 pBands=[5, 25, 50, 75, 95], # Probability Bands to show
                                 dpi= 150)  # Resolution (for publicaiton figures)


plt.close(fig)

# Create an LTSA
save_LTSA = os.path.join(figDir, f"{Glider_id}_5min_ltsa.png")
fig = plot_ltsa(hdf_file['GliderRodeo'], 
                title=Glider_id, 
                save_path=save_LTSA,
                averaging_period='8min',  #Pandas offset alias for time-averaging (e.g., '5min', '1min', '1H',12d).
                freq_scaled=True,   # real frequency on y
                log_freq=False,
                dpi=150)
plt.close(fig)


# And finally third ocatve, as above you really only need the loaded file and deployment name
plot_third_octave_bands(hdf_file['GliderRodeo'])




#%% Heck with python, I want these data in a CSV

projectNames = list_hdf5_deployments(h5_path)

# With the included plotting function, make a plot
for proj in projectNames:
    with h5py.File(h5_path, 'r') as hdf_file:
        Project = hdf_file[proj]
        #plot_milidecade_statistics(Project, title=proj) # This takes a while
        #plot_third_octave_bands(hdf_file[proj])
        export_metric_csv(h5_path, 
                          metric = 'broadband', 
                          group_name = proj, 
                          utput_csv= proj+'broadband.csv' )