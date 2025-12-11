# SPACIOUS-NoiseProcessing

**SPACIOUS-NoiseProcessing** is a Python-based toolkit developed by the PIFSC Protected Species Division to support the processing and analysis of underwater noise data collected under the SPACIOUS program. These tools contribute to evaluating noise conditions in marine environments and assessing potential impacts on protected species, including cetaceans, turtles, and seals.

The repository includes core modules for reading and processing acoustic recordings, as well as example scripts demonstrating practical workflows.

<img width="999" height="787" alt="image" src="https://github.com/user-attachments/assets/3af06d4d-5992-467a-998f-aedf73fef275" />

____________
## TL;DR
An R-based methodology for storing noise metrics in [HDF5 (Hierarchical Data Format version 5)](https://www.hdfgroup.org/solutions/hdf5/)  files. The code is modularized so a user can pick and chose which metrics to store. This repository is licensed as open source software to help the community deal with data irregularities. 

Several pre-defined soundscape metrics that save to daily HDF5 files
1) Hybrid-millidecade band levels
2) Third-octave band levels
3) Decade band levels
4) Broadband level

Pre-written plotting functions. These returns standard plots in ggplot format.
1) LTSA (Long-term Spectral Average)
2) Probability distribution
_______________________
## Introduction

Measuring ambient noise levels is an important part of many ecological studies, particularly in the marine environment where noise levels are both an emerging conservation concern for many signaling or listening species and a hindrance to passive acoustic monitoring for signals from soniferous species. Anthropogenic noise has been implicated in the stranding of deep diving marine species such as beaked whales and as a stressor in other marine mammals including endangered killer whales and right whales. With ongoing acoustic monitoring efforts for many vocally active species, quantitative assessment of background noise levels is also key in understanding changes in detection range which could bias monitoring or real-time conservation efforts.

There are plethora of free and paid services used to calculate noise metrics including PAMGuard (www.Pamguard.org), [Triton](https://www.cetus.ucsd.edu/technologies_triton.html), and [PAMGuide](https://sourceforge.net/projects/pamguide/). Each of these systems have their benefits and limitations and bioacousicians often find themselves in need of modification for their own specific data needs. For instance, Triton and its associate Remoras are also free and there is a compiled version that does not require MATLAB. However, any customization does require a MATLAB license which is frequently cost prohibitive, especially for researchers in developing countries. PAMGuard is both free and an industry standard but can be difficult to work with and JAVA is not commonly known among biologists. This makes it challenging for researchers to troubleshoot without reaching out to a small but dedicated team of maintainers. Finally, PAMGuide was written in both R and MATLAB and includes a Matlab-GUI, allowing for user-friendly interface. 

The [PAMGuide paper (Merchant et. al 2014)](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.12330) was well received, with over 200 citations, in no small part because of the well-documented and published code provided by the author. However, it too has limitations have required extensive modifications for many long-term noise projects and, after 9 years, the code was due for some updates. Principle among the limitations the speed of analysis and the storage options initially provided (either mat files or .csv files). These become ungainly at best and untenable at worst when working with large, multi-instrument, or multi-year arrays. 

An ideal storage solution would allow storage for large, multi-level datasets with descriptors, metadata, efficient access speed and accessibility across multiple platforms. Presently HDF5 databases meet these criteria. Such formats are platform independent, self-described, and open source (sharing is caring).

Additionally [Martin et al., 2021](https://static1.squarespace.com/static/52aa2773e4b0f29916f46675/t/6033d0181ce4934ad7c3d913/1614008346204/Martin_et_al_2021_Hybrid+millidecade+spectra_practical+format+for+ambient+data+exchange.pdf) provided a efficient methodology for storing and sharing large database of sound metrics. These metrics provide an efficient methodology for sharing large-scale and long-term datasets. 


The principal goal of of this project is to produce a reliable and flexible system for recording noise metrics. In achieving this goal I required that the system be built in a well-established language within the biological field ([Python](https://www.python.org/)) which is free and open source software under the GPL license. These characteristics make the language as accessable to as many researchers as possible. I also required that the data storage should also allow for multi-level organization, storage of large files, be accessable to multiple software packages/languages, and ultimately itegrate with the sturdy metadata managment system, [Tethys](https://tethys.sdsu.edu/). 

In this repository I've used the Python-based FFT and level metric calculations by Merchant et. al (2014), but updated the functions and wrappers to allow for procesing of large numbers of sound files and storing in a HDF5 database. 

This repository contains a modidified version of [Merchant et. al's 2014](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12330) noise analysis tools. The major modifications between this version and the origional incude:

1) Saving data to HDF5 files rather than CSV (comma-separated variable) files. This allows for much larger data sets to be stored efficiently, including multiple deployments within a single study;
2) Saving large PSD (power spectral density) results as hybrid milidecades (e.g. [Martin et al. 2021](https://static1.squarespace.com/static/52aa2773e4b0f29916f46675/t/6033d0181ce4934ad7c3d913/1614008346204/Martin_et_al_2021_Hybrid+millidecade+spectra_practical+format+for+ambient+data+exchange.pdf));
3) Allowing user-specified anlysis parameters to be saved as part of database;
4) Ability to exclude the first N seconds of the file (this is useful for Soundtraps);
5) Example code for constructing popular noise level plots from the HDF5 files.
6) Allows for calibration curves (end-to-end only)

Computational results have been validated with test data against the most recent version of PAMGuide and [Pypam](https://github.com/lifewatch/pypam). Results were identical to PAMGuide and within 1 dB of Pypam with variations attributable to minute/second breaks for averaging periods.

## Features

- Modular functions for filtering and processing underwater acoustic data.  
- Example workflow (`Example.py`) illustrating how to use the processing modules.  
- Straightforward structure for users who wish to extend methods or integrate them into larger pipelines.

---

## Getting Started

### Prerequisites

- Python 3.x  
- Common scientific Python packages (e.g., `numpy`, `scipy`).  

A `requirements.txt` file should be added once dependencies are finalized.

### Installation

```bash
git clone https://github.com/PIFSC-Protected-Species-Division/SPACIOUS-NoiseProcessing.git
cd SPACIOUS-NoiseProcessing

# Optional: set up a virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies once requirements.txt is added
pip install -r requirements.txt
