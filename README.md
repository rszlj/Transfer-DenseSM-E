# Transfer-DenseSM-E
This repository is a simple implementation of the Transfer-DenseSM-E


## Part I: Data preparation

The preprocessing steps used in the paper is for 9_km grids, while these posted here are designed for points. The main reason of a simplified version includes: 
- EASE 2.0 is not supported by GEE and thus images were first downloaded with a size of 13x13 km, allowing a reprojection to cover the corresponding 9km grid cell 
- We export images to google drive first and download it from local in an app way, being too complex to be posted here.
- The prepared samples_x can be direclty used in models, being more straightfoward.
However, we are happy to share the codes for images

### In-situ soil moisture
Ground soil moisture measurements are available at https://ismn.geo.tuwien.ac.at/en/.

A jupyter notebook (Preprocessing_ISMN_Raw_Data.ipynb) was built for the preprocessing of the raw data

The outputs include: 
- Site-specific files named as network-station including the available daily averaged soil moisutre at 0 - 5 cm of each station;
- A csv file containing the details of each station.

### SMAP data
The SMAP data is available at https://nsidc.org/data/SPL3SMP. A pyhton script can be generated automatically for batch download

Use Extract the Extract_SMAP_9km.ipynb to extract the soil moisutre over each station.

### Remote sensing data and Reanalysis weather data from Google Earth Engine (GEE)
#### Setup
An google developer account is required to access the GEE

The https://github.com/giswqs/geemap is suggested for the setup of GEE

#### prepare the input variables using GEE
Use Prepare_samples.ipynb to extract all the input variables listed in Table 2. 

## Part II: Train/Val 9 km models
comming soon



## Reference
Liujun Zhu, Junjie Dai, Yi Liu, Shanshui Yuan & Jeffrey P. Walker (under review) A cross-resolution transfer learning approach for soil moisture retrieval with limited training samples, Remote Sensing of Environment
