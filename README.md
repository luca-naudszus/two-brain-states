This repository contains the data and analysis code for [redacted for peer review]. 
The analyses reproduce all results reported in the paper. 

### Overview
Data is provided for main and supplementary analyses. 

The folder `0_fNIRS_preprocessed` has been added for clarity. 
Preprocessed files are available at the original OSF repository (https://osf.io/xcgp6/overview?view_only=d655e36f24ba433c9f20bf0cc67b3062) and must be downloaded there to use the code provided here. 

### Main data
All main data was generated in the "four blocks" analysis (see below). 
File names therefore include "four-blocks" as well as other characteristics, 
in the following abbreviated by the asterisk \*. 

- `1_fNIRS_prepared contains` prepared fNIRS time series (`timeseries\*.csv`) alongside dyad-level information (`documentation\*.csv`) and information on the channels included (`channels\*.csv`). Further content is a folder with additional information necessary for preparation: 
    - `cutpoints_videos.xlsx`: visually retrieved timestamps for beginning and end of tasks
    - `dyadList.csv`: list of all dyad ids and associated participant ids
    - `fNIRS_chs_ROIproximal.csv`: information on which channels are optimal for their respective ROI per dyad and session
      
- `2_clustering contains` the clustering output: pipeline description, cluster means, backfit timeseries (`results-table\*.csv`), as well as a number of metrics evaluating the clustering (parameter-space\*.csv). 
  
- `3_time-series-metrics` contains a table with one task, session, and dyad per row detailing the values of the three time series 
  metrics used here (occurrence, coverage, duration). 

### Code
`riemannianKMeans.py` contains a number of helper functions particularly needed 
  for the Riemannian-geometry-based k-Means applied in the core section of this 
  analysis. 

`preparations.py` can be used to generate fNIRS time series ready for clustering from preprocessed fNIRS data. 
  To use `preparations.py`, first move preprocessed fNIRS data into the folder 0_fNIRS_preprocessed (see above). 
  `preparations.py` can generate three different preparations, only two of which are used in this paper: 
- `four-blocks`: Two-brain time series as detailed in the paper. 
- `one-brain`: Single-brain time series for shared brain-state analyses, described in the supplement. 
- `two-blocks`: Two-brain time series without between-brain connectivity. Not reported in the paper. 

`riemannian_clustering.py` can be used to cluster prepared fNIRS time series according to pre-specified hyperparameters. 
  Alternatively, it can be used for a brute-force grid search across a pre-specified hyperparameter space. 
  Further instructions can be found in its comments. 
  It generates the backfit time series, the cluster centroids, as well as an additional file (too large for upload into this repository) 
  detailing the kernel matrices. 
  
`metrics.py` calculates brain state metrics on the backfit time series. It outputs into one metrics table. 
  
# Supplementary data
We provide data clustered into three brain states, as detailed in the supplement, in the folder `supplementary/two-brain_clusters-3`. 
  This folder contains only clustering and metrics outputs as it is based on the same prepared time series as the seven-cluster analysis. 
  
We also provide data prepared for the shared brain-state analysis, as detailed in the supplement, in the folder `supplementary/shared-brain`. 
  The clustering and metrics subfolders contain two subfolders, each for one number of clusters used in the clustering algorithm. 
  
# Citation
When using this work, please cite: [redacted for peer review]
