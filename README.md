# Feature-based Magnitude Estimates for Small Earthquakes in the Yellowstone Region 

In this study we use features derived from short-duration waveforms segments and event location information in support vector machines to produce local magnitude estimates that are consistent with the existing University of Utah Seismograph Stations (UUSS) earthquake catalog for the Yellowstone Volcanic Region. 

This code implements the methods and produces the figures described in Armstrong *et al.*, which has been accepted at BSSA in May 2025.

Magnitudes are produced using station and distance corrections and [code](https://github.com/sgjholt/ynp_local_magnitude_recalibration) from [Holt et al., (2022)](https://doi.org/10.1785/0120210240)

## Armstrong *et al.*, 2025

### Data 
- **data/**
  - **catalogs/yellowstone.events.2024.csv** - UUSS events in the Yellowstone region through 2023
  - **catalogs/yellowstone.arrivals.gains.2024.csv** - Arrivals associated with the events, includes station gain information
  - **catalogs/yellowstone.events.ypml-v5.2024.csv** - Info for events with a YP21 M_L computed
  - **catalogs/yellowstone.amps.ypml-v5.2024.csv** - Arrival info the YP21 events
  - **features/p_features.2024.csv** - waveform features for UUSS P arrivals
  - **features/s_features.2024.csv** - waveform features for UUSS S arrivals
- Format UUSS catalogs for use in [magnitute recalibration](https://github.com/sgjholt/ynp_local_magnitude_recalibration) using [reformatAmpCatalogs.ipynb](notebooks/dataprocessing/reformatAmpCatalogs.ipynb)
- Features created using [mlmodels.features](https://github.com/uofuseismo/mlmodels.git)
  - P features generated with [pMagnitude.py](scripts/pMagnitude.py)
  - S features generated with [sMagnitude.py](scripts/sMagnitude.py)
- Train and test splits made using [make_feature_splits.py](scripts/make_feature_splits.py)
  
### Methods
- Recursive Feature Elimination 
  - Main implementation in [feature_selection.CustomRFECV](src/feature_selection.py)
  - Run for P arrival data in [rfecvAllPStations.ipynb](notebooks/feature_selection/rfecvAllPStations.ipynb)
  - Run of S arrival data in [rfecvAllSStations.ipynb](notebooks/feature_selection/rfecvAllSStations.ipynb)
- Final SVR models
  - P models produced in [trainPModelsWithSelectedFeatures.ipynb](notebooks/final_models/trainPModelsWithSelectedFeatures.ipynb)
  - S models produced in [trainSModelsWithSelectedFeatures.ipynb](notebooks/final_models/trainSModelsWithSelectedFeatures.ipynb)
- Single Gradient Boosted Tree models
  - using all P arrivals in [allPStationsSimpleBoostedTree.ipynb](notebooks/results/allPStationsSimpleBoostedTree.ipynb)
  - using all S arrivals in [allSStationsSimpleBoostedTree.ipynb](notebooks/results/allSStationsSimpleBoostedTree.ipynb)
- Examine event location feature importance in [examineDistanceCorrections.ipynb](notebooks/results/examineDistanceCorrections.ipynb)
- Analysis of WY.YDC P model in [notebooks/results/YDC](notebooks/results/YDC)

### Figures
- Figure 1 produced in [featureStationMaps.ipynb](notebooks/figures/featureStationMaps.ipynb)
- Figure 2 produced in [ModelResultsP.ipynb](notebooks/results/ModelResultsP.ipynb)
- Figure 3 produced with Adobe Illustrator in [psuedoCode_v2.ai](figures)
- Figure 4 produced in  [RFE_example_fig.ipynb](notebooks/feature_selection/RFE_example_fig.ipynb)
- Figure 5 produced in [combinePandSFeatureSelectionHeatmaps.ipynb](notebooks/feature_selection/combinePandSFeatureSelectionHeatmaps.ipynb)
- Figure 6 produced in [summaryFigsR2Boxplots.ipynb](notebooks/figures/summaryFigsR2Boxplots.ipynb)
- Figure 7 produced in [summaryFigsResiduals.ipynb](notebooks/figures/summaryFigsResiduals.ipynb)
- Figure 8 produced in [combinePandSPredictions.ipynb](notebooks/results/combinePandSPredictions.ipynb)
- FIgure 9 produced in [summaryFigsResiduals.ipynb](notebooks/figures/summaryFigsResiduals.ipynb)