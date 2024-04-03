import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import json
import sys
import os
import joblib
from sklearn.base import clone
# make paths above 'notebooks/' visible for local imports. Have to use '.' not 
#'..' when debugging in VS code b/c cwd is featmags not scripts
# +----------------------------------------------------------------------------+
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.processing import GatherFeatureDatasets
from src.train import TrainStationModel, OptModelSelectionMethods
from src.plotting import plot_r2_heatmap
from src.utils import CrossValidation as cv

proc = GatherFeatureDatasets(is_p=True)

data_dir = '/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits'
train = pd.read_csv(f'{data_dir}/p.train.csv')
test = pd.read_csv(f'{data_dir}/p.test.csv')
holdout = pd.read_csv(f'{data_dir}/p.2022.csv')
ynr_feature_dict, feature_names = proc.process_station_datasets('YNR',
                                                                train,
                                                                test,
                                                                holdout_df=holdout,
                                                                scaler=False,
                                                                linear_model=False,
                                                                source_dist_type='dist')
station_feature_dict = {'YNR': ynr_feature_dict}
selected_features = ['amp_1', 'amp_2', 'signal_dominant_amplitude',
                     'signal_max_amplitude', 'signal_variance',
                     'noise_variance', 'source_depth_km',
                     'source_receiver_distance_logkm',
                     'source_receiver_back_azimuth_deg']

selected_feat_dict, selected_feature_names = proc.filter_station_dict_features(station_feature_dict,
                                                                               feature_names,
                                                                               selected_features)

# CV Parameters
cv_random_state = 2652124
cv_folds_inner = 5
svr_C_range = [10]  # 10**np.arange(-3, 5, dtype=float)
svr_gamma_range = [0.1]  # 10**np.arange(-4, 3, dtype=float)
param_grid = [
    {'m__C': svr_C_range, 'm__gamma': svr_gamma_range},
]
model = SVR(kernel='rbf')
model_scaler = True
scoring_method = 'r2'
n_jobs_inner = 5

# Model parameters
outdir = '/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/p_models'
phase = 'P'
model_selector = OptModelSelectionMethods.select_cv_ind_min_C
model_selector_tol = 0.005

stations = ['YNR']
results_dict_list = []

for station in stations:
    # Set up the grid search
    search, cv_inner = cv.setup_cv(model,
                                   param_grid,
                                   model_scaler=model_scaler,
                                   scoring_method=scoring_method,
                                   n_jobs=n_jobs_inner,
                                   cv_folds=cv_folds_inner,
                                   cv_random_state=cv_random_state,
                                   refit_model=False)

    trainer = TrainStationModel(station,
                                phase,
                                selected_feat_dict[station])

    opt_pipeline = cv.make_simple_pipeline(clone(model), model_scaler)
    gs_results, train_results_dict = trainer.train_model_with_cv(search,
                                                                 opt_pipeline,
                                                                 model_selector_fn=model_selector,
                                                                 model_selector_tol=model_selector_tol
                                                                 )

    train_results_dict