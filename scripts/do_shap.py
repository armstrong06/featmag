import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import sys
import os
import shap
import argparse
# make paths above 'notebooks/' visible for local imports.
# +----------------------------------------------------------------------------+
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.processing import GatherFeatureDatasets
from src.utils import CrossValidation

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--station", type=str, help="station code")
argParser.add_argument("-p", "--is_p", action=argparse.BooleanOptionalAction, help="is P data")
args = argParser.parse_args()

is_p = args.is_p
station = args.station

datadir = '/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits'

proc = GatherFeatureDatasets(is_p=is_p)
if is_p:
    phase = "P"
    train = pd.read_csv(os.path.join(datadir, 'p.train.csv'))
else:
    phase = "S"
    train = pd.read_csv(os.path.join(datadir, 's.train.csv'))

print(f"Using phase={phase} for station {station}")
outdir = f"/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/SHAP/{phase}_data/{station}"

if not os.path.exists(outdir):
   print(f"Making {outdir}")
   os.makedirs(outdir)
else:
    raise ValueError(f"{outdir} already exists...")

feature_dict, meta_dict, feature_names = proc.process_station_datasets(station,
                                                                        train, 
                                                                        scaler=False,
                                                                        linear_model=False,
                                                                        source_dist_type='dist')

feature_plot_names = proc.get_feature_plot_names(source_dist_type='dist')

## Model Settings ##
cv_random_state=2652124
C = 1
gamma = 0.1
cv_folds = 10
# The main model to fit
predictor_model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.1)
# Boolean - True if data for model needs to be scaled 
model_scaler = True
##

cv = KFold(n_splits=cv_folds,
            shuffle=True,
            random_state=cv_random_state)

full_train_X = feature_dict["X_train"]
full_train_y = meta_dict["y_train"]
fold = 0
all_shap_values = []
all_test_inds = []
for train_i_inds, test_i_inds in cv.split(full_train_X):
    print("fold:", fold)
    train_i_X = full_train_X[train_i_inds, :]
    train_i_y = full_train_y[train_i_inds]
    test_i_X = full_train_X[test_i_inds, :]
    test_i_y = full_train_y[test_i_inds]
    print(train_i_X.shape, test_i_X.shape)
    print(train_i_y.shape, test_i_y.shape)
    pipeline = CrossValidation.make_simple_pipeline(predictor_model, model_scaler)
    fold_model = pipeline.fit(train_i_X, train_i_y)
    
    # This is the primary interface that autoselects the algorithm to use. 
    # It chooses permutation for this model (explainer.__class__.__name__)
    # explainer = shap.Explainer(fold_model['m'].predict, 
    #                             fold_model['scaler'].transform(train_i_X))

    # Just explicitly use the permutation algorithm  
    explainer = shap.explainers.Permutation(fold_model['m'].predict, 
                                            fold_model['scaler'].transform(train_i_X),
                                            feature_names=feature_plot_names,
                                            seed=cv_random_state+1)
    
    shap_values = explainer(fold_model['scaler'].transform(test_i_X))
    
    with open(os.path.join(outdir, f'{station}.{phase}.shap.permutation.fold{fold}.explainer'), 'wb') as file:
        explainer.save(file)

    np.savez(os.path.join(outdir, f"{station}.{phase}.shap.permutation.fold{fold}.values"),
        values=shap_values.values, base_values=shap_values.base_values, data=shap_values.data, 
        feature_names=feature_plot_names, test_inds=test_i_inds)
    
    all_shap_values.append(shap_values.values)
    all_test_inds.append(test_i_inds)

    fold += 1