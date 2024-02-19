import numpy as np
import json
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

def write_dict_to_json(filename, wdict):
    print('Writing', filename)
    with open(filename, 'w') as fp:
        json.dump(wdict, fp, indent=4, cls=NumpyEncoder)

def combine_prediction_files(stations_list,
                             datapath,
                             phase,
                             split):
    pred_df_arr = []
    for stat in stations_list:
        df = pd.read_csv(os.path.join(datapath, f'{stat}.{phase}.preds.{split}.csv'))
        df['station'] = stat
        pred_df_arr.append(df)

    df = pd.concat(pred_df_arr)
    return df

def combine_p_and_s_predictions(p_preds_df, s_preds_df):
    p_preds_df['phase'] = 'P'
    s_preds_df['phase'] = 'S'
    combined_df = pd.concat([p_preds_df, s_preds_df])
    print('Original number of predictions:', combined_df.shape[0])
    combined_df = combined_df.groupby('Evid').filter(lambda x: True if len(x['phase'].unique()) == 2 else False)
    print('Filtered number of predictions:',combined_df.shape[0])
    return combined_df

def compute_network_avg_prediction(df):
    avg_df = df.groupby('Evid')[['magnitude', 
                                      'predicted_magnitude']].mean('predicted_magnitude').reset_index()
    std_df = df.groupby('Evid')[['predicted_magnitude']].std().reset_index().rename(
        columns={'predicted_magnitude':'predicted_magntiude_std'})
    assert np.array_equal(avg_df['Evid'], std_df['Evid']), 'Evids do not mat'
    avg_df['predicted_magnitude_std'] = std_df['predicted_magntiude_std']
    return avg_df

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

@staticmethod
def score_comparison_func(s_old, s_new, larger_score_is_better, tol=0):
    """Check if a new score is better than an old score.

    Args:
        s_old (_type_): _description_
        s_new (_type_): _description_
        larger_score_is_better (_type_): _description_

    Returns:
        _type_: _description_
    """
    if larger_score_is_better:
        if s_new-s_old > tol:
            return True
    else:
        if s_new-s_old < tol:
            return True
        
    return False
    
class CrossValidation:

    @staticmethod
    def setup_cv(model,
                 param_grid,
                 model_scaler=True,
                 scoring_method='r2',
                 n_jobs=1,
                 cv_folds=5,
                 cv_random_state=2652124,
                 refit_model=True):

        cv_inner = KFold(n_splits=cv_folds,
                         shuffle=True,
                         random_state=cv_random_state)

        # If the main model needs scaled features, add to the model pipeline (m_pipe)
        # Can use this pipeline in GridCV and evaluating the final models
        m_pipe = CrossValidation.make_simple_pipeline(model, model_scaler)

        #### Define the grid search ####
        search = GridSearchCV(m_pipe,
                              param_grid=param_grid,
                              scoring=scoring_method,
                              n_jobs=n_jobs,
                              cv=cv_inner,
                              refit=refit_model)

        return search, cv_inner

    @staticmethod
    def do_gridsearchcv(gs, Xtrain, ytrain, Xtest=None):
        """Fit the gridsearch (gs) and make the test predictions"""
        gs_results = gs.fit(Xtrain, ytrain)
        yhat = None
        if Xtest is not None:
            yhat = gs_results.predict(Xtest)
        return gs_results, yhat

    @staticmethod
    def get_gridsearchcv_best_results(gs_results):
        """Return the mean, std, and model parameters from the refit GridSearchCV model"""
        cv_mean = gs_results.best_score_
        cv_std = gs_results.cv_results_[
            'std_test_score'][gs_results.best_index_]
        params = gs_results.best_params_

        return cv_mean, cv_std, params

    @staticmethod
    def get_cv_results_from_ind(gs_results, ind):
        params = gs_results.cv_results_['params'][ind]
        cv_mean = gs_results.cv_results_['mean_test_score'][ind]
        cv_std = gs_results.cv_results_['std_test_score'][ind]

        return cv_mean, cv_std, params

    @staticmethod
    def make_simple_pipeline(model, scaler):
        pipe = []
        if scaler:
            pipe.append(('scaler', StandardScaler()))

        pipe.append(('m', model))

        return Pipeline(pipe)
