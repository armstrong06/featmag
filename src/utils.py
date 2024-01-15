import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


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
    def do_gridsearchcv(gs, Xtrain, ytrain, Xtest):
        """Fit the gridsearch (gs) and make the test predictions"""
        gs_results = gs.fit(Xtrain, ytrain)
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
