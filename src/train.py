
from utils import CrossValidation as cv
import numpy as np
import joblib
import os


class TrainStationModel():
    def __init__(self,
                 station,
                 phase,
                 feature_split_dict) -> None:
        self.station = station
        self.phase = phase
        self.feature_split_dict = feature_split_dict
        self.X_train = self.feature_split_dict['X_train']
        self.y_train = self.feature_split_dict['y_train']

    def eval_all_splits(self,
                        final_pipeline):
        # Get the datasets
        X_train = ['X_train']
        y_train = self.feature_split_dict['y_train']
        X_test = self.feature_split_dict['X_test']
        y_test = self.feature_split_dict['y_test']
        X_holdout, y_holdout = None, None
        if 'X_holdout' in self.feature_split_dict.keys():
            X_holdout = self.feature_split_dict['X_holdout']
            y_holdout = self.feature_split_dict['y_holdout']

        results_dict = {}

        # Evaluate the training dataset
        yhat_train, r2_train, rmse_train = self.eval_model(final_pipeline,
                                                           X_train,
                                                           y_train)
        results_dict['train_r2'] = np.round(r2_train, 3)
        results_dict['train_rmse'] = np.round(rmse_train, 3)
        # Evaluate the testing dataset
        yhat_test, r2_test, rmse_test = self.eval_model(final_pipeline,
                                                        X_test,
                                                        y_test)
        results_dict['test_r2'] = np.round(r2_test, 3)
        results_dict['test_rmse'] = np.round(rmse_test, 3)

        if X_holdout is not None:
            yhat_holdout, r2_holdout, rmse_holdout = self.eval_model(final_pipeline,
                                                                     X_holdout,
                                                                     y_holdout)
            results_dict['holdout_r2'] = np.round(r2_holdout, 3)
            results_dict['holdout_rmse'] = np.round(rmse_holdout, 3)
            all_yhat = (yhat_train, yhat_test, yhat_holdout)
        else:
            all_yhat = (yhat_train, yhat_test)

        return all_yhat, results_dict

    def train_model_with_cv(self,
                            search,
                            final_pipeline,
                            model_selector_fn=None,
                            model_selector_tol=0):

        gs_results, final_params, results_dict = self.select_hyperparameters(self.X_train,
                                                                             self.y_train,
                                                                             search,
                                                                             model_selector_fn,
                                                                             model_selector_tol)

        final_pipeline.set_params(**final_params)
        final_pipeline.fit(self.X_train, self.y_train)

        return final_pipeline, gs_results, results_dict

    def select_hyperparameters(self,
                               search,
                               model_selector_fn=None,
                               model_selector_tol=0):

        gs_results, results_dict = self.do_hyperparameter_gridsearch(self.X_train,
                                                                     self.y_train,
                                                                     search)
        final_params = results_dict['params_best']
        if model_selector_fn is not None:
            opt_cv_model_summary = self.select_opt_model(gs_results,
                                                         model_selector_fn,
                                                         model_selector_tol)
            final_params = opt_cv_model_summary['params_sel']

            results_dict = results_dict | opt_cv_model_summary

        return gs_results, final_params, results_dict

    def save_model(self, trained_pipe, outpath):
        model = trained_pipe.get_params()['m']
        joblib.dump(model, os.path.join(
            outpath, f'{self.station}.{self.phase}.SVR.joblib'))
        try:
            scaler = trained_pipe.get_params()['scaler']
            joblib.dump(scaler, os.path.join(
                outpath, f'{self.station}.{self.phase}.scaler.joblib'))
        except:
            pass

    def save_all_predictions(self,
                            feature_dict,
                            all_yhat,
                            outdir):

        yhat_datasets = ['train', 'test', 'holdout']
        for i in range(len(all_yhat)):
            yhat = all_yhat[i]
            name = yhat_datasets[i]
            self.save_predictions_to_csv(feature_dict[f'evids_{name}'],
                                         feature_dict[f'y_{name}'],
                                         yhat,
                                         outdir,
                                         self.station,
                                         self.phase,
                                         name)

    @staticmethod
    def save_predictions_to_csv(evids, y, yhat, outdir, station, phase, split):
        cols = ['Evid', 'magnitude', 'predicted_magnitude']
        results = np.concatenate([np.expand_dims(evids, 1), 
                                    np.expand_dims(y, 1),
                                    np.expand_dims(yhat, 1)], axis=1)
        results_df = pd.DataFrame(results, columns=cols)
        results_df["Evid"] = results_df.Evid.astype(int)
        results_df.to_csv(os.path.join(outdir, f'{station}.{phase}.{split}.preds.csv'), index=False)


    @staticmethod
    def format_results_dict(self, train_results_dict, eval_results_dict):

        params_best = train_results_dict.pop('params_best')
        for param_key in params_best:
            train_results_dict[param_key] = params_best[param_key]
        try:
            params_opt = train_results_dict.pop('params_sel')
            for param_key in params_opt:
                train_results_dict[param_key] = params_opt[param_key]
        except:
            pass

        results_dict = train_results_dict | eval_results_dict
        results_dict['station'] = self.station
        results_dict['phase'] = self.station

        return results_dict

    @staticmethod
    def do_hyperparameter_gridsearch(X_train,
                                     y_train,
                                     search):

        gs_results = search.fit(X_train, y_train)

        # Get the cv results and parameters corresponding to 'best' model (highest cv mean test score)
        best_cv_mean, best_cv_std, best_cv_params = cv.get_gridsearchcv_best_results(
            gs_results)

        results_dict = {'cv_mean_best': best_cv_mean,
                        'cv_std_best': best_cv_std,
                        'params_best': best_cv_params
                        }

        return gs_results, results_dict

    @staticmethod
    def select_opt_model(gs_results,
                         model_selector_fn,
                         model_selector_tol):
        # Find the best cv ind using the model_selector func
        opt_ind = model_selector_fn(gs_results, model_selector_tol)
        # Get the cv results and parameters corresponding to this ind
        opt_cv_mean, opt_cv_std, opt_params = cv.get_cv_results_from_ind(
            opt_ind)

        results_dict = {'cv_ind_sel': opt_ind,
                        'cv_mean_sel': opt_cv_mean,
                        'cv_std_sel': opt_cv_std,
                        'params_sel': opt_params
                        }

        return results_dict

    def eval_model(self, model, X, y):
        yhat = model.predict(X)
        r2, rmse = self.get_regression_scores(y, yhat)
        return yhat, r2, rmse

    @staticmethod
    def get_regression_scores(y, yhat):
        score_r2 = r2_score(y, yhat)
        score_rmse = mean_squared_error(y, yhat, squared=False)

        return score_r2, score_rmse

    @staticmethod
    def select_cv_ind_min_C(gs_results, tol):
        min_C = np.inf
        min_ind = -1
        for ind in np.where((gs_results.best_score_ - gs_results.cv_results_['mean_test_score']) < tol)[0]:
            p = gs_results.cv_results_['params'][ind]
            if p['m__C'] < min_C:
                min_ind = ind
                min_C = p['m__C']

        return min_ind