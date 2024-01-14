import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
import time

from utils import CrossValidation


class IntrinsicFeatureSelection():
    pass


class RFE():
    def nested_rfecv(self,
                     X,
                     y,
                     estimator_model,
                     model,
                     param_grid,
                     estimator_scaler=True,
                     model_scaler=True,
                     scoring_method='r2',
                     score_func=r2_score,
                     n_jobs=1,
                     cv_folds_outer=10,
                     cv_folds_inner=5,
                     n_outer_repeats=1,
                     cv_random_state=2652124,
                     estimator_feats_transforms=None,
                     model_feats_transforms=None,
                     run_gridsearchcv_all_feats=False,
                     ):
        """Use SKLearn's RFECV to select the optimal number of features for each 
        outer fold using the estimator model. Use SKLearn's GridSearchCV to find the optimal
        model parameters for the main model using all features and the optimal features. 

        Args:
            X (np.array): Training features
            y (np.array): Training target
            estimator_model (_type_): SKLearn model for selecting features. Must have coef_ or feature_importances_ attribute.
            model (_type_): SKlearn model to train.
            param_grid (dict): Parameter space to search for model.
            estimator_scaler (bool, optional): If the data needs to be scaled using StandardScaler for the estimator model. Defaults to True.
            model_scaler (bool, optional): If the data needs to be scaled using StandardScaler for the model. Defaults to True.
            scoring_method(str, optional): The scoring method to use in RFECV and GridSearchCV. Defaults to 'r2'.
            score_func (function, optional): The function used to evaluate the final models in each fold. Defaults to r2_score.
            n_jobs (int, optional): The number of jobs to use in RFECV and GridSearchCV. Defaults to 1.
            cv_folds_outer (int, optional): The number of folds in the outer KFold CV loop. Defaults to 10.
            cv_folds_inner (int, optional): The number of folds to use in RFECV and GridSearchCV. Defaults to 5.
            n_outer_repeats (int, optional): The number of times to repeat the outer KFold CV. Defaults to 1.
            cv_random_state (int, optional): The random state to use for the inner and outer KFolds. Defaults to 2652124. 
            estimator_feat_transform
        Raises:
            ValueError: If the estimator model does not have oef_ or feature_importances_ attribute.

        Returns:
            dict: Dictionary containing the results from each of the outer folds.
        """

        cv_outer = RepeatedKFold(
            n_splits=cv_folds_outer, n_repeats=n_outer_repeats, random_state=cv_random_state)
        # cv_inner = KFold(n_splits=cv_folds_inner, shuffle=True, random_state=cv_random_state)

        search, cv_inner = self.setup_cv(model,
                                         param_grid,
                                         model_scaler=model_scaler,
                                         scoring_method=scoring_method,
                                         n_jobs=n_jobs,
                                         cv_folds_outer=cv_folds_outer,
                                         cv_folds_inner=cv_folds_inner,
                                         n_outer_repeats=n_outer_repeats,
                                         cv_random_state=cv_random_state)

        # Lists to store the results of outer loop
        # Store results of cross-validation and the best model when
        # using all the features - I probably do not need all of these,
        # really just the test score for comparison
        outer_cv_params_all = []
        outer_cv_mean_all = []
        outer_cv_std_all = []
        outer_test_score_all = []

        # Store the CV results and the best model when using
        # the selected subset of features
        outer_cv_params_best = []
        outer_cv_mean_best = []
        outer_cv_std_best = []
        outer_test_score_best = []

        # Store the boolean array of features that were kept
        outer_kept_feats = []
        # Store the number of kept features
        outer_n_feats = []

        #### Define the pipelines for RFECV and GridSearchCV ####
        # If the estimator model needs scaled features, add scaling to the
        # selector pipeline (s_pipe)
        # Each fold in RFECV should be scaled independently
        s_pipe = self.make_simple_pipeline(estimator_model, estimator_scaler)

        # # If the main model needs scaled features, add to the model pipeline (m_pipe)
        # # Can use this pipeline in GridCV and evaluating the final models
        # m_pipe = self.make_simple_pipeline(model, model_scaler)

        # #### Define the grid search ####
        # search = GridSearchCV(m_pipe,
        #                     param_grid=param_grid,
        #                     scoring=scoring_method,
        #                     n_jobs=n_jobs,
        #                     cv=cv_inner,
        #                     refit=True)

        start_outer = time.time()
        for i, data in enumerate(cv_outer.split(X)):
            train_ix, test_ix = data

            start_inner = time.time()

            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            # Do RFECV to select the optimal number of features
            X_rfecv = X_train
            if estimator_feats_transforms:
                X_rfecv = self.apply_feats_transforms(
                    X_train, estimator_feats_transforms)

            n_feats, best_feats = self.do_rfecv(X_rfecv,
                                                y_train,
                                                s_pipe,
                                                cv_inner,
                                                importance_getter=self.get_estimator_importance_getter(
                                                    estimator_model),
                                                scoring_method=scoring_method,
                                                n_jobs=n_jobs)

            outer_kept_feats.append(best_feats)
            outer_n_feats.append(n_feats)

            # Do model param. grid search when using all features
            if model_feats_transforms is not None:
                X_train = self.apply_feats_transforms(
                    X_train, model_feats_transforms)
                X_test = self.apply_feats_transforms(
                    X_test, model_feats_transforms)

            if run_gridsearchcv_all_feats:
                gs_results_all, yhat_all = CrossValidation.do_gridsearchcv(
                    search, X_train, y_train, X_test)
                score_all = score_func(y_test, yhat_all)
                outer_test_score_all.append(score_all)
                # Could probably remove the next 4 lines... I don't think I really need to save these
                cv_mean_all, cv_std_all, params_all = CrossValidation.get_gridsearchcv_best_results(
                    gs_results_all)
                outer_cv_mean_all.append(cv_mean_all)
                outer_cv_std_all.append(cv_std_all)
                outer_cv_params_all.append(params_all)

            # Do GridCV using the optimal number of features
            X_train = X_train[:, best_feats]
            X_test = X_test[:, best_feats]
            gs_results_best, yhat_best = self.do_gridsearchcv(
                search, X_train, y_train, X_test)
            score_best = score_func(y_test, yhat_best)
            outer_test_score_best.append(score_best)

            cv_mean_best, cv_std_best, params_best = CrossValidation.get_gridsearchcv_best_results(
                gs_results_best)
            outer_cv_mean_best.append(cv_mean_best)
            outer_cv_std_best.append(cv_std_best)
            outer_cv_params_best.append(params_best)

            end_inner = time.time()

            if run_gridsearchcv_all_feats:
                print(f'Fold {i}: test score ({n_feats} best feats): {score_best:0.3f}, test score (all feats): {score_all:0.3f}, diff: {(score_best - score_all):0.3f}, time: {end_inner-start_inner:0.2f} s, best model params: {params_best}')
            else:
                print(
                    f'Fold {i}: test score ({n_feats} best feats): {score_best:0.3f}, time: {end_inner-start_inner:0.2f} s, best model params: {params_best}')

        outer_time = (time.time() - start_outer)
        print(f'Total time: {outer_time:0.2f} s ({outer_time/60:0.2f} min)')

        results_dict = {'n_feats': np.array(outer_n_feats),
                        'optfts_bool': np.array(outer_kept_feats),
                        'test_score_optfts': np.array(outer_test_score_best),
                        'cv_mean_optfts': np.array(outer_cv_mean_best),
                        'cv_std_optfts': np.array(outer_cv_std_best),
                        'cv_params_optfts': np.array(outer_cv_params_best),
                        'test_score_allfts': np.array(outer_test_score_all),
                        'cv_mean_allfts': np.array(outer_cv_mean_all),
                        'cv_std_allfts': np.array(outer_cv_std_all),
                        'cv_params_allfts': np.array(outer_cv_params_all),
                        }

        return results_dict

    @staticmethod
    def do_rfecv(f, X, y, pipe, cv, importance_getter=None, scoring_method='r2', n_jobs=1):
        # Do RFECV to select the optimal number of features
        rfe = RFECV(pipe,
                    cv=cv,
                    scoring=scoring_method,
                    n_jobs=n_jobs,
                    importance_getter=importance_getter)

        rfe.fit(X, y)

        # Get the best features from the RFECV
        n_feats = rfe.n_features_
        best_feats = rfe.support_

        return n_feats, best_feats

    @staticmethod
    def get_estimator_importance_getter(estimator_model):
        """Make the importance_getter argument for RFECV given the estimator model type"""
        # Do RFECV to select the optimal number of features
        if 'feature_importances_' in dir(estimator_model):
            importance_getter = 'named_steps.m.feature_importances_'
        elif 'coef_' in dir(estimator_model) or type(estimator_model).__name__ == 'Lasso':
            importance_getter = 'named_steps.m.coef_'
        else:
            raise ValueError(
                'estimator_model must have coef_ or feature_importances_ attribute')

        return importance_getter

    @staticmethod
    def apply_feats_transforms(X, feats_tranform_dict):
        Xt = np.copy(X)
        for ind in feats_tranform_dict.keys():
            assert ind < Xt.shape[1], 'col index is greater than the number of cols'
            Xt[:, ind] = feats_tranform_dict[ind](Xt[:, ind])
            assert ~np.array_equal(
                Xt[:, ind], X[:, ind]), f'No transform happened for col {ind}'
        return Xt

# Below are functions for looking at which features are important for a single station

    @staticmethod
    def count_feature_usage(cv_optfts_bool, feature_names):
        """Count the number of times a feature is selected in the outer folds.

        Args:
            cv_optfts_bool (np.array): np.array or list of the boolean feature selection arrays
            feature_names (list): The names of the features

        Returns:
            pd.DataFrame: DataFrame with columns of feature names and their count.
        """
        feat_sum = (cv_optfts_bool[0])*1
        for f_i in cv_optfts_bool[1:]:
            feat_sum += (f_i)*1

        return pd.DataFrame({'Feature': feature_names, 'cvcnt': feat_sum}).sort_values('cvcnt', ascending=False)

    @staticmethod
    def import_feats_by_usage(feat_usage_df, thresh=None):
        """Get list of features that occur in at least thresh folds. 
        thresh default is 1/2 the max count."""
        if thresh is None:
            thresh = feat_usage_df['cvcnt'].max()//2

        return feat_usage_df[feat_usage_df['cvcnt'] >= thresh]['Feature'].values

    @staticmethod
    def important_feats_by_best_model(results_dict, feature_names, use_max_score=True):
        """Get the features selected in the fold with the highest test score"""
        if use_max_score:
            best_model_score = np.max(results_dict['test_score_optfts'])
        else:
            best_model_score = np.min(results_dict['test_score_optfts'])

        best_model_ind = np.where(
            results_dict['test_score_optfts'] == best_model_score)[0][0]

        return feature_names[results_dict['optfts_bool'][best_model_ind]]

    @staticmethod
    def important_feats_by_cv_mean(results_dict, feature_names, use_max_score=True):
        """Get the features selected in the fold with the highest mean cv score"""
        if use_max_score:
            best_model_score = np.max(results_dict['cv_mean_optfts'])
        else:
            best_model_score = np.min(results_dict['cv_mean_optfts'])

        best_model_ind = np.where(
            results_dict['cv_mean_optfts'] == best_model_score)[0][0]

        return feature_names[results_dict['optfts_bool'][best_model_ind]]

    @staticmethod
    def important_feats_by_largest_gain(results_dict, feature_names):
        """Get the features selected in the fold that has the largest performance gain compared
        to the model trained with all features"""
        diff = np.subtract(
            results_dict['test_score_optfts'], results_dict['test_score_allfts'])
        best_model_ind = np.argmax(diff)
        return feature_names[results_dict['optfts_bool'][best_model_ind]]

    @staticmethod
    def combine_important_features(impfeats_list, cvcnt_df=None):
        """Count the number of times a feature was selected as important using different criteria.

        Args:
            impfeats_list (list): List of lists or arrays containing feature names.
            cvcnt_df (pd.DataFrame, optional): DataFrame with the counts per fold 
            (from count_feature_usage) to merge the with. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with columns Feature (feature name) and impcnt (number of times the feature 
            occured in impfeats_list). If cvcnt_df is provided, there will also be a cvcnt column (number of times
            the feature was selected as important in the outer folds).
        """
        selectedfts, selectedcnts = np.unique(
            np.concatenate(impfeats_list), return_counts=True)
        imp_df = pd.DataFrame({'Feature': selectedfts, 'impcnt': selectedcnts}).sort_values(
            'impcnt', ascending=False)
        if cvcnt_df is not None:
            imp_df = cvcnt_df.merge(imp_df, on='Feature', how='left').fillna(
                0).astype({'impcnt': int})

        return imp_df

