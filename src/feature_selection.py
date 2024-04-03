import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score
import time
from sklearn.base import clone

from src.utils import CrossValidation, score_comparison_func, select_N_one_standard_error


class IntrinsicFeatureSelection():

    @staticmethod
    def f_reg_feature_selection(X, y):
        # configure to select all features
        fs = SelectKBest(score_func=f_regression, k='all')
        # learn relationship from training data
        fs.fit(X, y)
        # transform train input data
        fs.transform(X)

        return fs

    @staticmethod
    def compute_multiple_station_feature_scores(n_features, 
                                                stat_feat_dict, 
                                                stat_meta_dict, 
                                                fs_method,
                                                discrete_bool=None):
        scores = np.full((n_features, len(stat_feat_dict.keys())), -1, dtype=float)
        station_order = []
        for i, station in enumerate(stat_feat_dict.keys()):
            fs_method_args = [stat_feat_dict[station]['X_train'],
                              stat_meta_dict[station]['y_train']]
            if discrete_bool is not None:
                fs_method_args.append(discrete_bool)
            fs = fs_method(*fs_method_args)
            scores[:, i] = fs.scores_
            station_order.append(station)
        station_order = np.array(station_order)
        assert np.all(scores > -1)
        return scores, station_order

    @staticmethod
    def feature_rankings_for_individual_stations(scores):
        # Rankings in ascending order (lower number is more important feature)
        rankings = np.full_like(scores.T, -1, dtype='int')
        for stat_ind in range(scores.shape[1]):
            rankings[stat_ind, :] = np.argsort(-1*scores[:, stat_ind])

        return rankings

    @staticmethod
    def rank_features_across_stations(scores, stat_rankings):
        feat_rankings = np.full_like(scores, -1, dtype='int')
        for feat_ind in range(stat_rankings.shape[1]):
            feat_rankings[feat_ind, :] = np.where(stat_rankings == feat_ind)[1]

        return feat_rankings

    @staticmethod
    def mutual_reg_feature_selection(X, y):
        # configure to select all features
        fs = SelectKBest(score_func=mutual_info_regression, k='all')
        # learn relationship from training data
        fs.fit(X, y)
        # transform train input data
        fs.transform(X)
        
        return fs

    # Function that will use the discrete mask - shouldn't really matter b/c only one
    # discrete feature
    @staticmethod
    def mutual_reg_feature_selection_discrete(X, y, discrete_feature_bool):
        def mi_discrete_features(X, y):
            return mutual_info_regression(X, y, discrete_features=discrete_feature_bool)

        # configure to select all features
        fs = SelectKBest(score_func=mi_discrete_features, k='all')
        # learn relationship from training data
        fs.fit(X, y)
        # transform train input data
        fs.transform(X)
        
        return fs

    @staticmethod
    def make_discrete_feature_bool(X_train, 
                                   feature_names,
                                   discrete_perc_thresh=1):
        # Make boolean mask of whether features are discrete or not
        # I don't think this is very important b/c I don't expect signal_dominant_freq to be 
        # very well correlated anyway. That is the only feature getting set to discrete here. 
        # I also tried making lat, lon, and depth discrete but then the MI calculation took forever
        discrete_feat_bool = []
        for i in range(len(feature_names)):
            n_uniq = np.unique(X_train[:, i]).shape[0]
            perc = (n_uniq/X_train.shape[0])*100
            print(f'{feature_names[i]} {perc:0.2f}')
            discrete = False
            if perc < discrete_perc_thresh:
                discrete = True
            discrete_feat_bool .append(discrete)

        return discrete_feat_bool

    @staticmethod
    def MI_filter_func(X, y, subsets, K, verbose=False, input_inds=None):
        if input_inds is None:
            input_inds = np.arange(0, X.shape[1])

        other_inds = input_inds[np.isin(input_inds, 
                                        np.concatenate(subsets), 
                                        invert=True)]
        filtered_inds = np.zeros((len(subsets), K), dtype=int)
        for i, subset in enumerate(subsets):
            # filter the subset to only contain values in the input_inds
            subset = subset[np.isin(subset, input_inds)]
            Xi = X[:, subset]
            yi = y[:]
            filtered_inds[i, :] = subset[np.argsort(-1*IntrinsicFeatureSelection.mutual_reg_feature_selection(Xi, yi).scores_)[0:K]]
        new_inds = np.sort(np.concatenate([other_inds, filtered_inds.flatten()]))
        return filtered_inds, new_inds

class SequentialFeatureSelection():

    @staticmethod
    def sequential_cv_N(X,
                y,
                predictor_model,
                param_grid,
                feature_ids_to_select,
                required_feature_ids=None,
                predictor_scaler=True,
                scoring_method='r2',
                score_func=r2_score,
                n_jobs=1,
                cv_folds_outer=10,
                cv_folds_inner=5,
                n_outer_repeats=1,
                cv_random_state=2652124,
                larger_score_is_better=True,
                intrinsic_filter_func=None,
                feature_inds_to_filter=None,
                intrinsic_filter_K=5,
                verbose = False
                ):
        """
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

        cv_outer = RepeatedKFold(n_splits=cv_folds_outer, 
                                n_repeats=n_outer_repeats, 
                                random_state=cv_random_state)

        inner_grid_search, cv_inner = CrossValidation.setup_cv(predictor_model, 
                                                                param_grid, 
                                                                model_scaler=predictor_scaler, 
                                                                scoring_method=scoring_method, 
                                                                n_jobs=n_jobs, 
                                                                cv_folds=cv_folds_inner, 
                                                                cv_random_state=cv_random_state, 
                                                                refit_model=True)

        N = len(feature_ids_to_select)
        if intrinsic_filter_func is not None:
            N = N - np.isin(np.concatenate(feature_inds_to_filter), feature_ids_to_select).sum() + len(feature_inds_to_filter)*intrinsic_filter_K
        N_required = 0
        if required_feature_ids is not None:
            N_required = len(required_feature_ids)
        # Array to store the test performance values for each fold and each N
        # output of forward selection is N+1
        cv_N_scores = np.full((cv_folds_outer, N + 1), 0, dtype=float)
        # Store the scores for each of the forward selection iterations 
        cv_all_feature_scores = np.full((cv_folds_outer, N, N), 0, dtype=float)
        # Store the best features for each fold (array of arrays)
        selected_feats_order = np.zeros((cv_folds_outer, N), dtype=int)
        if_K_selected_feats = None
        if intrinsic_filter_func is not None:
            if_K_selected_feats = np.full((cv_folds_outer, 
                                            len(feature_inds_to_filter), 
                                            intrinsic_filter_K), -1)
        start_time = time.time()
        # Loop over the outer folds
        for i, splits_inds in enumerate(cv_outer.split(X)):
            feature_ids_to_select_i = np.copy(feature_ids_to_select)
            # Get the folds training and testing data
            train_ix, test_ix = splits_inds
            Xi_train, Xi_test = X[train_ix, :], X[test_ix, :]
            yi_train, yi_test = y[train_ix], y[test_ix]

            # Filter features based on intrinsic information like mutual information
            if intrinsic_filter_func is not None:
                print(f"Starting features to select from {feature_ids_to_select_i}")
                filtered_subsets, feature_ids_to_select_i = intrinsic_filter_func(Xi_train, 
                                                            yi_train, 
                                                            feature_inds_to_filter, 
                                                            intrinsic_filter_K,
                                                            input_inds=feature_ids_to_select_i)
                if_K_selected_feats[i, :, :] = filtered_subsets
                print(f"reducing features to {feature_ids_to_select_i}")



            selected_features_ids, selected_test_scores, all_test_scores = SequentialFeatureSelection.do_forward_selection(Xi_train,
                                                                                            yi_train,
                                                                                            inner_grid_search,
                                                                                            feature_ids_to_select_i,
                                                                                            X_test=Xi_test,
                                                                                            y_test=yi_test,
                                                                                            score_func=score_func,
                                                                                            larger_score_is_better=larger_score_is_better,
                                                                                            required_feature_ids=required_feature_ids,
                                                                                            verbose=verbose)
            # Don't save the required features (unless I updated it to rank the required features)
            selected_feats_order[i, :] = selected_features_ids[N_required:]
            cv_N_scores[i, :] = selected_test_scores
            cv_all_feature_scores[i, :, :] = all_test_scores
            # Print the best N (in addition to required features) for the fold
            print(f"Fold {i}: N={np.argmax(selected_test_scores)}, test_score={np.max(selected_test_scores):0.3f}")

        print(f"total time: {time.time()-start_time:0.2f} s")
        # Set the final n as the one with the best mean performance over all folds
        average_N_scores = cv_N_scores.mean(axis=0)
        if larger_score_is_better:
            final_N = np.argmax(average_N_scores)
            final_N_score_avg = np.max(average_N_scores)
        else:
            final_N = np.argmin(average_N_scores)
            final_N_score_avg = np.min(average_N_scores)

        print(f"Selected number of features: {final_N} (avg. score of {final_N_score_avg:0.2f})")

        results = {"N_scores": cv_N_scores,
                    "selected_feats":selected_feats_order,
                    "intrinsic_K_feature_selection":if_K_selected_feats,
                    "all_sequential_scores":cv_all_feature_scores}

        return final_N, results
    
    @staticmethod
    def do_forward_selection(X_train,
                             y_train,
                             inner_grid_search,
                             feature_ids_to_select,
                             X_test=None,
                             y_test=None,
                             score_func=r2_score,
                             larger_score_is_better=True,
                             required_feature_ids=None,
                             early_stopping_tol=-1,
                             verbose=False):
        
        assert not np.any(np.isin(required_feature_ids, 
                              feature_ids_to_select)), ValueError("Required feature cannot be in the features to select")

        n_features_to_select_from = len(feature_ids_to_select)
        feat_selection = np.arange(n_features_to_select_from) #np.copy(feature_inds_to_select)

        cnt = 0
        best_score = 0        
        starting_feature_size = 0
        selected_test_scores = []
        all_test_scores = np.full((n_features_to_select_from, n_features_to_select_from), np.nan)
        selected_features_ids = [] 

        if required_feature_ids is not None:
            starting_feature_size = len(required_feature_ids)
            selected_features_ids = list(required_feature_ids)
            print(selected_features_ids)
            cnt += starting_feature_size
            X_req = np.copy(X_train[:, required_feature_ids])
            X_req_test = None
            if X_test is not None:
                X_req_test = np.copy(X_test[:, required_feature_ids])
            gs_results, yhat = CrossValidation.do_gridsearchcv(inner_grid_search,
                                                                X_req,
                                                                y_train,
                                                                Xtest=X_req_test)
            if y_test is not None:
                score = score_func(y_test, yhat)
            else:
                score, cv_std, params = CrossValidation.get_gridsearchcv_best_results(gs_results)
            selected_test_scores.append(score)
            best_score = score
        else:
            selected_test_scores.append(0)

        for it in range(n_features_to_select_from):
            for feature_ind in feat_selection:
                feature_id = feature_ids_to_select[feature_ind]
                feature_sub = [feature_id]
                X_sub_test = None
                if cnt == 0:
                    X_sub = np.copy(X_train[:, feature_id:feature_id+1])
                    if X_test is not None:
                        X_sub_test = np.copy(X_test[:, feature_id:feature_id+1])
                else:
                    feature_sub = np.concatenate([selected_features_ids, [feature_id]])
                    X_sub = np.copy(X_train[:, feature_sub])
                    if X_test is not None:
                        X_sub_test = np.copy(X_test[:, feature_sub])
                    #print(feature_sub)

                gs_results, yhat = CrossValidation.do_gridsearchcv(inner_grid_search,
                                                                X_sub,
                                                                y_train,
                                                                Xtest=X_sub_test)
                if y_test is not None:
                    score = score_func(y_test, yhat)
                else:
                    score, cv_std, params = CrossValidation.get_gridsearchcv_best_results(gs_results)
                all_test_scores[it, feature_ind] = score
                if verbose:
                    print(feature_sub, score)

            if larger_score_is_better:
                feat_to_add_ind = np.nanargmax(all_test_scores[it, :])
                best_it_score = np.nanmax(all_test_scores[it, :])
            else:
                feat_to_add_ind = np.nanargmin(all_test_scores[it, :])
                best_it_score = np.nanmin(all_test_scores[it, :])

            selected_features_ids.append(feature_ids_to_select[feat_to_add_ind])
            selected_test_scores.append(best_it_score)
            
            feat_selection = np.delete(feat_selection, 
                                       np.where(feat_selection==feat_to_add_ind)[0][0])
            
            cnt += 1

            if not score_comparison_func(best_score, 
                                     best_it_score, 
                                     larger_score_is_better,
                                     tol=early_stopping_tol) and it < n_features_to_select_from-1:
                print('Stopping early')
                all_test_scores = all_test_scores[:it+1, :]
                break

            if score_comparison_func(best_score, 
                                     best_it_score, 
                                     larger_score_is_better):
                best_score = best_it_score
        

        return selected_features_ids, selected_test_scores, all_test_scores
    
    @staticmethod
    def do_forward_selection_cv(X,
                                y,
                                feature_ids_to_select,
                                cv_outer,
                                inner_grid_search,
                                scoring_method,
                                larger_score_is_better=True,
                                required_feature_ids=None,
                                early_stopping_tol=-1,
                                verbose=False,
                                n_jobs=1):
        
        assert not np.any(np.isin(required_feature_ids, 
                              feature_ids_to_select)), ValueError("Required feature cannot be in the features to select")

        n_features_to_select_from = len(feature_ids_to_select)
        feat_selection = np.arange(n_features_to_select_from) #np.copy(feature_inds_to_select)

        cnt = 0
        best_mean_score = 0        
        starting_feature_size = 0
        # mean, min, max
        selected_test_scores_stats = []
        all_test_scores = np.full((n_features_to_select_from, n_features_to_select_from, 3), np.nan)
        selected_features_ids = [] 

        if required_feature_ids is not None:
            starting_feature_size = len(required_feature_ids)
            selected_features_ids = list(required_feature_ids)
            cnt += starting_feature_size
            X_req = np.copy(X[:, required_feature_ids])
            cv_scores = cross_validate(inner_grid_search, 
                                        X_req, 
                                        y, 
                                        scoring=scoring_method, 
                                        cv=cv_outer, 
                                        return_estimator=False,
                                        n_jobs=n_jobs)
            mean_cv_score = np.mean(cv_scores['test_score'])
            min_cv_score = np.min(cv_scores['test_score'])
            max_cv_score = np.max(cv_scores['test_score'])
            selected_test_scores_stats.append([mean_cv_score, min_cv_score, max_cv_score])
            best_mean_score = mean_cv_score
            print(selected_features_ids, mean_cv_score)
        else:
            selected_test_scores_stats.append([0, 0, 0])

        for it in range(n_features_to_select_from):
            for feature_ind in feat_selection:
                feature_id = feature_ids_to_select[feature_ind]
                feature_sub = feature_id
                if cnt == 0:
                    X_sub = np.copy(X[:, feature_id:feature_id+1])
                else:
                    feature_sub = np.concatenate([selected_features_ids, [feature_id]])
                    X_sub = np.copy(X[:, feature_sub])

                cv_scores = cross_validate(inner_grid_search, 
                                            X_sub, 
                                            y, 
                                            scoring=scoring_method, 
                                            cv=cv_outer, 
                                            return_estimator=False,
                                            n_jobs=n_jobs)
                mean_cv_score = np.mean(cv_scores['test_score'])
                min_cv_score = np.min(cv_scores['test_score'])
                max_cv_score = np.max(cv_scores['test_score'])                
                all_test_scores[it, feature_ind, :] = [mean_cv_score, min_cv_score, max_cv_score]
                if verbose:
                    print(feature_sub, mean_cv_score)


            if larger_score_is_better:
                feat_to_add_ind = np.nanargmax(all_test_scores[it, :, 0])
                best_it_score = np.nanmax(all_test_scores[it, :, 0])
            else:
                feat_to_add_ind = np.nanargmin(all_test_scores[it, :, 0])
                best_it_score = np.nanmin(all_test_scores[it, :, 0])

            selected_features_ids.append(feature_ids_to_select[feat_to_add_ind])
            selected_test_scores_stats.append(all_test_scores[it, feat_to_add_ind, :])
            
            feat_selection = np.delete(feat_selection, 
                                       np.where(feat_selection==feat_to_add_ind)[0][0])
            
            cnt += 1

            if not score_comparison_func(best_mean_score, 
                                     best_it_score, 
                                     larger_score_is_better,
                                     tol=early_stopping_tol) and it < n_features_to_select_from-1:
                print('Stopping early')
                all_test_scores = all_test_scores[:it+1, :]
                break

            if score_comparison_func(best_mean_score, 
                                     best_it_score, 
                                     larger_score_is_better):
                best_mean_score = best_it_score
        
        selected_test_scores_stats = np.concatenate(selected_test_scores_stats).reshape(it+2, 3)

        return selected_features_ids, selected_test_scores_stats, all_test_scores

class UnsupervisedFeatureSelection():
    @staticmethod
    def remove_highly_correlated_features(X, thresh=0.75):
        # Get correlation values below the diagonal
        feature_corr_tril = np.tril(pd.DataFrame(X).corr(), -1)
        feature_corr = pd.DataFrame(X).corr()
        dropped_feature_inds = []
        for i in range(feature_corr.shape[0]):
            max_corr = np.nanmax(feature_corr_tril)
            if max_corr < thresh:
                break
            A, B = np.unravel_index(np.nanargmax(feature_corr_tril), feature_corr.shape)
            A_corrs = np.delete(feature_corr.iloc[A].values, [A, B])
            B_corrs = np.delete(feature_corr.iloc[B].values, [A, B])
            drop_ind = B
            if np.nanmean(A_corrs) > np.nanmean(B_corrs):
                drop_ind = A
            dropped_feature_inds.append(drop_ind)
            feature_corr.iloc[drop_ind] = np.nan
            feature_corr[drop_ind] = np.nan
            feature_corr_tril[:, drop_ind] = np.nan
            feature_corr_tril[drop_ind, :] = np.nan

        kept_features = np.isin(np.arange(0, feature_corr.shape[0]), dropped_feature_inds, invert=True)

        return kept_features
    
class CustomRFECV():
    @staticmethod
    def get_estimator_feature_importance(estimator_pipeline):
        """Make the importance_getter argument for RFECV given the estimator model type"""
        # Do RFECV to select the optimal number of features
        if 'feature_importances_' in dir(estimator_pipeline['m']):
            importance = estimator_pipeline['m'].feature_importances_
        elif 'coef_' in dir(estimator_pipeline['m']) or type(estimator_pipeline['m']).__name__ == 'Lasso':
            importance = estimator_pipeline['m'].coef_[0]
        else:
            raise ValueError(
                'estimator_model must have coef_ or feature_importances_ attribute')

        return importance

    @staticmethod
    def apply_feats_transforms(X, feats_tranform_dict):
        Xt = np.copy(X)
        for ind in feats_tranform_dict.keys():
            assert ind < Xt.shape[1], 'col index is greater than the number of cols'
            Xt[:, ind] = feats_tranform_dict[ind](Xt[:, ind])
            assert ~np.array_equal(
                Xt[:, ind], X[:, ind]), f'No transform happened for col {ind}'
        return Xt

    @staticmethod
    def do_rfecv(X,
                y,
                estimator_model,
                predictor_model,
                param_grid,
                estimator_scaler=True,
                predictor_scaler=True,
                estimator_params_grid=None,
                scoring_method='r2',
                score_func=r2_score,
                n_jobs=1,
                cv_folds_outer=10,
                cv_folds_inner=5,
                n_outer_repeats=1,
                cv_random_state=2652124,
                larger_score_is_better=True,
                intrinsic_filter_func=None,
                feature_inds_to_filter=None,
                intrinsic_filter_K=5
                ):
        """
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

        cv_outer = RepeatedKFold(n_splits=cv_folds_outer, 
                                n_repeats=n_outer_repeats, 
                                random_state=cv_random_state)

        inner_grid_search, cv_inner = CrossValidation.setup_cv(predictor_model, 
                                                                param_grid, 
                                                                model_scaler=predictor_scaler, 
                                                                scoring_method=scoring_method, 
                                                                n_jobs=n_jobs, 
                                                                cv_folds=cv_folds_inner, 
                                                                cv_random_state=cv_random_state, 
                                                                refit_model=True)
       

        estimator_grid_search = None
        if estimator_params_grid is not None:
            estimator_grid_search, estimator_cv = CrossValidation.setup_cv(estimator_model, 
                                                                estimator_params_grid, 
                                                                model_scaler=estimator_scaler, 
                                                                scoring_method=scoring_method, 
                                                                n_jobs=n_jobs, 
                                                                cv_folds=cv_folds_inner, 
                                                                cv_random_state=cv_random_state+1, 
                                                                refit_model=False)
        N = X.shape[-1]
        if intrinsic_filter_func is not None:
            N = N - np.concatenate(feature_inds_to_filter).shape[0] + len(feature_inds_to_filter)*intrinsic_filter_K
        
        # Array to store the test performance values for each fold and each N
        rfecv_N_scores = np.full((cv_folds_outer*n_outer_repeats, N), 0, dtype=float)
        # Store the best features for each fold (array of arrays)
        rfecv_selected_feats = []
        if_K_selected_feats = None
        if intrinsic_filter_func is not None:
            if_K_selected_feats = np.full((cv_folds_outer*n_outer_repeats, 
                                           len(feature_inds_to_filter), 
                                           intrinsic_filter_K), -1)
        start_time = time.time()
        # Loop over the outer folds
        selected_inds = np.arange(N)
        for i, splits_inds in enumerate(cv_outer.split(X)):
            # Get the folds training and testing data
            train_ix, test_ix = splits_inds
            Xi_train, Xi_test = X[train_ix, :], X[test_ix, :]
            yi_train, yi_test = y[train_ix], y[test_ix]

            # Filter features based on intrinsic information like mutual information
            if intrinsic_filter_func is not None:
                filtered_subsets, selected_inds = intrinsic_filter_func(Xi_train, 
                                                                        yi_train, 
                                                                        feature_inds_to_filter, 
                                                                        intrinsic_filter_K)
                if_K_selected_feats[i, :, :] = filtered_subsets
                print(f"reducing features to {len(selected_inds)}")
                Xi_train = Xi_train[:, selected_inds]
                Xi_test = Xi_test[:, selected_inds]

            best_score = -1
            best_N = -1
            best_feats = None
            feature_inds_ranked = CustomRFECV.rank_features_by_importance(Xi_train,
                                                                          yi_train,
                                                                          estimator_model,
                                                                          estimator_scaler,
                                                                          estimator_grid_search=estimator_grid_search)
            # Loop over all possible number of features (1,...,N)
            for n_feats in range(1, N+1):
                feature_subset = feature_inds_ranked[0:n_feats]
                Xfeat_train = Xi_train[:, feature_subset]
                Xfeat_test = Xi_test[:, feature_subset]
                # Do a cv grid search over the predictor models hyperparameters when using n features 
                # from the folds training set. Train a model with the best hyperparameters and the 
                # n features from the full folds training set as long as refit_model=True in GridSearchCV
                pred_gs_result, yhat = CrossValidation.do_gridsearchcv(clone(inner_grid_search),
                                                                        Xfeat_train,
                                                                        yi_train,
                                                                        Xfeat_test)
                # Evaluate the predictor on the folds test set
                predictor_score = score_func(yi_test, yhat)
                # Save the score
                rfecv_N_scores[i, n_feats-1] = predictor_score

                # Keep track of the best features for every fold
                if score_comparison_func(best_score, predictor_score, larger_score_is_better):
                    best_score = predictor_score
                    best_N = n_feats
                    best_feats = selected_inds[feature_subset]

            # Store the best feature set for every fold
            print(f"Fold {i}: N={best_N}, test_score={best_score:0.3f}")
            rfecv_selected_feats.append(best_feats)

        total_time = time.time()-start_time
        print(f"total time: {total_time:0.2f} s")
        # Set the final n as the one with the best mean performance over all folds
        average_N_scores = rfecv_N_scores.mean(axis=0)
        if larger_score_is_better:
            final_N = np.argmax(average_N_scores) + 1
            final_N_score_avg = np.max(average_N_scores)
        else:
            final_N = np.argmin(average_N_scores) + 1
            final_N_score_avg = np.min(average_N_scores)

        oste_N = select_N_one_standard_error(average_N_scores, larger_score_is_better)
        oste_score = average_N_scores[oste_N-1]
        print(f"Selected number of features: {final_N} (avg. score of {final_N_score_avg:0.2f}); 1 STE: N={oste_N} (avg. {oste_score:0.2f})")

        results = {"best_N":final_N,
                   "best_N_score":final_N_score_avg,
                   'oste_N': oste_N,
                   'oste_N_score': oste_score,
                   "N_scores": rfecv_N_scores,
                   "selected_feats":rfecv_selected_feats,
                   "intrinsic_K_feature_selection":if_K_selected_feats,
                   'total_time':total_time}

        return results


    @staticmethod
    def rank_features_by_importance(X_train,
                                    y_train,
                                    estimator_model,
                                    estimator_scaler,
                                    estimator_grid_search):
        
        estimator_pipeline = CrossValidation.make_simple_pipeline(estimator_model, 
                                                                estimator_scaler)
        # Tune estimator hyperparameters if necessary
        if estimator_grid_search is not None:
            estimator_grid_search = clone(estimator_grid_search)
            estimator_grid_search = estimator_grid_search.fit(X_train, y_train)
            estimator_params = estimator_grid_search.best_params_
            print(f"Using {estimator_params} for the estimator model")
            estimator_pipeline = estimator_pipeline.set_params(**estimator_params)
            print(estimator_pipeline['m'])

        fselector = estimator_pipeline.fit(X_train, y_train)
        feat_importances = CustomRFECV.get_estimator_feature_importance(fselector)
        feature_inds_ranked = feat_importances.argsort()[::-1][:len(feat_importances)]
        return feature_inds_ranked
    
    @staticmethod
    def custom_rfe(X_train, 
                   y_train,
                   estimator_pipeline,
                   n_feats,
                   predictor_hyperparam_grid_search,
                   X_test=None):
        
        # Select n features from the folds training set
        fselector = estimator_pipeline.fit(X_train, y_train)
        feat_importances = CustomRFECV.get_estimator_feature_importance(fselector)
        feature_inds_ranked = feat_importances.argsort()[::-1][:n_feats]
        Xfeat_train = X_train[:, feature_inds_ranked]
        Xfeat_test = None
        if X_test is not None:
            Xfeat_test = X_test[:, feature_inds_ranked]
        # Do a cv grid search over the predictor models hyperparameters when using n features 
        # from the folds training set. Train a model with the best hyperparameters and the 
        # n features from the full folds training set as long as refit_model=True in GridSearchCV
        pred_gs_result, yhat = CrossValidation.do_gridsearchcv(predictor_hyperparam_grid_search,
                                                                Xfeat_train,
                                                                y_train,
                                                                Xfeat_test)
        return feature_inds_ranked, pred_gs_result, yhat

    @staticmethod
    def get_final_N_features(X,
                    y,
                    rfecv_results_dict,
                    estimator_model,
                    estimator_scaler,
                    predictor_gs,
                    filtered_feat_inds = None
                    ):
        estimator_pipeline = CrossValidation.make_simple_pipeline(estimator_model, 
                                                             estimator_scaler)
    
        N_results = {}
        N_feats_to_use = {'best':rfecv_results_dict['best_N'], 
                        'oste': rfecv_results_dict['oste_N']}
        if N_feats_to_use['best'] == N_feats_to_use['oste']:
            N_feats_to_use['oste'] = None
            N_results['oste'] = None

        print(N_feats_to_use)
        for N_key in N_feats_to_use.keys():
            N_i = N_feats_to_use[N_key]
            if N_i is None:
                continue
                
            feature_subset, gs_results, _ = CustomRFECV.custom_rfe(X, 
                                                            y,
                                                            estimator_pipeline,
                                                            N_i,
                                                            predictor_gs)
            cv_mean, cv_std, params = CrossValidation.get_gridsearchcv_best_results(gs_results)
            print(f'{N_i}: CV Mean: {cv_mean:0.2f}, CV STD: {cv_std:0.2f}')
            if filtered_feat_inds is not None:
                feature_subset = filtered_feat_inds[feature_subset]
            N_results[N_key] = {'N': N_i,
                            'selected_feature_inds':feature_subset,
                            'pred_cv_mean':cv_mean,
                            'pred_cv_std':cv_std,
                            'pred_cv_params':params}
            
        return N_results
    

    @staticmethod
    def get_final_N_features_estimator_tuning(X,
                                                y,
                                                rfecv_results_dict,
                                                estimator_model,
                                                estimator_scaler,
                                                estimator_gs,
                                                predictor_gs,
                                                filtered_feat_inds = None
                                                ):
        
        feature_inds_ranked = CustomRFECV.rank_features_by_importance(X,
                                                                    y,
                                                                    estimator_model,
                                                                    estimator_scaler,
                                                                    estimator_gs)
        
        N_results = {}
        N_feats_to_use = {'best':rfecv_results_dict['best_N'], 
                        'oste': rfecv_results_dict['oste_N']}
        if N_feats_to_use['best'] == N_feats_to_use['oste']:
            N_feats_to_use['oste'] = None
            N_results['oste'] = None

        print(N_feats_to_use)
        for N_key in N_feats_to_use.keys():
            N_i = N_feats_to_use[N_key]
            if N_i is None:
                continue
                
            feature_subset = feature_inds_ranked[0:N_i]
            print(feature_subset)
            gs_results, yhat = CrossValidation.do_gridsearchcv(predictor_gs,
                                                                X[:, feature_subset],
                                                                y)
            cv_mean, cv_std, params = CrossValidation.get_gridsearchcv_best_results(gs_results)
            print(f'{N_i}: CV Mean: {cv_mean:0.2f}, CV STD: {cv_std:0.2f}')
            if filtered_feat_inds is not None:
                feature_subset = filtered_feat_inds[feature_subset]
            N_results[N_key] = {'N': N_i,
                            'selected_feature_inds':feature_subset,
                            'pred_cv_mean':cv_mean,
                            'pred_cv_std':cv_std,
                            'pred_cv_params':params}
            
        return N_results
    
# Below are functions for looking at which features are important

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
    
    @staticmethod
    def get_rfecv_important_feature_counts(results_dict, 
                                    feature_names,
                                    oste_feats=False,
                                    larger_score_is_better=True):
        important_feats_df_dict = {}
        for stat in results_dict.keys():
            stat_feature_dict =  results_dict[stat]['selected_feats']
            stat_N_scores = np.array(results_dict[stat]['N_scores'])
            selected_feats_bool = np.zeros((len(stat_feature_dict), len(feature_names)), dtype=bool)
            for i, fold_feats in enumerate(stat_feature_dict):
                if oste_feats:
                    oste_N = select_N_one_standard_error(stat_N_scores[i, :],
                                                        larger_score_is_better=larger_score_is_better)
                    fold_feats = fold_feats[0:oste_N-1]
                bool_arr = np.zeros(len(feature_names), dtype=bool)
                bool_arr[fold_feats] = True
                selected_feats_bool[i, :] = bool_arr
            feat_usage_df = CustomRFECV.count_feature_usage(selected_feats_bool, feature_names)
            important_feats_df_dict[stat] = feat_usage_df[['Feature', 'cvcnt']].set_index('Feature')

        return important_feats_df_dict

    @staticmethod
    def get_selected_feature_counts(results_dict, 
                                    feature_names,
                                    oste_feats=False,
                                    larger_score_is_better=True):
        important_feats_df_dict = {}
        N_feats_key = 'best'
        if oste_feats:
            N_feats_key = 'oste'
        for stat in results_dict.keys():
            stat_feature_dict =  results_dict[stat]
            if stat_feature_dict[N_feats_key] is not None:
                fold_feats = stat_feature_dict[N_feats_key]['selected_feature_inds']
            else:
                print(stat, "N best == N oste")
                fold_feats = stat_feature_dict["best"]['selected_feature_inds']
            bool_arr = np.zeros(len(feature_names), dtype=bool)
            bool_arr[fold_feats] = True
            feat_usage_df = CustomRFECV.count_feature_usage([bool_arr], feature_names)
            important_feats_df_dict[stat] = feat_usage_df[['Feature', 'cvcnt']].set_index('Feature')

        return important_feats_df_dict
    
    @staticmethod
    def make_feature_count_df(results_dict, 
                          feature_names,
                          feature_counting_func,
                          oste_feats=False,
                          larger_score_is_better=True,
                          filter_zeros=True):
        feature_counts_dict = feature_counting_func(results_dict,
                                                        feature_names,
                                                        oste_feats=oste_feats,
                                                        larger_score_is_better=larger_score_is_better)
        # Combine the counts from each station
        mega_df = None
        for key in feature_counts_dict.keys():
            key_dict = feature_counts_dict[key].rename(columns={'cvcnt': key})
            if mega_df is None:
                mega_df = key_dict
            else:
                mega_df = mega_df.merge(key_dict, on='Feature')
        
        mega_df = mega_df.loc[feature_names][mega_df.columns.sort_values()]
        if filter_zeros:
            mega_df_filtered = mega_df.loc[~(mega_df==0).all(axis=1)]
            return mega_df_filtered
        return mega_df

    @staticmethod
    def get_feature_cnts_across_stats(cnts_df):
        return cnts_df.T.sum().reset_index().rename(columns={0:'cnt'}).sort_values('cnt', ascending=False)
