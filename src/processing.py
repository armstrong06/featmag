from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrainTestSplit():

    def print_feature_df_event_counts(phase_df, phase, ev_df):
        print(f"There are {len(phase_df.drop_duplicates('event_identifier'))} events and \
{len(phase_df)} arrivals in the {phase.upper()} feature catalog between \
{ev_df[ev_df['Evid'].isin(phase_df['event_identifier'].unique())]['Date'].min()} and \
{ev_df[ev_df['Evid'].isin(phase_df['event_identifier'].unique())]['Date'].max()}")

    def get_feature_event_info(feature_df, event_df, phase='P', include_ev_col=None):
        evs = event_df[event_df['Evid'].isin(
            feature_df['event_identifier'].unique())].sort_values('Date')
        print(f"{phase} event count: {evs.shape[0]}")
        return evs

    def date_train_test_split(ev_df, train_cutoff):
        train_evids = ev_df[ev_df['Date'] < train_cutoff]['Evid']
        test_evids = ev_df[ev_df['Date'] >= train_cutoff]['Evid']
        return train_evids, test_evids

    def evid_train_test_split(ev_df, time_cutoff=None, train_frac=0.8, random_state=843823):
        if time_cutoff is not None:
            ev_df = ev_df[ev_df['Date'] < time_cutoff]
        train_evids, test_evids = train_test_split(ev_df['Evid'].unique(
        ), test_size=1-train_frac, train_size=train_frac, random_state=random_state)

        return train_evids, test_evids

    def get_features_by_evid(feature_df, train_evids, test_evids):
        train_feats = feature_df[feature_df['event_identifier'].isin(
            train_evids)]
        test_feats = feature_df[feature_df['event_identifier'].isin(
            test_evids)]

        return train_feats, test_feats

    def print_split_percentage(split_evids, all_evids, split_type='Train', phase='P'):
        print(
            f"{phase} {split_type} size: {(split_evids.shape[0]/all_evids.shape[0])*100:0.2f} %")

    def get_station_train_test_counts(feats_train_df, feats_test_df):
        return feats_train_df.groupby('station')['station'].count().rename('cnt').reset_index().merge(feats_test_df.groupby('station')['station'].count().rename('cnt').reset_index(),
                                                                                                      how='outer', on='station').sort_values('cnt_x', ascending=False).rename(columns={'cnt_x': 'cnt_train',
                                                                                                                                                                                       'cnt_y': 'cnt_test'})

    def compute_min_test_examples(min_train, train_frac, phase='P'):
        frac = round((1-train_frac)/train_frac, 2)
        min_test = round(min_train*frac)
        print(f"{phase} train min: {min_train}, test min: {min_test}, total example min: {min_test+min_train}")
        return min_test

    def get_stations_with_min_examples(counts_df, min_train, min_test, phase='P'):
        filtered_df = counts_df[(counts_df['cnt_train'] > min_train) & (
            counts_df['cnt_test'] > min_test)]
        print(
            f"{phase} stations meeting the min. criteria: {filtered_df.shape[0]}")
        return filtered_df

    def get_stations_close_to_criteria(counts_df, min_train, min_test, phase='P'):
        filtered_df = counts_df[(np.nansum([counts_df['cnt_train'], counts_df['cnt_test']], axis=0) > (min_test+min_train)) &
                                ((counts_df['cnt_test'] < min_test) | np.isnan(counts_df['cnt_test']) | (counts_df['cnt_train'] < min_train))]
        print(
            f"{phase} stations close to min. criteria: {filtered_df.shape[0]}")
        return filtered_df

    def get_station_feature_time_span(feats_df, station_df, evid_df):
        return feats_df[feats_df.station.isin(station_df['station'])].groupby('station').apply(lambda x: evid_df[evid_df['Evid'].isin(x['event_identifier'])]['Date'].describe().loc[['min', 'max']])

    def add_YP21_magnitude_to_features(feature_df, event_df, phase='P'):
        original_feature_cnt = feature_df.shape[0]
        original_ev_count = feature_df['event_identifier'].unique().shape[0]
        feature_df = feature_df.merge(
            event_df[['Evid', 'Event-Mean-YPML-S']], how='inner', left_on='event_identifier', right_on='Evid')
        assert ~np.any(np.isnan(feature_df['Event-Mean-YPML-S']))
        assert ~np.any(np.isnan(feature_df['arrival_identifier']))
        print(
            f"The original number of {phase} features: {original_feature_cnt} ({original_ev_count} events) \nThe number of {phase} features with a YP21 mag: {feature_df.shape[0]} ({feature_df['event_identifier'].unique().shape[0]} events)")
        return feature_df

    def filter_feature_stations(features_df, station_df, phase='P'):
        features_filt = features_df[features_df['station'].isin(
            station_df['station'])]
        print(
            f"Original {phase} size: {features_df.shape[0]} ({features_df['station'].unique().shape[0]} stations) \nFiltered {phase} size: {features_filt.shape[0]} ({features_filt['station'].unique().shape[0]} stations)")
        return features_filt


class GatherFeatureDatasets():
    def __init__(self, is_p) -> None:
        if is_p:
            self.feature_maker = PFeatures
        else:
            self.feature_maker = SFeatures

        self.compute_feature_matrix = self.feature_maker.compute_feature_matrix

    def get_X_y(self,
                df,
                freq_max=18,
                scaler=True,
                source_dist_type='dist',
                linear_model=True,
                target_column='Event-Mean-YPML-S'):

        X, scaler, feature_names = self.compute_feature_matrix(df,
                                                               freq_max=freq_max,
                                                               scaler=scaler,
                                                               source_dist_type=source_dist_type,
                                                               linear_model=linear_model)
        y = df[target_column].values

        assert X.shape[0] == y.shape[0], 'X size does not match y size'
        print(f'X shape: {X.shape}, y shape: {y.shape}')

        return X, y, scaler, feature_names

    def process_all_stations_datasets(self,
                                      train_df,
                                      test_df,
                                      holdout_df=None,
                                      freq_max=18,
                                      scaler=True,
                                      source_dist_type='all',
                                      linear_model=True,
                                      target_column='Event-Mean-YPML-S'):
        all_station_features_dict = {}
        all_station_meta_dict = {}

        feature_names = None
        for station in train_df['station'].unique():
            feat_dict, meta_dict, stat_feat_names = self.process_station_datasets(station,
                                                                          train_df,
                                                                          test_df,
                                                                          holdout_df=holdout_df,
                                                                          freq_max=freq_max,
                                                                          scaler=scaler,
                                                                          source_dist_type=source_dist_type,
                                                                          linear_model=linear_model,
                                                                          target_column=target_column)
            all_station_features_dict[station] = feat_dict
            all_station_meta_dict[station] = meta_dict

            if feature_names is None:
                feature_names = stat_feat_names

        return all_station_features_dict, all_station_meta_dict, feature_names

    def process_station_datasets(self,
                                 station,
                                 train_df,
                                 test_df=None,
                                 holdout_df=None,
                                 freq_max=18,
                                 scaler=True,
                                 source_dist_type='all',
                                 linear_model=True,
                                 target_column='Event-Mean-YPML-S'):
        print(station)
        strain = self.filter_by_station(train_df, station)
        s_X_train, s_y_train, s_scaler, feature_names = self.get_X_y(strain,
                                                                     freq_max=freq_max,
                                                                     scaler=scaler,
                                                                     source_dist_type=source_dist_type,
                                                                     linear_model=linear_model,
                                                                     target_column=target_column)
        feat_dict = {'scaler': s_scaler,
                     'X_train': s_X_train,
                    }
        
        meta_dict = {'scaler': s_scaler,
                     'y_train': s_y_train,
                     'evids_train': strain['Evid'],
                     }
       
        if test_df is not None:
            stest = self.filter_by_station(test_df, station)
            s_X_test, s_y_test, _, _ = self.get_X_y(stest,
                                                    scaler=s_scaler,
                                                    freq_max=freq_max,
                                                    source_dist_type=source_dist_type,
                                                    linear_model=linear_model,
                                                    target_column=target_column)
            
            feat_dict['X_test'] = s_X_test
            meta_dict['y_test'] = s_y_test
            meta_dict['evids_test'] = stest['Evid']

        if holdout_df is not None:
            # Not every station has a holdout set
            X_holdout, y_holdout, evids_holdout = None, None, None
            if station in holdout_df['station'].unique():
                sholdout = self.filter_by_station(holdout_df, station)
                X_holdout, y_holdout, _, _ = self.get_X_y(sholdout,
                                                          scaler=s_scaler,
                                                          freq_max=freq_max,
                                                          source_dist_type=source_dist_type,
                                                          linear_model=linear_model,
                                                          target_column=target_column)
                evids_holdout = sholdout['Evid']
            feat_dict['X_holdout'] = X_holdout
            meta_dict['y_holdout'] = y_holdout
            meta_dict['evids_holdout'] = evids_holdout

        # feat_dict = {station: feat_dict}
        # meta_dict = {station: meta_dict}

        return feat_dict, meta_dict, feature_names

    # def process_station_split(self,
    #                           split_name,
    #                           station,
    #                           df,
    #                           scaler,
    #                           freq_max,
    #                           source_dist_type,
    #                           linear_model,
    #                           target_column):

    #     ssplit = self.filter_by_station(df, station)
    #     X, y, s_scaler, feature_names = self.get_X_y(ssplit,
    #                                                  scaler=scaler,
    #                                                  freq_max=freq_max,
    #                                                  source_dist_type=source_dist_type,
    #                                                  linear_model=linear_model,
    #                                                  target_column=target_column)
    #     stat_dict = {
    #         f'X_{split_name}': X,
    #         f'y_{split_name}': y,
    #         f'evids_{split_name}': ssplit['Evid'],
    #     }

    #     return stat_dict, s_scaler, feature_names

    @staticmethod
    def get_feature_subset_correct_order(all_col_names, subset_col_names):
        return all_col_names[np.where(np.isin(all_col_names, subset_col_names))[0]]

    @staticmethod
    def filter_station_dict_features(station_feature_dict,
                                     all_feature_col_names,
                                     subset_feature_col_names):
        assert np.all(np.isin(subset_feature_col_names, all_feature_col_names)), \
            'Subset column names must be in the existing column names'

        feature_subset_cols = np.where(
            np.isin(all_feature_col_names, subset_feature_col_names))[0]
        filtered_station_feature_dict = {}
        for station in station_feature_dict.keys():
            print(station)
            s_dict = station_feature_dict[station]
            s_X_train = s_dict['X_train'][:, feature_subset_cols]
            s_X_test = s_dict['X_test'][:, feature_subset_cols]
            stat_dict = {'X_train': s_X_train,
                         'X_test': s_X_test,
                         }
            if 'X_holdout' in s_dict.keys():
                s_X_holdout = None
                holdout_shape = 0
                if s_dict['X_holdout'] is not None:
                    s_X_holdout = s_dict['X_holdout'][:, feature_subset_cols]
                    holdout_shape = s_X_holdout.shape
                stat_dict['X_holdout'] = s_X_holdout
                print(
                    f'X_train: {s_X_train.shape}, X_test: {s_X_test.shape}, X_holdout: {holdout_shape}')

            else:
                print(f'X_train: {s_X_train.shape}, X_test: {s_X_test.shape}')

            filtered_station_feature_dict[station] = stat_dict

        return filtered_station_feature_dict, all_feature_col_names[feature_subset_cols]

    @staticmethod
    def filter_by_station(df, stat):
        df = df[df['station'] == stat].copy(deep=True)
        assert len(df['station'].unique()) == 1

        return df

    def get_feature_plot_names(self, freq_max=18, source_dist_type='all'):
        return self.feature_maker.make_feature_plot_names(freq_max=freq_max,
                                                          source_dist_type=source_dist_type)

    def get_feature_names(self,
                          freq_max=18,
                            source_dist_type='dist',
                            linear_model=True):
        return self.feature_maker.make_feature_names(freq_max=freq_max, 
                                                     source_dist_type=source_dist_type,
                                                     linear_model=linear_model)

class PFeatures():

    @staticmethod
    def compute_feature_matrix(df,
                               freq_max=18,
                               scaler=True,
                               source_dist_type='dist',
                               linear_model=True):
        # Loosely speaking empirical magnitudes look like:
        # M = log10(A) + Q(Delta)
        # where A is the amplitude and Q a distance dependent correction term.
        # Additionally, een log10 and log amounts to a scalar
        # that a machine can learnthe difference betw.
        # Basically, I'm interested in features that:
        #   (1) Measure size in, potentially, different amplitudes.
        # different `passbands' deviates from the noise,
        n_rows = len(df)
        n_columns = 2*freq_max + 9
        if source_dist_type == 'all':
            n_columns += 2
        X = np.zeros([n_rows, n_columns])
        # Get a proxy on size
        column_names = []

        # These are effectively amplitude ratios.  Note,
        # log(a/b) = log(a) - log(b)
        def amp_ratio(freq, column_names):
            freq = f'{int(freq)}.00'
            column_names.append(f'amp_ratio_{freq[:-3]}')
            return np.log(df[f'avg_signal_{freq}']) - np.log(df[f'avg_noise_{freq}'])

        for i in range(freq_max):
            X[:, i] = amp_ratio(i+1, column_names)

        # Look at amplitudes
        def amplitudes(freq, column_names):
            freq = f'{int(freq)}.00'
            column_names.append(f'amp_{freq[:-3]}')
            return np.log(df[f'avg_signal_{freq}'])

        for j in range(1, freq_max+1):
            X[:, i+j] = amplitudes(j, column_names)

        i += j

        # Frequency and max amplitude
        X[:, i+1] = np.log(df['signal_dominant_frequency'])
        X[:, i+2] = np.log(df['signal_dominant_amplitude'])
        # X[:,i+3] = np.log(df['noise_dominant_frequency'])
        # X[:,i+4] = np.log(df['noise_dominant_amplitude'])

        # Time-based features: Look at max amplitudes of noise/signal
        X[:, i+3] = np.log(df['noise_maximum_value'] -
                           df['noise_minimum_value'])
        X[:, i+4] = np.log(df['signal_maximum_value'] -
                           df['signal_minimum_value'])
        X[:, i+5] = np.log(df['signal_variance'])
        X[:, i+6] = np.log(df['noise_variance'])

        # Source/recv distance (take log to flatten this)
        X[:, i+7] = df['source_depth_km']

        column_names += ['signal_dominant_frequency',
                         'signal_dominant_amplitude',
                         'noise_max_amplitude',
                         'signal_max_amplitude',
                         'signal_variance',
                         'noise_variance',
                         'source_depth_km']

        if source_dist_type == 'coord':
            X[:, i+8] = df['source_latitude']
            X[:, i+9] = df['source_longitude']
            column_names += ['source_latitude', 'source_longitude']
        elif source_dist_type == 'dist':
            X[:, i+8] = np.log(df['source_receiver_distance_km'])
            column_names.append('source_receiver_distance_logkm')
            if linear_model:
                X[:, i +
                    9] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
                column_names.append('source_receiver_back_azimuth_sine')
            else:
                X[:, i+9] = df['source_receiver_back_azimuth']
                column_names.append('source_receiver_back_azimuth_deg')
        elif source_dist_type == 'all':
            X[:, i+8] = df['source_latitude']
            X[:, i+9] = df['source_longitude']
            X[:, i+10] = np.log(df['source_receiver_distance_km'])
            column_names += ['source_latitude',
                             'source_longitude',
                             'source_receiver_distance_logkm']
            if linear_model:
                X[:, i +
                    11] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
                column_names.append('source_receiver_back_azimuth_sine')
            else:
                X[:, i+11] = df['source_receiver_back_azimuth']
                column_names.append('source_receiver_back_azimuth_deg')
        else:
            raise ValueError('source_dist_type must be in [dist, coord, all]')

        assert len(column_names) == n_columns

        column_names = np.array(column_names)
        # Standardize features
        if (scaler):
            scaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            return X, scaler, column_names

        return X, False, column_names

    @staticmethod
    def make_feature_names(freq_max=18,
                            source_dist_type='dist',
                            linear_model=True):
        
        if source_dist_type not in ['all', 'coord', 'dist']:
            raise ValueError('source_dist_type must be in [dist, coord, all]')
        names = []

        for i in range(freq_max):
            names.append(f'amp_ratio_{i+1}')

        for i in range(freq_max):
            names.append(f'amp_{i+1}')

        names += ['signal_dominant_frequency',
                    'signal_dominant_amplitude',
                    'noise_max_amplitude',
                    'signal_max_amplitude',
                    'signal_variance',
                    'noise_variance',
                    'source_depth_km']
        
        coord_names = ['source_latitude', 'source_longitude']
        sr_linear = ['source_receiver_distance_logkm', 
                     'source_receiver_back_azimuth_sine']
        sr_nonlin = ['source_receiver_distance_logkm', 
                     'source_receiver_back_azimuth_deg']
        
        if source_dist_type == 'coord' or source_dist_type == 'all':
            names += coord_names
        
        if source_dist_type == 'dist' or source_dist_type == 'all':
            if linear_model:
                names += sr_linear
            else:
                names += sr_nonlin

        return np.array(names)

    @staticmethod
    def make_feature_plot_names(freq_max=18, source_dist_type='all'):
        # Make list of shorter feature names for plots
        alt_names = []

        for i in range(freq_max):
            alt_names.append(f'ratio {i+1}')

        for i in range(freq_max):
            alt_names.append(f'amp. {i+1}')

        alt_names += ['sig. dom. freq.', 'sig. dom. amp.',
                      'noise max. amp.', 'sig. max. amp.', 'sig. var.',
                      'noise var.', 'depth']

        if source_dist_type in ['all', 'coord']:
            alt_names += ['lat.', 'long.']

        if source_dist_type in ['all', 'dist']:
            alt_names += ['distance', 'back az.']

        return np.array(alt_names)


class SFeatures():

    @staticmethod
    def compute_feature_matrix(df,
                               freq_max=18,
                               scaler=True,
                               source_dist_type='dist',
                               linear_model=True,
                               w_r=0.5):
        # Loosely speaking empirical magnitudes look like:
        # M = log10(A) + Q(Delta)
        # where A is the amplitude and Q a distance dependent correction term.
        # Additionally, een log10 and log amounts to a scalar
        # that a machine can learnthe difference betw.
        # Basically, I'm interested in features that:
        #   (1) Measure size in, potentially, different amplitudes.
        # different `passbands' deviates from the noise,
        assert w_r <= 1.0 and w_r >= 0.0, 'w_r is invalid must be in [0, 1]'
        w_t = 1 - w_r

        n_rows = len(df)
        n_columns = 2*freq_max + 9
        if source_dist_type == 'all':
            n_columns += 2
        X = np.zeros([n_rows, n_columns])

        column_names = []

        # These are effectively amplitude ratios.  Note,
        # log(a/b) = log(a) - log(b)
        def amp_ratio(freq, column_names):
            freq = f'{int(freq)}.00'
            column_names.append(f'amp_ratio_{freq[:-3]}')
            ratio = w_r*(np.log(df[f'radial_avg_signal_{freq}']) - np.log(df[f'radial_avg_noise_{freq}'])) \
            + w_t*(np.log(df[f'transverse_avg_signal_{freq}']) - np.log(df[f'transverse_avg_noise_{freq}']))
            return ratio

        for i in range(freq_max):
            X[:, i] = amp_ratio(i+1, column_names)

        # Look at amplitudes
        def amplitudes(freq, column_names):
            freq = f'{int(freq)}.00'
            column_names.append(f'amp_{freq[:-3]}')
            amp = w_r*np.log(df[f'radial_avg_signal_{freq}']) \
                + w_t*np.log(df[f'transverse_avg_signal_{freq}']) 
            return amp
        
        for j in range(1, freq_max+1):
            X[:, i+j] = amplitudes(j, column_names)

        i += j

        # Frequency and max amplitude
        X[:, i+1] = w_r*np.log(df['radial_signal_dominant_frequency']) \
            + w_t*np.log(df['transverse_signal_dominant_frequency'])
        X[:, i+2] = w_r*np.log(df['radial_signal_dominant_amplitude']) \
            + w_t*np.log(df['transverse_signal_dominant_amplitude'])

        # Time-based features: Look at max amplitudes of noise/signal
        X[:, i+3] = w_r*(np.log(df['radial_noise_maximum_value'] - df['radial_noise_minimum_value'])) \
            + w_t*(np.log(df['transverse_noise_maximum_value'] - df['transverse_noise_minimum_value']))
        X[:, i+4] = w_r*(np.log(df['radial_signal_maximum_value'] - df['radial_signal_minimum_value'])) \
            + w_t*(np.log(df['transverse_signal_maximum_value'] - df['transverse_signal_minimum_value']))
        X[:, i+5] = w_r*np.log(df['radial_signal_variance']) \
            + w_t*np.log(df['transverse_signal_variance'])
        X[:, i+6] = w_r*np.log(df['radial_noise_variance']) \
            + w_t*np.log(df['transverse_noise_variance'])

        # Source/recv distance (take log to flatten this)
        X[:, i+7] = df['source_depth_km']

        column_names += ['signal_dominant_frequency',
                         'signal_dominant_amplitude',
                         'noise_max_amplitude',
                         'signal_max_amplitude',
                         'signal_variance',
                         'noise_variance',
                         'source_depth_km']
        # TODO: should change dist_types to "relative", "absolute", "both"
        if source_dist_type == 'coord':
            X[:, i+8] = df['source_latitude']
            X[:, i+9] = df['source_longitude']
            column_names += ['source_latitude', 'source_longitude']
        elif source_dist_type == 'dist':
            X[:, i+8] = np.log(df['source_receiver_distance_km'])
            column_names.append('source_receiver_distance_logkm')
            if linear_model:
                X[:, i +
                    9] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
                column_names.append('source_receiver_back_azimuth_sine')
            else:
                X[:, i+9] = df['source_receiver_back_azimuth']
                column_names.append('source_receiver_back_azimuth_deg')
        elif source_dist_type == 'all':
            X[:, i+8] = df['source_latitude']
            X[:, i+9] = df['source_longitude']
            X[:, i+10] = np.log(df['source_receiver_distance_km'])
            column_names += ['source_latitude',
                             'source_longitude',
                             'source_receiver_distance_logkm']
            if linear_model:
                X[:, i +
                    11] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
                column_names.append('source_receiver_back_azimuth_sine')
            else:
                X[:, i+11] = df['source_receiver_back_azimuth']
                column_names.append('source_receiver_back_azimuth_deg')
        else:
            raise ValueError('source_dist_type must be in [dist, coord, all]')

        assert len(column_names) == n_columns

        column_names = np.array(column_names)
        # Standardize features
        if (scaler):
            scaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            return X, scaler, column_names

        return X, False, column_names

    @staticmethod
    def make_feature_plot_names(freq_max=18, source_dist_type='all'):
        # Make list of shorter feature names for plots
        alt_names = []

        for i in range(freq_max):
            alt_names.append(f'ratio {i+1}')

        for i in range(freq_max):
            alt_names.append(f'amp. {i+1}')

        alt_names += ['sig. dom. freq.', 'sig. dom. amp.',
                      'noise max. amp.', 'sig. max. amp.', 'sig. var.',
                      'noise var.', 'depth']

        if source_dist_type in ['all', 'coord']:
            alt_names += ['lat.', 'long.']

        if source_dist_type in ['all', 'dist']:
            alt_names += ['distance', 'back az.']

        return np.array(alt_names)

    @staticmethod
    def make_feature_names(freq_max=18,
                            source_dist_type='dist',
                            linear_model=True):
        
        if source_dist_type not in ['all', 'coord', 'dist']:
            raise ValueError('source_dist_type must be in [dist, coord, all]')
        names = []

        for i in range(freq_max):
            names.append(f'amp_ratio_{i+1}')

        for i in range(freq_max):
            names.append(f'amp_{i+1}')

        names += ['signal_dominant_frequency',
                    'signal_dominant_amplitude',
                    'noise_max_amplitude',
                    'signal_max_amplitude',
                    'signal_variance',
                    'noise_variance',
                    'source_depth_km']
        
        coord_names = ['source_latitude', 'source_longitude']
        sr_linear = ['source_receiver_distance_logkm', 
                     'source_receiver_back_azimuth_sine']
        sr_nonlin = ['source_receiver_distance_logkm', 
                     'source_receiver_back_azimuth_deg']
        
        if source_dist_type == 'coord' or source_dist_type == 'all':
            names += coord_names
        
        if source_dist_type == 'dist' or source_dist_type == 'all':
            if linear_model:
                names += sr_linear
            else:
                names += sr_nonlin

        return np.array(names)