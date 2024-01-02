from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

class SplitFeatures():

    def print_feature_df_event_counts(phase_df, phase, ev_df):
        print(f"There are {len(phase_df.drop_duplicates('event_identifier'))} events and \
{len(phase_df)} arrivals in the {phase.upper()} feature catalog between \
{ev_df[ev_df['Evid'].isin(phase_df['event_identifier'].unique())]['Date'].min()} and \
{ev_df[ev_df['Evid'].isin(phase_df['event_identifier'].unique())]['Date'].max()}")
        
    def get_feature_event_info(feature_df, event_df, phase='P', include_ev_col=None):
        evs = event_df[event_df['Evid'].isin(feature_df['event_identifier'].unique())].sort_values('Date')
        print(f"{phase} event count: {evs.shape[0]}")
        return evs

    def date_train_test_split(ev_df, train_cutoff):
        train_evids = ev_df[ev_df['Date'] < train_cutoff]['Evid']
        test_evids =  ev_df[ev_df['Date'] >= train_cutoff]['Evid']
        return train_evids, test_evids
        
    def evid_train_test_split(ev_df, time_cutoff=None, train_frac=0.8, random_state=843823):
        if time_cutoff is not None:
            ev_df = ev_df[ev_df['Date'] < time_cutoff]
        train_evids, test_evids = train_test_split(ev_df['Evid'].unique(), test_size=1-train_frac, train_size=train_frac, random_state=random_state)

        return train_evids, test_evids

    def get_features_by_evid(feature_df, train_evids, test_evids):
        train_feats = feature_df[feature_df['event_identifier'].isin(train_evids)]
        test_feats = feature_df[feature_df['event_identifier'].isin(test_evids)]

        return train_feats, test_feats

    def print_split_percentage(split_evids, all_evids, split_type='Train', phase='P'):
        print(f"{phase} {split_type} size: {(split_evids.shape[0]/all_evids.shape[0])*100:0.2f} %")

    def get_station_train_test_counts(feats_train_df, feats_test_df):
        return feats_train_df.groupby('station')['station'].count().rename('cnt').reset_index().merge(feats_test_df.groupby('station')['station'].count().rename('cnt').reset_index(),
                                                                                        how='outer', on='station').sort_values('cnt_x', ascending=False).rename(columns={'cnt_x':'cnt_train',
                                                                                                                                                                        'cnt_y':'cnt_test'})
    def compute_min_test_examples(min_train, train_frac, phase='P'):
        frac = round((1-train_frac)/train_frac, 2)
        min_test =  round(min_train*frac)
        print(f"{phase} train min: {min_train}, test min: {min_test}, total example min: {min_test+min_train}")
        return min_test

    def get_stations_with_min_examples(counts_df, min_train, min_test, phase='P'):
        filtered_df = counts_df[(counts_df['cnt_train'] > min_train) & (counts_df['cnt_test'] > min_test)]
        print(f"{phase} stations meeting the min. criteria: {filtered_df.shape[0]}")
        return filtered_df

    def get_stations_close_to_criteria(counts_df, min_train, min_test, phase='P'):
        filtered_df = counts_df[(np.nansum([counts_df['cnt_train'], counts_df['cnt_test']], axis=0) > (min_test+min_train)) &
                            ((counts_df['cnt_test'] < min_test) | np.isnan(counts_df['cnt_test']) | (counts_df['cnt_train'] < min_train))]
        print(f"{phase} stations close to min. criteria: {filtered_df.shape[0]}")
        return filtered_df

    def get_station_feature_time_span(feats_df, station_df, evid_df):
        return feats_df[feats_df.station.isin(station_df['station'])].groupby('station').apply(lambda x: evid_df[evid_df['Evid'].isin(x['event_identifier'])]['Date'].describe().loc[['min', 'max']])


    def add_YP21_magnitude_to_features(feature_df, event_df, phase='P'):
        original_feature_cnt = feature_df.shape[0]
        original_ev_count = feature_df['event_identifier'].unique().shape[0]
        feature_df = feature_df.merge(event_df[['Evid', 'Event-Mean-YPML-S']], how='inner', left_on='event_identifier', right_on='Evid')
        assert ~np.any(np.isnan(feature_df['Event-Mean-YPML-S']))
        assert ~np.any(np.isnan(feature_df['arrival_identifier']))
        print(f"The original number of {phase} features: {original_feature_cnt} ({original_ev_count} events) \nThe number of {phase} features with a YP21 mag: {feature_df.shape[0]} ({feature_df['event_identifier'].unique().shape[0]} events)")
        return feature_df

    def filter_feature_stations(features_df, station_df, phase='P'):
        features_filt = features_df[features_df['station'].isin(station_df['station'])]
        print(f"Original {phase} size: {features_df.shape[0]} ({features_df['station'].unique().shape[0]} stations) \nFiltered {phase} size: {features_filt.shape[0]} ({features_filt['station'].unique().shape[0]} stations)")
        return features_filt

class FeaturePlots():
    def plot_station_feature_box_whisker(df, feature, ylabel=None, sort_counts=True, thresholds=None, showfliers=True, min_y_line=None):
        fig = plt.figure(figsize=(15, 5))
        station_feat = df.groupby("station").apply(lambda x: x[feature].values).to_frame()
        station_feat.columns = ["values"]
        if sort_counts:
            station_feat.loc[:, "counts"] = station_feat.apply(lambda x: len(x["values"]), axis=1)
            station_feat = station_feat.sort_values("counts", ascending=False)
            
        if thresholds is not None:
            for thresh in thresholds: 
                thresh_ind = np.where(station_feat.counts < thresh)[0][0] + 0.5
                plt.axvline(thresh_ind, color="red", alpha=0.5)

        if min_y_line is not None:
            plt.axhline(min_y_line, color='r')
        labels = station_feat.index
        plt.boxplot(station_feat["values"].to_numpy(), showfliers=showfliers);
        plt.xticks(range(1, len(labels)+1), labels=labels, rotation="vertical");
        if ylabel is None:
            ylabel = feature
        plt.ylabel(ylabel)

    def plot_pairwise_correlations(X, y, column_names, title,
                               xticklabels=2):
        df = pd.DataFrame(X, columns=column_names)
        df['magnitude'] = y

        corr = df.corr()

        sns.set_theme(style="white")

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
                    yticklabels=1, xticklabels=xticklabels).set_title(title)

class SelectPFeatures():

    @staticmethod
    def filter_by_station(df, stat):
        df = df[df['station'] == stat].copy(deep=True)
        assert len(df['station'].unique()) == 1

        return df

    def get_X_y(self, 
                df,
                freq_max=18,
                scaler = True,
                source_dist_type='dist',
                linear_model = True,
                target_column = 'Event-Mean-YPML-S'):
    
        X, scaler, feature_names = self.compute_feature_matrix(df, 
                                                        freq_max=freq_max,
                                                        scaler = scaler,
                                                        source_dist_type=source_dist_type,
                                                        linear_model = linear_model)
        y = df[target_column]

        assert X.shape[0] == y.shape[0], 'X size does not match y size'
        print(f'X shape: {X.shape}, y shape: {y.shape}')

        return X, y, scaler, feature_names

    @staticmethod
    def make_feature_plot_names(freq_max=18):
        # Make list of shorter feature names for plots
        alt_names = []

        for i in range(freq_max):
            alt_names.append(f'ratio {i+1}')

        for i in range(freq_max):
            alt_names.append(f'amp. {i+1}')

        alt_names += ['sig. dom. freq.', 'sig. dom. amp.',
        'noise max. amp.', 'sig. max. amp.', 'sig. var.',
        'noise var.', 'depth', 'lat.', 'long.',
        'distance', 'back az.']

        return alt_names

    def process_station_datasets(self, 
                                 train_df, 
                                 test_df,
                                freq_max=18,
                                scaler = True,
                                source_dist_type='all',
                                linear_model = True,
                                target_column = 'Event-Mean-YPML-S'):
        station_feature_dict = {}
        for station in train_df['station'].unique():
            print(station)
            strain = self.filter_by_station(train_df, station)
            s_X_train, s_y_train, s_scaler, feature_names = self.get_X_y(strain, 
                                                        freq_max=freq_max,
                                                        scaler = scaler,
                                                        source_dist_type=source_dist_type,
                                                        linear_model = linear_model,
                                                        target_column=target_column)
            
            stest = self.filter_by_station(test_df, station)
            s_X_test, s_y_test, _, _ = self.get_X_y(stest, 
                                                    scaler=s_scaler,
                                                    freq_max=freq_max,
                                                    source_dist_type=source_dist_type,
                                                    linear_model = linear_model,
                                                    target_column=target_column)

            station_feature_dict[station] = {'X_train':s_X_train,
                                            'y_train':s_y_train,
                                            'scaler':s_scaler,
                                            'X_test':s_X_test,
                                            'y_test':s_y_test}
            
        return station_feature_dict, feature_names


    @staticmethod
    def filter_station_dict_features(station_feature_dict,
                                    all_feature_col_names, 
                                    subset_feature_col_names):
            assert np.all(np.isin(subset_feature_col_names, all_feature_col_names)),\
                    'Subset column names must be in the existing column names'
            
            feature_subset_cols = np.where(np.isin(all_feature_col_names, subset_feature_col_names))[0]

            filtered_station_feature_dict = {}
            for station in station_feature_dict.keys():
                    print(station)
                    s_dict = station_feature_dict[station]
                    s_X_train = s_dict['X_train'][:, feature_subset_cols]
                    s_X_test = s_dict['X_test'][:, feature_subset_cols]
                    print(f'X_train: {s_X_train.shape}, X_test: {s_X_test.shape}')
                    filtered_station_feature_dict[station] = {'X_train':s_X_train,
                                                    'y_train':s_dict['y_train'],
                                                    'X_test':s_X_test,
                                                    'y_test':s_dict['y_test']}

            return filtered_station_feature_dict

    @staticmethod
    def compute_feature_matrix(df,
                            freq_max=18,
                            scaler = True,
                            source_dist_type='dist',
                            linear_model = True):
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
        X[:,i+1] = np.log(df['signal_dominant_frequency'])
        X[:,i+2] = np.log(df['signal_dominant_amplitude'])
        # X[:,i+3] = np.log(df['noise_dominant_frequency'])
        # X[:,i+4] = np.log(df['noise_dominant_amplitude'])

        # Time-based features: Look at max amplitudes of noise/signal
        X[:,i+3] = np.log(df['noise_maximum_value']  - df['noise_minimum_value'])
        X[:,i+4] = np.log(df['signal_maximum_value'] - df['signal_minimum_value'])
        X[:,i+5] = np.log(df['signal_variance'])
        X[:,i+6] = np.log(df['noise_variance'])

        # Source/recv distance (take log to flatten this)
        X[:,i+7] = df['source_depth_km']

        column_names += ['signal_dominant_frequency', 
                            'signal_dominant_amplitude',
                            'noise_max_amplitude',
                            'signal_max_amplitude',
                            'signal_variance',
                            'noise_variance',
                            'source_depth_km' ]

        if source_dist_type == 'coord':
            X[:,i+8] = df['source_latitude']
            X[:,i+9] = df['source_longitude']
            column_names += ['source_latitude', 'source_longitude']
        elif source_dist_type == 'dist':
            X[:,i+8] = np.log(df['source_receiver_distance_km'])
            column_names.append('source_receiver_distance_logkm')
            if linear_model:
                X[:,i+9] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
                column_names.append('source_receiver_back_azimuth_sine')
            else:
                X[:,i+9] = df['source_receiver_back_azimuth']
                column_names.append('source_receiver_back_azimuth_deg')
        elif source_dist_type == 'all':
            X[:,i+8] = df['source_latitude']
            X[:,i+9] = df['source_longitude']
            X[:,i+10] = np.log(df['source_receiver_distance_km'])
            column_names += ['source_latitude', 
                                'source_longitude',
                                'source_receiver_distance_logkm']
            if linear_model:
                X[:,i+11] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
                column_names.append('source_receiver_back_azimuth_sine')
            else:
                X[:,i+11] = df['source_receiver_back_azimuth']
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

    # def compute_feature_matrix_old(df, scaler = True):
    #     # Loosely speaking empirical magnitudes look like:
    #     # M = log10(A) + Q(Delta) 
    #     # where A is the amplitude and Q a distance dependent correction term.
    #     # Additionally, the difference between log10 and log amounts to a scalar
    #     # that a machine can learn.
    #     # Basically, I'm interested in features that:
    #     #   (1) Measure size in, potentially, different amplitudes.
    #     # different `passbands' deviates from the noise, 
    #     n_rows = len(df)
    #     n_columns = 48
    #     X = np.zeros([n_rows, n_columns])
    #     # Get a proxy on size
        
    #     # These are effectively amplitude ratios.  Note,
    #     # log(a/b) = log(a) - log(b)
    #     X[:,0]  = np.log(df['avg_signal_1.00']) - np.log(df['avg_noise_1.00'])
    #     X[:,1]  = np.log(df['avg_signal_2.00']) - np.log(df['avg_noise_2.00'])
    #     X[:,2]  = np.log(df['avg_signal_3.00']) - np.log(df['avg_noise_3.00'])
    #     X[:,3]  = np.log(df['avg_signal_4.00']) - np.log(df['avg_noise_4.00'])
    #     X[:,4]  = np.log(df['avg_signal_5.00']) - np.log(df['avg_noise_5.00'])
    #     X[:,5]  = np.log(df['avg_signal_6.00']) - np.log(df['avg_noise_6.00'])
    #     X[:,6]  = np.log(df['avg_signal_7.00']) - np.log(df['avg_noise_7.00'])
    #     X[:,7]  = np.log(df['avg_signal_8.00']) - np.log(df['avg_noise_8.00'])
    #     X[:,8]  = np.log(df['avg_signal_9.00']) - np.log(df['avg_noise_9.00'])
    #     X[:,9]  = np.log(df['avg_signal_10.00']) - np.log(df['avg_noise_10.00'])
    #     X[:,10] = np.log(df['avg_signal_11.00']) - np.log(df['avg_noise_11.00'])
    #     X[:,11] = np.log(df['avg_signal_12.00']) - np.log(df['avg_noise_12.00'])
    #     X[:,12] = np.log(df['avg_signal_13.00']) - np.log(df['avg_noise_13.00'])
    #     X[:,13] = np.log(df['avg_signal_14.00']) - np.log(df['avg_noise_14.00'])
    #     X[:,14] = np.log(df['avg_signal_15.00']) - np.log(df['avg_noise_15.00'])
    #     X[:,15] = np.log(df['avg_signal_16.00']) - np.log(df['avg_noise_16.00'])
    #     X[:,16] = np.log(df['avg_signal_17.00']) - np.log(df['avg_noise_17.00'])
    #     X[:,17] = np.log(df['avg_signal_18.00']) - np.log(df['avg_noise_18.00'])
    #     # Look at amplitudes
    #     X[:,18] = np.log(df['avg_signal_1.00']) 
    #     X[:,19] = np.log(df['avg_signal_2.00']) 
    #     X[:,20] = np.log(df['avg_signal_3.00']) 
    #     X[:,21] = np.log(df['avg_signal_4.00']) 
    #     X[:,22] = np.log(df['avg_signal_5.00'])
    #     X[:,23] = np.log(df['avg_signal_6.00'])
    #     X[:,24] = np.log(df['avg_signal_7.00'])
    #     X[:,25] = np.log(df['avg_signal_8.00'])
    #     X[:,26] = np.log(df['avg_signal_9.00'])
    #     X[:,27] = np.log(df['avg_signal_10.00'])
    #     X[:,28] = np.log(df['avg_signal_11.00'])
    #     X[:,29] = np.log(df['avg_signal_12.00'])
    #     X[:,30] = np.log(df['avg_signal_13.00'])
    #     X[:,31] = np.log(df['avg_signal_14.00'])
    #     X[:,32] = np.log(df['avg_signal_15.00'])
    #     X[:,33] = np.log(df['avg_signal_16.00'])
    #     X[:,34] = np.log(df['avg_signal_17.00'])
    #     X[:,35] = np.log(df['avg_signal_18.00'])
    #     # Frequency and max amplitude
    #     X[:,36] = np.log(df['signal_dominant_frequency'])
    #     X[:,37] = np.log(df['signal_dominant_amplitude'])
    #     # Time-based featuers: Look at max amplitudes of noise/signal
    #     X[:,38] = np.log(df['noise_maximum_value']  - df['noise_minimum_value'])
    #     X[:,39] = np.log(df['signal_maximum_value'] - df['signal_minimum_value'])
    #     X[:,40] = np.log(df['signal_variance'])
    #     X[:,41] = np.log(df['noise_variance'])
    #     # Source/recv distance (take log to flatten this)
    #     X[:,42] = df['source_depth_km']
    #     # Single stations can learn location
    #     X[:,43] = df['source_latitude']
    #     X[:,44] = df['source_longitude']
    #     X[:,45] = np.log(df['source_receiver_distance_km'])
    #     X[:,46] = np.sin(df['source_receiver_back_azimuth']*np.pi/180)
    #     X[:,47] = df['source_receiver_back_azimuth']

    #     # Standardize features
    #     if (scaler):
    #         scaler = StandardScaler()
    #         scaler = scaler.fit(X)
    #         X = scaler.transform(X)
    #         return X, scaler
        
    #     return X