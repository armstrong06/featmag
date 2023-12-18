from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

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
    def plot_box_whisker(df, feature, ylabel=None, sort_counts=True, thresholds=None, showfliers=True, min_y_line=None):
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