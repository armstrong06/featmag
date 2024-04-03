import pandas as pd
import numpy as np
import os
from src.features import SplitFeatures as sf

# Do P and S at the same time to ensure they have the same event split
train_frac = 0.8
p_min_train = 300
s_min_train = 150
outdir = '/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits'

# Read files in 
data_path = '../data'
p_feats = pd.read_csv(f'{data_path}/features/p_features.csv')
s_feats = pd.read_csv(f'{data_path}/features/s_features.csv')
p_feats_2022 = pd.read_csv(f'{data_path}/features/p_features.2022.csv')
s_feats_2022 = pd.read_csv(f'{data_path}/features/s_features.2022.csv')

ev_cat = pd.read_csv(f'{data_path}/catalogs/yellowstone.events.ypml-v5.2023.csv')
ev_cat['Date'] = pd.to_datetime(ev_cat['Date'], format='mixed')

sf.print_feature_df_event_counts(p_feats, "P", ev_cat)
sf.print_feature_df_event_counts(s_feats, "S", ev_cat)

# Get the event information for the P and S features
p_ev = sf.get_feature_event_info(p_feats, ev_cat)
s_ev = sf.get_feature_event_info(s_feats, ev_cat, 'S')

# Limit features to those with a YP21 mag, add the YP21 mag to the feature df
p_feats = sf.add_YP21_magnitude_to_features(p_feats, p_ev)
s_feats = sf.add_YP21_magnitude_to_features(s_feats, s_ev, 'S')

# Split into training and testing datasets
train_evids, test_evids = sf.evid_train_test_split(pd.concat([p_ev, s_ev]), time_cutoff=None)
p_train_feats, p_test_feats = sf.get_features_by_evid(p_feats, train_evids, test_evids)
s_train_feats, s_test_feats = sf.get_features_by_evid(s_feats, train_evids, test_evids)
sf.print_split_percentage(p_train_feats['event_identifier'].unique(), p_ev)
sf.print_split_percentage(s_train_feats['event_identifier'].unique(), s_ev, phase='S')

# Count the number of examples per station, select those that meet the minimum requirements
p_counts = sf.get_station_train_test_counts(p_train_feats, p_test_feats)
s_counts = sf.get_station_train_test_counts(s_train_feats, s_test_feats)
p_min_test = sf.compute_min_test_examples(p_min_train, train_frac)
s_min_test = sf.compute_min_test_examples(s_min_train, train_frac, phase='S')
p_good_stats_evid = sf.get_stations_with_min_examples(p_counts, p_min_train, p_min_test)
s_good_stats_evid = sf.get_stations_with_min_examples(s_counts, s_min_train, s_min_test, 'S')
p_almost_stats_evid = sf.get_stations_close_to_criteria(p_counts, p_min_train, p_min_test)
s_almost_stats_evid = sf.get_stations_close_to_criteria(s_counts, s_min_train, s_min_test, 'S')

# Filter the train/test sets to only the stations with enough examples
p_train_feats = sf.filter_feature_stations(p_train_feats, p_good_stats_evid)
p_test_feats = sf.filter_feature_stations(p_test_feats, p_good_stats_evid)
s_train_feats = sf.filter_feature_stations(s_train_feats, s_good_stats_evid, phase='S')
s_test_feats = sf.filter_feature_stations(s_test_feats, s_good_stats_evid, phase='S')

# Cleanup the held out datasets
print("Cleaning up the 2022 held out dataset")
p_feats_2022 = sf.add_YP21_magnitude_to_features(p_feats_2022, ev_cat)
s_feats_2022 = sf.add_YP21_magnitude_to_features(s_feats_2022, ev_cat, 'S')
p_feats_2022 = sf.filter_feature_stations(p_feats_2022, p_good_stats_evid)
s_feats_2022 = sf.filter_feature_stations(s_feats_2022, s_good_stats_evid, 'S')

# Save the training and testing feature sets
p_train_feats.to_csv(os.path.join(outdir, "p.train.csv"), index=False)
p_test_feats.to_csv(os.path.join(outdir, "p.test.csv"), index=False)
p_feats_2022.to_csv(os.path.join(outdir, "p.2022.csv"), index=False)

s_train_feats.to_csv(os.path.join(outdir, "s.train.csv"), index=False)
s_test_feats.to_csv(os.path.join(outdir, "s.test.csv"), index=False)
s_feats_2022.to_csv(os.path.join(outdir, "s.2022.csv"), index=False)

# Save the evid train/test splits
np.savetxt(os.path.join(outdir, 'evids.train.txt'), train_evids, fmt='%i')
np.savetxt(os.path.join(outdir, 'evids.test.txt'), test_evids, fmt='%i')

# Save the station counts
p_counts.to_csv(os.path.join(outdir, 'p.station.ex.counts.csv'), index=False)
s_counts.to_csv(os.path.join(outdir, 's.station.ex.counts.csv'), index=False)