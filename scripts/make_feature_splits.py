import pandas as pd
import numpy as np
import os
import os
import sys
# make paths above 'notebooks/' visible for local imports.
# +----------------------------------------------------------------------------+
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.features import SplitFeatures as sf
from datetime import datetime, timezone

# Do P and S at the same time to ensure they have the same event split
train_frac = 0.8
p_min_train = 300
s_min_train = 150
evid_split_max_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
station_lats = [44, 45.1]
station_lons = [-111.4, -109.8]
outdir = '/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes/feature_splits'

# Read files in 
data_path = '../data'
all_p_feats = pd.read_csv(f'{data_path}/features/p_features.2024.csv')
all_s_feats = pd.read_csv(f'{data_path}/features/s_features.2024.csv')

ev_cat = pd.read_csv(f'{data_path}/catalogs/yellowstone.events.ypml-v5.2024.csv')
ev_cat['Date'] = pd.to_datetime(ev_cat['Date'], format='mixed')
station_info_df = pd.read_csv(f'{data_path}/stat.info.csv')

sf.print_feature_df_event_counts(all_p_feats, "P", ev_cat)
sf.print_feature_df_event_counts(all_s_feats, "S", ev_cat)

# Get the event information for the P and S features
p_ev = sf.get_feature_event_info(all_p_feats, ev_cat)
s_ev = sf.get_feature_event_info(all_s_feats, ev_cat, 'S')
p_and_s_ev = pd.concat([p_ev, s_ev]).drop_duplicates('Evid')

# Limit features to those with a YP21 mag, add the YP21 mag to the feature df
print('Limiting events to those with a YP21 mag...')
p_feats = sf.add_YP21_magnitude_to_features(all_p_feats, p_ev)
s_feats = sf.add_YP21_magnitude_to_features(all_s_feats, s_ev, 'S')

# Split into training and testing and holdout datasets
train_evids, test_evids = sf.evid_train_test_split(p_and_s_ev.copy(), 
                                                   time_cutoff=evid_split_max_time)
_, temporal_split_evids = sf.date_train_test_split(p_and_s_ev.copy(), evid_split_max_time)
p_train_feats, p_test_feats = sf.get_features_by_evid(p_feats, train_evids, test_evids)
s_train_feats, s_test_feats = sf.get_features_by_evid(s_feats, train_evids, test_evids)
sf.print_split_percentage(p_train_feats['event_identifier'].unique(), 
                          p_ev[p_ev['Date'] < evid_split_max_time])
sf.print_split_percentage(s_train_feats['event_identifier'].unique(), 
                          s_ev[s_ev['Date'] < evid_split_max_time],
                          phase='S')

# Count the number of examples per station, select those that meet the minimum requirements
p_counts = sf.get_station_train_test_counts(p_train_feats, p_test_feats)
s_counts = sf.get_station_train_test_counts(s_train_feats, s_test_feats)
p_min_test = sf.compute_min_test_examples(p_min_train, train_frac)
s_min_test = sf.compute_min_test_examples(s_min_train, train_frac, phase='S')
p_good_stats = sf.get_stations_with_min_examples(p_counts, p_min_train, p_min_test)
s_good_stats = sf.get_stations_with_min_examples(s_counts, s_min_train, s_min_test, 'S')
p_almost_stats_evid = sf.get_stations_close_to_criteria(p_counts, p_min_train, p_min_test)
s_almost_stats_evid = sf.get_stations_close_to_criteria(s_counts, s_min_train, s_min_test, 'S')

# Remove stations outside of region of interest
print("Removing stations outside of region of interest")
p_good_stats = sf.filter_good_stations_by_location(p_good_stats,
                                                   station_info_df,
                                                   station_lats[0],
                                                   station_lats[1],
                                                   station_lons[0],
                                                   station_lons[1])
s_good_stats = sf.filter_good_stations_by_location(s_good_stats,
                                                   station_info_df,
                                                   station_lats[0],
                                                   station_lats[1],
                                                   station_lons[0],
                                                   station_lons[1])

# Filter the train/test sets to only the stations with enough examples in the region of interest
p_train_feats = sf.filter_feature_stations(p_train_feats, p_good_stats)
p_test_feats = sf.filter_feature_stations(p_test_feats, p_good_stats)
s_train_feats = sf.filter_feature_stations(s_train_feats, s_good_stats, phase='S')
s_test_feats = sf.filter_feature_stations(s_test_feats, s_good_stats, phase='S')

# Print train and test stats
sf.print_feature_df_event_counts(p_train_feats, "P train", ev_cat)
sf.print_feature_df_event_counts(s_train_feats, "S train", ev_cat)
sf.print_feature_df_event_counts(p_test_feats, "P test", ev_cat)
sf.print_feature_df_event_counts(s_test_feats, "S test", ev_cat)

# Make the held out datasets
print("Making temporal split...")
p_feats_temporal = p_feats[p_feats['event_identifier'].isin(temporal_split_evids)]
s_feats_temporal = s_feats[s_feats['event_identifier'].isin(temporal_split_evids)]

p_feats_temporal = sf.filter_feature_stations(p_feats_temporal, p_good_stats)
s_feats_temporal = sf.filter_feature_stations(s_feats_temporal, s_good_stats, 'S')

sf.print_feature_df_event_counts(p_feats_temporal, "P temporal", ev_cat)
sf.print_feature_df_event_counts(s_feats_temporal, "S temporal", ev_cat)

# Add holdout info to counts
p_counts = sf.add_station_holdout_counts(p_feats_temporal, p_counts)
s_counts = sf.add_station_holdout_counts(s_feats_temporal, s_counts)

# Save the training and testing feature sets
holdout_startdate_str = evid_split_max_time.strftime("%Y%m%d")
p_train_feats.to_csv(os.path.join(outdir, "p.train.csv"), index=False)
p_test_feats.to_csv(os.path.join(outdir, "p.test.csv"), index=False)
p_feats_temporal.to_csv(os.path.join(outdir, f"p.{holdout_startdate_str}.csv"), index=False)

s_train_feats.to_csv(os.path.join(outdir, "s.train.csv"), index=False)
s_test_feats.to_csv(os.path.join(outdir, "s.test.csv"), index=False)
s_feats_temporal.to_csv(os.path.join(outdir, f"s.{holdout_startdate_str}.csv"), index=False)

# Save the evid train/test splits
np.savetxt(os.path.join(outdir, 'evids.train.txt'), train_evids, fmt='%i')
np.savetxt(os.path.join(outdir, 'evids.test.txt'), test_evids, fmt='%i')
np.savetxt(os.path.join(outdir, f'evids.{holdout_startdate_str}.txt'), temporal_split_evids, fmt='%i')

# Save the station counts
p_counts.to_csv(os.path.join(outdir, 'p.station.ex.counts.csv'), index=False)
s_counts.to_csv(os.path.join(outdir, 's.station.ex.counts.csv'), index=False)