#!/usr/bin/env python3
import sys 
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/waveformArchive/gcc_build')
#sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/features/np4_build')
import h5py
import pyWaveformArchive as pwa 
import pyuussmlmodels as uuss
import pyuussFeatures as pf
#import libpyrtseis as rtseis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

import glob

def create_features(archive_manager,
                    arrival_catalog_df,
                    magnitude_type = 'l',
                    output_file = 'p_1c_features.csv'):
    evids = arrival_catalog_df['evid'].values
    arids = arrival_catalog_df['arrival_id'].values
    networks = arrival_catalog_df['network'].values
    stations = arrival_catalog_df['station'].values
    channels = arrival_catalog_df['channelz'].values
    locations = arrival_catalog_df['location'].values
    gains = arrival_catalog_df['gain_z'].values
    gain_units = arrival_catalog_df['gain_units'].values
    arrival_times = arrival_catalog_df['arrival_time'].values
    event_latitudes = arrival_catalog_df['event_lat'].values
    event_longitudes = arrival_catalog_df['event_lon'].values
    event_depths = arrival_catalog_df['event_depth'].values
    magnitudes = arrival_catalog_df['magnitude'].values
    magnitude_types = arrival_catalog_df['magnitude_type'].values
    receiver_latitudes  = arrival_catalog_df['receiver_lat'].values
    receiver_longitudes = arrival_catalog_df['receiver_lon'].values
    channel_azimuths = arrival_catalog_df['channel_azimuth_z'].values

    n_events = 0
    features = []
    for i in range(len(evids)):
        if (magnitude_types[i] != magnitude_type):
            print("Wrong magnitude type - skipping")
            continue
        exists = archive_manager.waveform_exists(evids[i],
                                                 networks[i], stations[i],
                                                 channels[i], locations[i])
        if (not exists):
            print("No waveform for:", evids[i], networks[i], stations[i], channels[i], locations[i], '  Skipping...')
            continue
        try:
            waveform = archive_manager.read_waveform(evids[i],
                                                     networks[i], stations[i],
                                                     channels[i], locations[i])
        except:
            print("Failed to load waveform - skipping")
            continue
        arrival_time_relative_to_start = arrival_times[i] - waveform.start_time
        if (arrival_time_relative_to_start < 0):
            print("Arrival is before trace start - skipping")
            continue
        waveform.remove_trailing_zeros() # Clean up any junk at end of waveform
        signal = waveform.signal
        #signal_time = len(waveform.signal - 1)/waveform.sampling_rate

        simple_response = pf.Magnitude.SimpleResponse()
        simple_response.units = gain_units[i]
        simple_response.value = gains[i]

        channel = pf.Magnitude.Channel()
        channel.network_code = networks[i]
        channel.station_code = stations[i] 
        channel.channel_code = channels[i]
        channel.location_code = locations[i]
        channel.latitude = receiver_latitudes[i]
        channel.longitude = receiver_longitudes[i]
        try:
            channel.simple_response = simple_response
        except ValueError as err:
            print(err)
            print("Failed to set response for %s.%s.%s.%s.   Skipping..."%(networks[i], channels[i], stations[i], locations[i]))
            continue
        channel.sampling_rate = waveform.sampling_rate
        channel.azimuth = channel_azimuths[i]
        hypocenter = pf.Magnitude.Hypocenter()
        hypocenter.latitude   = event_latitudes[i]
        hypocenter.longitude  = event_longitudes[i]
        hypocenter.depth      = event_depths[i] 
        hypocenter.identifier = evids[i]

        vc = pf.Magnitude.PFeatures()
        vc.initialize(channel) 
        vc.hypocenter = hypocenter
        try:
            vc.process(signal, arrival_time_relative_to_start)
        except (RuntimeError, ValueError) as err:
            print(err)
            print("Failed to process signal %s.%s.%s.%s.  Skipping..."%(networks[i], channels[i], stations[i], locations[i]))
            continue
        velocity_signal = vc.velocity_signal
        temporal_noise_features  = vc.temporal_noise_features
        temporal_signal_features = vc.temporal_signal_features
        spectral_noise_features  = vc.spectral_noise_features
        spectral_signal_features = vc.spectral_signal_features
        [frequencies, noise_amplitudes]  = spectral_noise_features.average_frequencies_and_amplitudes
        [frequencies, signal_amplitudes] = spectral_signal_features.average_frequencies_and_amplitudes

        # Likely a problem with the gain Mw8.8 in Maule had PGV of 100
        if (max(abs(velocity_signal)) > 200*10000):
            #if (temporal_signal_features.variance > 100):
            plt.title("Magnitude %f/Distance %f/Max %f"%(magnitudes[i], vc.source_receiver_distance, max(abs(velocity_signal))) )
            plt.plot(velocity_signal)
            plt.show()
            #plt.plot(signal)
            #plt.show()
            continue

        d = {'event_identifier' : evids[i],
             'arrival_identifier' : arids[i],
             'network' : networks[i],
             'station' : stations[i],
             'channel' : channels[i],
             'location_code' : locations[i],
             'source_latitude' : hypocenter.latitude,
             'source_longitude' : hypocenter.longitude,
             'source_receiver_distance_km' : vc.source_receiver_distance,
             'source_receiver_back_azimuth' : vc.back_azimuth,
             'source_depth_km' : hypocenter.depth,
             'noise_variance' : temporal_noise_features.variance,
             'noise_minimum_value' : temporal_noise_features.minimum_and_maximum_value[0],
             'noise_maximum_value' : temporal_noise_features.minimum_and_maximum_value[1],
             'signal_variance' : temporal_signal_features.variance,
             'signal_minimum_value' : temporal_signal_features.minimum_and_maximum_value[0],
             'signal_maximum_value' : temporal_signal_features.minimum_and_maximum_value[1],
             'noise_dominant_frequency' : spectral_noise_features.dominant_frequency_and_amplitude[0],
             'noise_dominant_amplitude' : spectral_noise_features.dominant_frequency_and_amplitude[1],
             'signal_dominant_frequency' : spectral_signal_features.dominant_frequency_and_amplitude[0],
             'signal_dominant_amplitude' : spectral_signal_features.dominant_frequency_and_amplitude[1],
             'magnitude_type' : magnitude_types[i],
             'magnitude' : magnitudes[i]}
        for f in range(len(frequencies)):
            d['avg_noise_%.2f'%frequencies[f]] = noise_amplitudes[f]
            d['avg_signal_%.2f'%frequencies[f]] = signal_amplitudes[f]
        #plt.plot(velocity_signal)
        #plt.show()
        features.append(d)
        n_events = n_events + 1
    # Loop
    df = pd.DataFrame(features) 
    df.to_csv(output_file, index = False)

if __name__ == "__main__":
    #print(dir(rtseis.PostProcessing.Waveform))
    archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/waveformArchive/archives/'
    h5_archive_files = glob.glob(archive_dir + '/archive_????.h5')
    catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group3/alysha/ben_catalogs/20220728'
    arrival_catalog_1c = f'{catalog_dir}/currentEarthquakeArrivalInformation1CWithGains.csv'
    arrival_catalog_3c = f'{catalog_dir}/currentEarthquakeArrivalInformation3CWithGains.csv'
    startdate = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()
    print(f'Using events occuring on or after {startdate}')

    print("Loading arrival catalog...")
    arrival_catalog_1c_df = pd.read_csv(arrival_catalog_1c, dtype = {'location' : object})
    arrival_catalog_3c_df = pd.read_csv(arrival_catalog_3c, dtype = {'location' : object})
    print(arrival_catalog_1c_df.dtypes)
    arrival_catalog_3c_df = arrival_catalog_3c_df[arrival_catalog_1c_df.columns] # Only keep relevant columns
    arrival_catalog_df = pd.concat([arrival_catalog_1c_df, arrival_catalog_3c_df], ignore_index = True)
    # Regress on Ml
    arrival_catalog_df = arrival_catalog_df[ (arrival_catalog_df.phase == 'P') &
                                             (arrival_catalog_df.magnitude_type == 'l') &
                                             (arrival_catalog_df.origin_time >= startdate)]
    # Focus on Yellowstone
    arrival_catalog_df = arrival_catalog_df[ (arrival_catalog_df.event_lat > 44) &
                                             (arrival_catalog_df.event_lat < 45.167) &
                                             (arrival_catalog_df.event_lon > -111.133) &
                                             (arrival_catalog_df.event_lon < -109.75) ]

    print("Opening archive files for reading...")
    archive_manager = pwa.ArchiveManager()
    archive_manager.open_files_for_reading(h5_archive_files)

    create_features(archive_manager, arrival_catalog_df,
                    magnitude_type = 'l',
                    output_file = 'data/features/p_features.2022.csv')
