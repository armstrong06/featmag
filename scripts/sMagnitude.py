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
                    output_file = 's_3c_features.csv'):
    evids = arrival_catalog_df['evid'].values
    arids = arrival_catalog_df['arrival_id'].values
    networks = arrival_catalog_df['network'].values
    stations = arrival_catalog_df['station'].values
    channels_1 = arrival_catalog_df['channel1'].values
    channels_2 = arrival_catalog_df['channel2'].values
    locations = arrival_catalog_df['location'].values
    gains_1 = arrival_catalog_df['gain_1'].values
    gains_2 = arrival_catalog_df['gain_2'].values
    gain_units = arrival_catalog_df['gain_units'].values
    arrival_times = arrival_catalog_df['arrival_time'].values
    event_latitudes = arrival_catalog_df['event_lat'].values
    event_longitudes = arrival_catalog_df['event_lon'].values
    event_depths = arrival_catalog_df['event_depth'].values
    magnitudes = arrival_catalog_df['magnitude'].values
    magnitude_types = arrival_catalog_df['magnitude_type'].values
    receiver_latitudes  = arrival_catalog_df['receiver_lat'].values
    receiver_longitudes = arrival_catalog_df['receiver_lon'].values
    channel_azimuths_1 = arrival_catalog_df['channel_azimuth_1'].values
    channel_azimuths_2 = arrival_catalog_df['channel_azimuth_2'].values

    n_events = 0
    features = []
    for i in range(len(evids)):
        if (magnitude_types[i] != magnitude_type):
            print("Wrong magnitude type - skipping")
            continue
        exists = archive_manager.waveform_exists(evids[i],
                                                 networks[i], stations[i],
                                                 channels_1[i], locations[i])
        if (not exists):
            print("No waveform for:", evids[i], networks[i], stations[i], channels_1[i], locations[i], '  Skipping...')
            continue
        exists = archive_manager.waveform_exists(evids[i],
                                                 networks[i], stations[i],
                                                 channels_2[i], locations[i])
        if (not exists):
            print("No waveform for:", evids[i], networks[i], stations[i], channels_2[i], locations[i], '  Skipping...')
            continue


        try:
            waveform_1 = archive_manager.read_waveform(evids[i],
                                                       networks[i], stations[i],
                                                       channels_1[i], locations[i])
            waveform_1.remove_trailing_zeros() # Clean up any junk at end of waveform
        except:
            print("Failed to load waveform - skipping")
            continue
        try:
            waveform_2 = archive_manager.read_waveform(evids[i],
                                                       networks[i], stations[i],
                                                       channels_2[i], locations[i])
            waveform_2.remove_trailing_zeros() # Clean up any junk at end of waveform
        except:
            print("Failed to load waveform - skipping")
            continue
        # Consistent sampling rates?
        if (abs(waveform_2.sampling_rate - waveform_1.sampling_rate) > 1.e-4):
            print("Inconsistent sampling rates - skipping")
            continue
        # Create hypocenter
        hypocenter = pf.Magnitude.Hypocenter()
        hypocenter.latitude   = event_latitudes[i]
        hypocenter.longitude  = event_longitudes[i]
        hypocenter.depth      = event_depths[i] 
        hypocenter.identifier = evids[i]

        # Create simple responses
        simple_response_1 = pf.Magnitude.SimpleResponse()
        simple_response_1.units = gain_units[i]
        simple_response_1.value = gains_1[i]

        simple_response_2 = pf.Magnitude.SimpleResponse()
        simple_response_2.units = gain_units[i]
        simple_response_2.value = gains_2[i]

        # Create channel information
        n_channel = pf.Magnitude.Channel()
        n_channel.network_code = networks[i]
        n_channel.station_code = stations[i]
        n_channel.channel_code = channels_1[i]
        n_channel.location_code = locations[i]
        n_channel.latitude = receiver_latitudes[i]
        n_channel.longitude = receiver_longitudes[i]
        n_channel.sampling_rate = waveform_1.sampling_rate
        n_channel.azimuth = channel_azimuths_1[i]
        try:
            n_channel.simple_response = simple_response_1
        except ValueError as err:
            print(err)
            print("Failed to set response for %s.%s.%s.%s.   Skipping..."%(networks[i], channels_1[i], stations[i], locations[i]))
            continue

        e_channel = pf.Magnitude.Channel()
        e_channel.network_code = networks[i]
        e_channel.station_code = stations[i]
        e_channel.channel_code = channels_2[i]
        e_channel.location_code = locations[i]
        e_channel.latitude = receiver_latitudes[i]
        e_channel.longitude = receiver_longitudes[i]
        e_channel.sampling_rate = waveform_2.sampling_rate
        e_channel.azimuth = channel_azimuths_2[i]
        try:
            e_channel.simple_response = simple_response_2
        except ValueError as err:
            print(err)
            print("Failed to set response for %s.%s.%s.%s.   Skipping..."%(networks[i], channels_2[i], stations[i], locations[i]))
            continue

        # Need to truncate signals
        n_signal = waveform_1.signal
        e_signal = waveform_2.signal
        dt = 1./waveform_1.sampling_rate
        start_time = max(waveform_1.start_time, waveform_2.start_time)
        end_time   = min(waveform_1.start_time + (len(n_signal) - 1)*dt,
                         waveform_2.start_time + (len(e_signal) - 1)*dt)
        arrival_time_relative_to_start = arrival_times[i] - start_time
        if (arrival_time_relative_to_start < 0):
            print("Arrival is before trace start - skipping")
            continue
        i0_n = int(round((start_time - waveform_1.start_time)/dt))
        i0_e = int(round((start_time - waveform_2.start_time)/dt))
        i1_n = int(round((end_time - waveform_1.start_time)/dt))
        i1_e = int(round((end_time - waveform_2.start_time)/dt))
        n_signal = np.copy(n_signal[i0_n:i1_n])
        e_signal = np.copy(e_signal[i0_e:i1_e])
        if (len(n_signal) != len(e_signal)):
            print("Inconsistent signal sizes:", len(n_signal), len(e_signal), "skipping...")
            continue
      
        #print(hypocenter.latitude, hypocenter.longitude,  hypocenter.depth)
        #print(n_channel.latitude, n_channel.longitude)
        #print(gains_1[i], gains_2[i], channel_azimuths_1[i], channel_azimuths_2[i])
        #print(evids[i], networks[i], channels_1[i], channels_2[i], stations[i], locations[i], waveform_1.sampling_rate)

        #ofl = open('wy_yhb_hhn_hhe_01_60000622.txt', 'w')
        #for i in range(len(e_signal)):
        #    ofl.write('%f,%f,%f\n'%(i*0.01,n_signal[i], e_signal[i]))
        #ofl.close()
        #print(arrival_time_relative_to_start)

        #plt.plot(e_signal)
        #plt.plot(n_signal)
        #plt.show()
        # Create S feature extractor
        sf = pf.Magnitude.SFeatures()
        try:
            sf.initialize(n_channel, e_channel) 
            sf.hypocenter = hypocenter
        except (RuntimeError, ValueError) as err:
            print(err)
            print("Failed to initialize signal %s.%s.%s.%s.%s.  Skipping..."%(networks[i], channels_1[i], channels_2[i], stations[i], locations[i]))
            continue
        except:
            print("runtime error")
        try:
            sf.process(n_signal, e_signal, arrival_time_relative_to_start)#e_signal, arrival_time_relative_to_start)
        except (RuntimeError, ValueError) as err:
            print(err)
            print("Failed to process signal %s.%s.%s.%s.%s.  Skipping..."%(networks[i], channels_1[i], channels_2[i], stations[i], locations[i]))
            continue
        except:
            print("other error")
        radial_velocity_signal = sf.radial_velocity_signal
        transverse_velocity_signal = sf.transverse_velocity_signal
        #if (networks[i] == 'PB'):
        #    print(sf.source_receiver_distance)
        #    plt.plot(radial_velocity_signal, color='black', linewidth=0.8)
        #    plt.plot(transverse_velocity_signal, color='red', linewidth=0.8)
        #    plt.show()
      
        transverse_temporal_noise_features  = sf.transverse_temporal_noise_features
        transverse_temporal_signal_features = sf.transverse_temporal_signal_features
        transverse_spectral_noise_features  = sf.transverse_spectral_noise_features
        transverse_spectral_signal_features = sf.transverse_spectral_signal_features
        radial_temporal_noise_features  = sf.radial_temporal_noise_features
        radial_temporal_signal_features = sf.radial_temporal_signal_features
        radial_spectral_noise_features  = sf.radial_spectral_noise_features
        radial_spectral_signal_features = sf.radial_spectral_signal_features

        [frequencies, transverse_noise_amplitudes]  = transverse_spectral_noise_features.average_frequencies_and_amplitudes
        [frequencies, transverse_signal_amplitudes] = transverse_spectral_signal_features.average_frequencies_and_amplitudes
        [frequencies, radial_noise_amplitudes]  = radial_spectral_noise_features.average_frequencies_and_amplitudes
        [frequencies, radial_signal_amplitudes] = radial_spectral_signal_features.average_frequencies_and_amplitudes

        # Likely a problem with the gain Mw8.8 in Maule had PGV of 100
        if (max(abs(radial_velocity_signal)) > 200*10000):
            #if (temporal_signal_features.variance > 100):
            plt.title("Magnitude %f/Distance %f/Max %f"%(magnitudes[i], sf.source_receiver_distance, max(abs(radial_velocity_signal))) )
            plt.plot(radial_velocity_signal)
            plt.plot(transverse_velocity_signal)
            plt.show()
            #plt.plot(signal)
            #plt.show()
            continue

        d = {'event_identifier' : evids[i],
             'arrival_identifier' : arids[i],
             'network' : networks[i],
             'station' : stations[i],
             'channel1' : channels_1[i],
             'channel2' : channels_2[i],
             'location_code' : locations[i],
             'source_latitude' : hypocenter.latitude,
             'source_longitude' : hypocenter.longitude,
             'source_receiver_distance_km' : sf.source_receiver_distance,
             'source_receiver_back_azimuth' : sf.back_azimuth,
             'source_depth_km' : hypocenter.depth,
             'transverse_noise_variance' : transverse_temporal_noise_features.variance,
             'transverse_noise_minimum_value' : transverse_temporal_noise_features.minimum_and_maximum_value[0],
             'transverse_noise_maximum_value' : transverse_temporal_noise_features.minimum_and_maximum_value[1],
             'radial_noise_variance' : radial_temporal_noise_features.variance,
             'radial_noise_minimum_value' : radial_temporal_noise_features.minimum_and_maximum_value[0],
             'radial_noise_maximum_value' : radial_temporal_noise_features.minimum_and_maximum_value[1],
             'transverse_signal_variance' : transverse_temporal_signal_features.variance,
             'transverse_signal_minimum_value' : transverse_temporal_signal_features.minimum_and_maximum_value[0],
             'transverse_signal_maximum_value' : transverse_temporal_signal_features.minimum_and_maximum_value[1],
             'radial_signal_variance' : radial_temporal_signal_features.variance,
             'radial_signal_minimum_value' : radial_temporal_signal_features.minimum_and_maximum_value[0],
             'radial_signal_maximum_value' : radial_temporal_signal_features.minimum_and_maximum_value[1],
             'transverse_noise_dominant_frequency' : transverse_spectral_noise_features.dominant_frequency_and_amplitude[0],
             'transverse_noise_dominant_amplitude' : transverse_spectral_noise_features.dominant_frequency_and_amplitude[1],
             'transverse_signal_dominant_frequency' : transverse_spectral_signal_features.dominant_frequency_and_amplitude[0],
             'transverse_signal_dominant_amplitude' : transverse_spectral_signal_features.dominant_frequency_and_amplitude[1],
             'radial_noise_dominant_frequency' : radial_spectral_noise_features.dominant_frequency_and_amplitude[0],
             'radial_noise_dominant_amplitude' : radial_spectral_noise_features.dominant_frequency_and_amplitude[1],
             'radial_signal_dominant_frequency' : radial_spectral_signal_features.dominant_frequency_and_amplitude[0],
             'radial_signal_dominant_amplitude' : radial_spectral_signal_features.dominant_frequency_and_amplitude[1],
             'magnitude_type' : magnitude_types[i],
             'magnitude' : magnitudes[i]}
        for f in range(len(frequencies)):
            d['transverse_avg_noise_%.2f'%frequencies[f]]  = transverse_noise_amplitudes[f]
            d['transverse_avg_signal_%.2f'%frequencies[f]] = transverse_signal_amplitudes[f]
            d['radial_avg_noise_%.2f'%frequencies[f]]  = radial_noise_amplitudes[f]
            d['radial_avg_signal_%.2f'%frequencies[f]] = radial_signal_amplitudes[f]
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
    arrival_catalog_3c = f'{catalog_dir}/currentEarthquakeArrivalInformation3CWithGains.csv'
    startdate = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()
    print(f'Using events occuring on or after {startdate}')

    print("Loading arrival catalog...")
    arrival_catalog_df = pd.read_csv(arrival_catalog_3c, dtype = {'location' : object})
    # Regress on Ml for S waves
    arrival_catalog_df = arrival_catalog_df[ (arrival_catalog_df.phase == 'S') &
                                             (arrival_catalog_df.magnitude_type == 'l') &
                                             (arrival_catalog_df.origin_time >= startdate) ]
    # Focus on Yellowstone
    arrival_catalog_df = arrival_catalog_df[ (arrival_catalog_df.event_lat > 44) &
                                             (arrival_catalog_df.event_lat < 45.167) &
                                             (arrival_catalog_df.event_lon > -111.133) &
                                             (arrival_catalog_df.event_lon < -109.75) ]

    print("Opening archive files for reading...")
    archive_manager = pwa.ArchiveManager()
    archive_manager.open_files_for_reading(h5_archive_files)

    create_features(archive_manager,
                    arrival_catalog_df,
                    magnitude_type = 'l',
                    output_file = '../data/features/s_features.2022.csv')
