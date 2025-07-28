#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import select
from seis_proc_db.database import Session
from seis_proc_db.tables import ArrWaveformFeats, ArrMagInputFeat, ArrMag, MagMethod
from seis_proc_db.pytables_backend import WaveformStorageReader
from seis_proc_db.services import upsert_mag_method, get_arr_info_for_S_mags

module_path = os.path.abspath(
    os.path.join("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/featmag")
)
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils import load_all_models

sys.path.insert(
    0, "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/waveformArchive/gcc_build"
)
# sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(
    0, "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build"
)
sys.path.insert(
    0, "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/features/np4_build"
)
import pyWaveformArchive as pwa
import pyuussmlmodels as uuss
import pyuussFeatures as pf


def create_features(query_results, stat_df, model_dir, mag_method_id):
    # LOAD THE MODELS
    models_dict = load_all_models(model_dir, stat_df["station"].unique(), "S")

    all_features = []
    all_arr_mags = []
    pytable_reader1 = None
    pytable_reader2 = None
    try:
        for i in range(len(query_results)):
            db_origin = query_results[i][0]
            arr = query_results[i][1]
            db_station = query_results[i][2]
            db_channel1 = query_results[i][3]
            wf_info1 = query_results[i][4]
            db_channel2 = query_results[i][5]
            wf_info2 = query_results[i][6]

            # Use featmag gain values if both channels have them, otherwise use simple_gain_vel
            use_featmag_gain = False
            if (db_channel1.featmag_gain is not None) and (
                db_channel2.featmag_gain is not None
            ):
                use_featmag_gain = True

            #  Read N/1 waveform
            pytable_reader1, pytable_row1, channel1, simple_response1 = (
                gather_channel_info(
                    db_station, db_channel1, wf_info1, pytable_reader1, use_featmag_gain
                )
            )
            if pytable_row1 is None:
                print(f"Failed to load waveform {wf_info1} - skipping")
                continue
            try:
                channel1.simple_response = simple_response1
            except ValueError as err:
                print(err)
                print(f"Failed to set N/1 response for {arr}.   Skipping...")
                continue

            #  Read E/2 waveform
            pytable_reader2, pytable_row2, channel2, simple_response2 = (
                gather_channel_info(
                    db_station, db_channel2, wf_info2, pytable_reader2, use_featmag_gain
                )
            )
            if pytable_row2 is None:
                print(f"Failed to load waveform {wf_info1} - skipping")
                continue
            try:
                channel2.simple_response = simple_response2
            except ValueError as err:
                print(err)
                print(f"Failed to set E/2 response for {arr}.   Skipping...")
                continue

            # Check arrival time and match signals
            start_time = max(wf_info1.start, wf_info2.start)
            start_ind = max(pytable_row1["start_ind"], pytable_row2["start_ind"])
            end_ind = min(pytable_row1["end_ind"], pytable_row2["end_ind"])
            arrival_time_relative_to_start = (arr.arrtime - start_time).total_seconds()
            if arrival_time_relative_to_start < 0:
                print("Arrival is before trace start - skipping")
                continue
            signal1 = pytable_row1["data"][start_ind:end_ind]
            signal2 = pytable_row2["data"][start_ind:end_ind]
            if len(signal1) != len(signal2):
                print(
                    f"Inconsistent signal sizes for {arr}: {len(signal1)}, {len(signal2)},skipping..."
                )
                continue

            # Create the hypocenter
            hypocenter = pf.Magnitude.Hypocenter()
            hypocenter.latitude = db_origin.lat
            hypocenter.longitude = db_origin.lon
            hypocenter.depth = db_origin.depth / 1000  # put in km
            hypocenter.identifier = db_origin.id

            # Create S Feature Extractor
            sf = pf.Magnitude.SFeatures()
            try:
                sf.initialize(channel1, channel2)
                sf.hypocenter = hypocenter
            except (RuntimeError, ValueError) as err:
                print(err)
                print(f"Failed to initialize signal for {arr}.  Skipping...")
                continue

            try:
                sf.process(signal1, signal2, arrival_time_relative_to_start)
            except (RuntimeError, ValueError) as err:
                print(err)
                print(f"Failed to process signal {arr}.  Skipping...")
                continue

            # Get the feature for radial and transverse
            feats_rad = make_radial_feature_dict(sf, arr.id, wf_info1.id, wf_info2.id)
            if feats_rad is None:
                print(f"Abnormally large velocity signal, skipping {arr}...")
                continue
            all_features.append(feats_rad)

            feats_trans = make_transverse_feature_dict(
                sf, arr.id, wf_info1.id, wf_info2.id
            )
            if feats_trans is None:
                print(f"Abnormally large velocity signal, skipping {arr}...")
                continue
            all_features.append(feats_trans)

            has_model = np.any(
                (stat_df["network"] == db_station.net)
                & (stat_df["station"] == db_station.sta)
                # Only comparing the station type because at least 1 station in stat_df
                # has EH[NE] and the db has EH[12] (which is what IRIS has)
                & (stat_df["channel1"].str[:2] == db_channel1.seed_code[:2])
                # & (stat_df["channel2"] == db_channel2.seed_code)
            )

            if has_model:
                source_feat_dict = {
                    "sr_dist_km": arr.sr_dist / 1000,
                    "sr_baz": arr.sr_baz,
                    "depth_km": db_origin.depth / 1000,
                }

                X, proc_col_names = process_S_selected_feats(
                    feats_rad, feats_trans, source_feat_dict
                )

                model = models_dict[db_station.sta]["model"]
                scaler = models_dict[db_station.sta]["scaler"]
                try:
                    X_scaled = scaler.transform(X)
                    mag_val = model.predict(X_scaled)
                except:
                    print(f"An issue in applying the model to {arr}. Skipping...")
                    continue

                arrmag = ArrMag(arid=arr.id, method_id=mag_method_id, mag=mag_val[0])
                for i, col_name in enumerate(proc_col_names):
                    arrmag.input_feats.append(
                        ArrMagInputFeat(name=col_name, val=X_scaled[0][i])
                    )

                all_arr_mags.append(arrmag)

    finally:
        if pytable_reader1 is not None:
            pytable_reader1.close()
        if pytable_reader2 is not None:
            pytable_reader2.close()

    return all_features, all_arr_mags


def make_transverse_feature_dict(sf, arr_id, wf_info1_id, wf_info2_id):
    transverse_velocity_signal = sf.transverse_velocity_signal
    transverse_temporal_noise_features = sf.transverse_temporal_noise_features
    transverse_temporal_signal_features = sf.transverse_temporal_signal_features
    transverse_spectral_noise_features = sf.transverse_spectral_noise_features
    transverse_spectral_signal_features = sf.transverse_spectral_signal_features
    # Likely a problem with the gain Mw8.8 in Maule had PGV of 100
    if max(abs(transverse_velocity_signal)) > 200 * 10000:
        return None

    [frequencies, transverse_noise_amplitudes] = (
        transverse_spectral_noise_features.average_frequencies_and_amplitudes
    )
    [frequencies, transverse_signal_amplitudes] = (
        transverse_spectral_signal_features.average_frequencies_and_amplitudes
    )
    feats_trans = {
        "arid": arr_id,
        "comp": "T",
        "wf_info_id": wf_info1_id,
        "wf_info2_id": wf_info2_id,
        "noise_var": transverse_temporal_noise_features.variance,
        "noise_mine": transverse_temporal_noise_features.minimum_and_maximum_value[0],
        "noise_max": transverse_temporal_noise_features.minimum_and_maximum_value[1],
        "signal_var": transverse_temporal_signal_features.variance,
        "signal_min": transverse_temporal_signal_features.minimum_and_maximum_value[0],
        "signal_max": transverse_temporal_signal_features.minimum_and_maximum_value[1],
        "noise_dom_freq": transverse_spectral_noise_features.dominant_frequency_and_amplitude[
            0
        ],
        "noise_dom_amp": transverse_spectral_noise_features.dominant_frequency_and_amplitude[
            1
        ],
        "signal_dom_freq": transverse_spectral_signal_features.dominant_frequency_and_amplitude[
            0
        ],
        "signal_dom_amp": transverse_spectral_signal_features.dominant_frequency_and_amplitude[
            1
        ],
    }
    for f in range(len(frequencies)):
        feats_trans[f"avg_noise_{frequencies[f]:.0f}hz"] = transverse_noise_amplitudes[
            f
        ]
        feats_trans[f"avg_signal_{frequencies[f]:.0f}hz"] = (
            transverse_signal_amplitudes[f]
        )

    return feats_trans


def make_radial_feature_dict(sf, arr_id, wf_info1_id, wf_info2_id):
    radial_velocity_signal = sf.radial_velocity_signal
    radial_temporal_noise_features = sf.radial_temporal_noise_features
    radial_temporal_signal_features = sf.radial_temporal_signal_features
    radial_spectral_noise_features = sf.radial_spectral_noise_features
    radial_spectral_signal_features = sf.radial_spectral_signal_features

    # Likely a problem with the gain Mw8.8 in Maule had PGV of 100
    if max(abs(radial_velocity_signal)) > 200 * 10000:
        return None

    [frequencies, radial_noise_amplitudes] = (
        radial_spectral_noise_features.average_frequencies_and_amplitudes
    )
    [frequencies, radial_signal_amplitudes] = (
        radial_spectral_signal_features.average_frequencies_and_amplitudes
    )
    feats_rad = {
        "arid": arr_id,
        "comp": "R",
        "wf_info_id": wf_info1_id,
        "wf_info2_id": wf_info2_id,
        "noise_var": radial_temporal_noise_features.variance,
        "noise_mine": radial_temporal_noise_features.minimum_and_maximum_value[0],
        "noise_max": radial_temporal_noise_features.minimum_and_maximum_value[1],
        "signal_var": radial_temporal_signal_features.variance,
        "signal_min": radial_temporal_signal_features.minimum_and_maximum_value[0],
        "signal_max": radial_temporal_signal_features.minimum_and_maximum_value[1],
        "noise_dom_freq": radial_spectral_noise_features.dominant_frequency_and_amplitude[
            0
        ],
        "noise_dom_amp": radial_spectral_noise_features.dominant_frequency_and_amplitude[
            1
        ],
        "signal_dom_freq": radial_spectral_signal_features.dominant_frequency_and_amplitude[
            0
        ],
        "signal_dom_amp": radial_spectral_signal_features.dominant_frequency_and_amplitude[
            1
        ],
    }
    for f in range(len(frequencies)):
        feats_rad[f"avg_noise_{frequencies[f]:.0f}hz"] = radial_noise_amplitudes[f]
        feats_rad[f"avg_signal_{frequencies[f]:.0f}hz"] = radial_signal_amplitudes[f]

    return feats_rad


def gather_channel_info(
    db_station, db_channel, wf_info, pytable_reader, use_featmag_gain
):
    wf_stor_filename = wf_info.hdf_file.name
    #  Read N/1 waveform
    pytable_reader = open_pytable(pytable_reader, wf_stor_filename)
    pytable_row = pytable_reader.select_row(wf_info.id)
    if pytable_row is None:
        print(f"Failed to load waveform {wf_info} - skipping")
        return None, None, None, None

    # Create the simple response for N/1
    simple_response = pf.Magnitude.SimpleResponse()
    if not use_featmag_gain:
        gain = db_channel.simple_gain_vel
        gain_units = "DU/M/S"
    else:
        gain = db_channel.featmag_gain
        gain_units = db_channel.featmag_gain_units
    simple_response.units = gain_units
    simple_response.value = gain

    # Create channel N/1 information
    channel = pf.Magnitude.Channel()
    channel.network_code = db_station.net
    channel.station_code = db_station.sta
    channel.channel_code = db_channel.seed_code
    channel.location_code = db_channel.loc
    channel.latitude = db_station.lat
    channel.longitude = db_station.lon
    channel.sampling_rate = wf_info.source.common_samp_rate
    channel.azimuth = db_channel.azimuth

    return pytable_reader, pytable_row, channel, simple_response


def open_pytable(pytable, wf_stor_filename):
    # Open pytables
    try:
        if pytable is None or pytable.stored_hdf_info != wf_stor_filename:
            if pytable is not None:
                pytable.close()

            pytable = WaveformStorageReader(wf_stor_filename)
        return pytable
    except Exception as e:
        if pytable is not None:
            pytable.close()

        raise e


def process_S_selected_feats(
    radial_feat_dict, transverse_feat_dict, source_feat_dict, w_r=0.5
):
    # amp_1, amp_2, amp_4, noise_var, distance, depth, back_az
    assert w_r <= 1.0 and w_r >= 0.0, "w_r is invalid must be in [0, 1]"
    w_t = 1 - w_r

    feats_dict = {}
    # Compute the amplitudes at 1, 2, and 4 hz
    for freq in [1, 2, 4]:
        amp = w_r * np.log(radial_feat_dict[f"avg_signal_{freq}hz"]) + w_t * np.log(
            transverse_feat_dict[f"avg_signal_{freq}hz"]
        )
        feats_dict[f"amp_{freq}hz"] = amp

    # Compute the noise variance
    feats_dict["noise_var"] = w_r * np.log(
        radial_feat_dict["noise_var"]
    ) + w_t * np.log(transverse_feat_dict["noise_var"])

    # Take the log of the sr distance
    feats_dict["sr_dist_logkm"] = np.log(source_feat_dict["sr_dist_km"])

    # These don't need any transforms
    feats_dict["depth_km"] = source_feat_dict["depth_km"]
    # This would take the sine of baz in radians if a linear model
    feats_dict["sr_baz_deg"] = source_feat_dict["sr_baz"]

    X = np.array(
        [
            [
                feats_dict["amp_1hz"],
                feats_dict["amp_2hz"],
                feats_dict["amp_4hz"],
                feats_dict["noise_var"],
                feats_dict["depth_km"],
                feats_dict["sr_dist_logkm"],
                feats_dict["sr_baz_deg"],
            ]
        ]
    )

    col_names = [
        "amp_1hz",
        "amp_2hz",
        "amp_4hz",
        "noise_var",
        "depth_km",
        "sr_dist_logkm",
        "sr_baz_deg",
    ]

    return X, col_names


if __name__ == "__main__":

    # QUERY SETTINGS
    min_date = datetime(2020, 1, 2, tzinfo=timezone.utc)
    max_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    min_lat = 44
    max_lat = 45.167
    min_lon = -111.333
    max_lon = -109.75
    # Hardcoding these values for now because each table only has one entry
    wf_source_id = 1
    loc_method_id = 1
    assoc_method_id = 6

    # MAG METHOD DETAILS
    phase = "S"
    name = "S-SVR-Armstrong2025"
    details = (
        "SVR models presented in Armstrong et al., 2025. Each model uses features: "
        "amp_1hz, amp_2hz, amp_4hz, noise_var, distance, depth, & back_az. Information "
        "from the radial and transverse components are averaged."
    )

    # LOAD THE TRAINING DATASET SO KNOW THE STATIONS TO USE
    magdir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes"
    datadir = os.path.join(magdir, "feature_splits")
    s_all_train_df = pd.read_csv(f"{datadir}/s.train.csv")
    stations_to_skip = []

    stat_df = s_all_train_df[
        ["network", "station", "channel1", "channel2"]
    ].drop_duplicates()
    stat_df = stat_df[~stat_df["station"].isin(stations_to_skip)]
    assert not np.any(np.isin(stat_df["station"], stations_to_skip))

    # Set the model dir
    model_dir = os.path.join(magdir, "s_models/selected_features_constHP")

    # DO DATABASE QUERY & SAVE THE MAG METHOD
    mag_method_id = None
    with Session() as session:
        with session.begin():
            print("Beginning Query...")
            query_results = get_arr_info_for_S_mags(
                session,
                wf_source_id=wf_source_id,
                loc_method_id=loc_method_id,
                assoc_method_id=assoc_method_id,
                min_date=min_date,
                max_date=max_date,
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
            )
            print(f"Gathered {len(query_results)} rows.")

            upsert_mag_method(
                session, name=name, phase=phase, details=details, path=model_dir
            )
            session.flush()
            mag_method = session.scalars(
                select(MagMethod).where(MagMethod.name == name)
            ).one()

            if mag_method is None:
                raise ValueError("mag_method cannot be None.")

            print(f"Creating features...")
            all_features, all_arr_mags = create_features(
                query_results, stat_df, model_dir, mag_method.id
            )

            print("Inserting ArrWaveformFeats")
            # Store the features in the database
            session.bulk_insert_mappings(ArrWaveformFeats, all_features)
            # Store the magnitudes in the database
            print("Inserting ArrMags")
            session.add_all(all_arr_mags)
