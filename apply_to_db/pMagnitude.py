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
from seis_proc_db.services import upsert_mag_method, get_arr_info_for_P_mags

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
    models_dict = load_all_models(model_dir, stat_df["station"].unique(), "P")

    # n_events = 0
    all_features = []
    all_arr_mags = []
    pytable_reader = None
    try:
        for i in range(len(query_results)):
            db_origin = query_results[i][0]
            arr = query_results[i][1]
            db_station = query_results[i][2]
            db_channel = query_results[i][3]
            wf_info = query_results[i][4]
            wf_stor_filename = wf_info.hdf_file.name

            # Open pytables
            pytable_reader = open_pytable(pytable_reader, wf_stor_filename)

            # Read waveform
            pytable_row = pytable_reader.select_row(wf_info.id)
            if pytable_row is None:
                print("Failed to load waveform - skipping")
                continue
            signal = pytable_row["data"][
                pytable_row["start_ind"] : pytable_row["end_ind"]
            ]

            arrival_time_relative_to_start = (
                arr.arrtime - wf_info.start
            ).total_seconds()
            if arrival_time_relative_to_start < 0:
                print("Arrival is before trace start - skipping")
                continue

            simple_response = pf.Magnitude.SimpleResponse()
            if db_channel.featmag_gain is None:
                gain = db_channel.simple_gain_vel
                gain_units = "DU/M/S"
            else:
                gain = db_channel.featmag_gain
                gain_units = db_channel.featmag_gain_units

            simple_response.units = gain_units
            simple_response.value = gain

            channel = pf.Magnitude.Channel()
            channel.network_code = db_station.net
            channel.station_code = db_station.sta
            channel.channel_code = db_channel.seed_code
            channel.location_code = db_channel.loc
            channel.latitude = db_station.lat
            channel.longitude = db_station.lon

            try:
                channel.simple_response = simple_response
            except ValueError as err:
                print(err)
                print(f"Failed to set response for {arr}.   Skipping...")
                continue

            channel.sampling_rate = wf_info.source.common_samp_rate
            channel.azimuth = db_channel.azimuth

            hypocenter = pf.Magnitude.Hypocenter()
            hypocenter.latitude = db_origin.lat
            hypocenter.longitude = db_origin.lon
            hypocenter.depth = db_origin.depth / 1000  # put in km
            hypocenter.identifier = db_origin.id

            vc = pf.Magnitude.PFeatures()
            vc.initialize(channel)
            vc.hypocenter = hypocenter

            try:
                vc.process(signal, arrival_time_relative_to_start)
            except (RuntimeError, ValueError) as err:
                print(err)
                print(f"Failed to process signal {arr}.  Skipping...")
                continue

            velocity_signal = vc.velocity_signal
            temporal_noise_features = vc.temporal_noise_features
            temporal_signal_features = vc.temporal_signal_features
            spectral_noise_features = vc.spectral_noise_features
            spectral_signal_features = vc.spectral_signal_features
            [frequencies, noise_amplitudes] = (
                spectral_noise_features.average_frequencies_and_amplitudes
            )
            [frequencies, signal_amplitudes] = (
                spectral_signal_features.average_frequencies_and_amplitudes
            )

            # Likely a problem with the gain Mw8.8 in Maule had PGV of 100
            if max(abs(velocity_signal)) > 200 * 10000:
                print(f"Abnormally large velocity signal, skipping {arr}...")
                continue

            d = {
                "arid": arr.id,
                "comp": "Z",
                "wf_info_id": wf_info.id,
                "noise_var": temporal_noise_features.variance,
                "noise_mine": temporal_noise_features.minimum_and_maximum_value[0],
                "noise_max": temporal_noise_features.minimum_and_maximum_value[1],
                "signal_var": temporal_signal_features.variance,
                "signal_min": temporal_signal_features.minimum_and_maximum_value[0],
                "signal_max": temporal_signal_features.minimum_and_maximum_value[1],
                "noise_dom_freq": spectral_noise_features.dominant_frequency_and_amplitude[
                    0
                ],
                "noise_dom_amp": spectral_noise_features.dominant_frequency_and_amplitude[
                    1
                ],
                "signal_dom_freq": spectral_signal_features.dominant_frequency_and_amplitude[
                    0
                ],
                "signal_dom_amp": spectral_signal_features.dominant_frequency_and_amplitude[
                    1
                ],
            }
            for f in range(len(frequencies)):
                d[f"avg_noise_{frequencies[f]:.0f}hz"] = noise_amplitudes[f]
                d[f"avg_signal_{frequencies[f]:.0f}hz"] = signal_amplitudes[f]

            all_features.append(d)

            has_model = np.any(
                (stat_df["network"] == db_station.net)
                & (stat_df["station"] == db_station.sta)
                & (stat_df["channel"] == db_channel.seed_code)
            )

            if has_model:
                source_feat_dict = {
                    "sr_dist_km": arr.sr_dist / 1000,
                    "sr_baz": arr.sr_baz,
                    "depth_km": db_origin.depth / 1000,
                }
                X, proc_col_names = process_P_selected_feats(d, source_feat_dict)

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

            # n_events = n_events + 1
    finally:
        if pytable_reader is not None:
            pytable_reader.close()

    return all_features, all_arr_mags


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


def process_P_selected_feats(z_feat_dict, source_feat_dict):
    # amp_1, amp_2, sig_var, noise_var, distance, depth, back_az
    feats_dict = {}
    # Compute the amplitudes at 1, 2, and 4 hz
    for freq in [1, 2]:
        amp = np.log(z_feat_dict[f"avg_signal_{freq}hz"])
        feats_dict[f"amp_{freq}hz"] = amp

    # Compute the noise variance
    feats_dict["noise_var"] = np.log(z_feat_dict["noise_var"])

    # Compute the signal variance
    feats_dict["signal_var"] = np.log(z_feat_dict["signal_var"])

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
                feats_dict["signal_var"],
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
        "signal_var",
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
    phase = "P"
    name = "P-SVR-Armstrong2025"
    details = (
        "SVR models presented in Armstrong et al., 2025. Each model uses features: "
        "amp_1, amp_2, sig_var, noise_var, distance, depth, & back_az."
    )

    # LOAD THE TRAINING DATASET SO KNOW THE STATIONS TO USE
    magdir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/magnitudes"
    datadir = os.path.join(magdir, "feature_splits")
    p_all_train_df = pd.read_csv(f"{datadir}/p.train.csv")
    stations_to_skip = ["YDC"]

    stat_df = p_all_train_df[["network", "station", "channel"]].drop_duplicates()
    stat_df = stat_df[~stat_df["station"].isin(stations_to_skip)]
    assert not np.any(np.isin(stat_df["station"], stations_to_skip))

    # Set the model dir
    model_dir = os.path.join(magdir, "p_models/selected_features_constHP")

    # DO DATABASE QUERY & SAVE THE MAG METHOD
    mag_method_id = None
    with Session() as session:
        with session.begin():
            print("Begining Query...")
            query_results = get_arr_info_for_P_mags(
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
