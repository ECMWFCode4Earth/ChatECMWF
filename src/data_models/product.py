"""
Pydantic model for the different Chats products available at ECMWF, together with a simple implementation of the similarity search based on sklearn Nearest Neighbours.
This module is needed for running locally a similarity search for figuring out which variable correlates best with the chatGPT-translated user input. This is a different approach with respect to the one taken in the CDS API, as well as for similar parameters, and has been dictated
by the fact that we wanted to test different strategies for the matching problem. This one is based on semantic similarity and similarity search over vector embeddings. 
"""
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


class Products(Enum):
    medium_mslp_wind850 = (
        "Mean sea level pressure and 850 hPa wind speed",
        "medium-mslp-wind850",
    )
    medium_z500_t850 = (
        "500 hPa geopotential height and 850 hPa temperature",
        "medium-z500-t850",
    )
    medium_2mt_wind30 = ("2 m temperature and 30 m wind", "medium-2mt-wind30")
    medium_wind_100m = ("100 m wind and mean sea level pressure", "medium-wind-100m")
    medium_mslp_wind200 = (
        "Mean sea level pressure and 200 hPa wind",
        "medium-mslp-wind200",
    )
    medium_mslp_rain = ("Rain and mean sea level pressure", "medium-mslp-rain")
    medium_clouds = ("Total cloud cover", "medium-clouds")
    medium_rv_div_uv = ("Vorticity and 700 hPa wind", "medium-rv-div-uv")
    medium_simulated_ir = ("Simulated images - infrared", "medium-simulated-ir")
    medium_simulated_wv = ("Simulated images - water vapour", "medium-simulated-wv")
    medium_simulated_wbpt = (
        "Simulated images (infrared) and 850 hPa wet bulb potential temperature",
        "medium-simulated-wbpt",
    )
    medium_simulated_wv_wbpt = (
        "Simulated images (water vapour) and mean sea level pressure",
        "medium-simulated-wv-wbpt",
    )
    medium_t_z = (
        "Temperature and geopotential at various pressure levels",
        "medium-t-z",
    )
    medium_2t_wind = ("2 m temperature and 10 m wind", "medium-2t-wind")
    medium_uv_z = (
        "Wind and geopotential heights at various pressure levels",
        "medium-uv-z",
    )
    medium_uv_rh = (
        "Wind and relative humidity at various pressure levels",
        "medium-uv-rh",
    )
    medium_thickness_mslp = (
        "500-1000 hPa thickness and mean sea level pressure",
        "medium-thickness-mslp",
    )
    medium_indices = ("Indices (MUCAPE/Kindex/Totalx)", "medium-indices")
    medium_cape_cin = ("MUCAPE and MUCIN", "medium-cape-cin")
    medium_precipitation_type = ("Precipitation type", "medium-precipitation-type")
    medium_tcw = ("Total column water", "medium-tcw")
    medium_sund_interval = ("Sunshine duration, last 24 hours", "medium-sund_interval")
    medium_rain_acc = ("Total accumulated precipitation", "medium-rain-acc")
    medium_snowfall = ("Total snowfall during last 6 hours", "medium-snowfall")
    medium_rain_rate = (
        "Precipitation rate (total / large scale / convective / snowfall)",
        "medium-rain-rate",
    )
    medium_frz_rain = ("Freezing rain during last 6 hours", "medium-frz-rain")
    medium_rain_detailed = (
        "Precipitation during last 6 hours (total / large scale / convective)",
        "medium-rain-detailed",
    )
    medium_lightning = (
        "Lightning flash density during last 6 hours",
        "medium-lightning",
    )
    medium_2t_dp = ("2 m dewpoint temperature", "medium-2t-dp")
    medium_bulk_shear = ("Bulk wind shear (computed from levels)", "medium-bulk-shear")
    medium_shear = ("0-6km wind shear and MUCAPE", "medium-shear")
    medium_wind_10wg = (
        "Maximum 10 m gust during last 6 hours and mean sea level pressure",
        "medium-wind-10wg",
    )
    medium_wind_10m = ("10 m wind and mean sea level pressure", "medium-wind-10m")
    medium_cloud_parameters = (
        "Cloud related parameters (cloud base height / ceiling / height of convective cloud top)",
        "medium-cloud-parameters",
    )
    medium_visibility = ("Visibility", "medium-visibility")
    medium_zero_level = (
        "Height of zero degree level (temperature / wet bulb temperature)",
        "medium-zero-level",
    )
    medium_specific_humidity = (
        "Specific humidity at various pressure levels.",
        "medium-specific-humidity",
    )
    medium_swh_mwd = ("Significant wave height and mean direction", "medium-swh-mwd")
    medium_tssh_mwd = (
        "Total swell: significant wave height and mean direction",
        "medium-tssh-mwd",
    )
    medium_wwsh_mwd = (
        "Windsea: significant wave height and mean direction",
        "medium-wwsh-mwd",
    )
    medium_mwp_mwd = ("Mean wave period and mean wave direction", "medium-mwp-mwd")
    medium_mwpts_mwd = (
        "Total swell: mean period and mean direction",
        "medium-mwpts-mwd",
    )
    medium_mwpww_mwd = (
        "Windsea: mean period of waves and direction",
        "medium-mwpww-mwd",
    )
    medium_pv = ("Potential vorticity at various pressure levels", "medium-pv")
    medium_divergence = ("Divergence at various pressure levels", "medium-divergence")
    medium_vorticity = ("Vorticity at various pressure levels", "medium-vorticity")
    medium_wave_period = (
        "Significant wave height of all waves with various periods",
        "medium-wave-period",
    )
    medium_wave_energy_flux = ("Ocean waves energy flux", "medium-wave-energy-flux")
    medium_albedo = ("Albedo", "medium-albedo")
    medium_orography = ("Orography and sea depth", "medium-orography")
    medium_lai = ("Leaf area index", "medium-lai")
    medium_snow_sic = ("Snow depth and sea ice", "medium-snow-sic")
    w_sst = ("Sea surface temperature", "w_sst")
    w_soil_moisture = ("Soil moisture", "w_soil_moisture")
    pincr = ("Analysis increments", "pincr")
    medium_mslp_mean_spread = (
        "Ensemble mean and spread for mean sea level pressure",
        "medium-mslp-mean-spread",
    )
    plot_ensm_essential = (
        "Ensemble mean and spread: four standard parameters",
        "plot_ensm_essential",
    )
    medium_2t_mean_spread = (
        "Ensemble mean and spread: 2 m temperature",
        "medium-2t-mean-spread",
    )
    medium_mslp_spread = (
        "High resolution mean sea level pressure and ensemble spread",
        "medium-mslp-spread",
    )
    medium_10ws_mean_spread = (
        "Ensemble mean and spread: 10 m wind speed",
        "medium-10ws-mean-spread",
    )
    medium_100ws_mean_spread = (
        "Ensemble mean and spread: 100 m wind speed",
        "medium-100ws-mean-spread",
    )
    medium_sst_mean_spread = (
        "Ensemble mean and spread: sea surface temperature",
        "medium-sst-mean-spread",
    )
    medium_t500_mean_spread = (
        "Ensemble mean and spread: 500 hPa geopotential height",
        "medium-t500-mean-spread",
    )
    medium_z300_mean_spread = (
        "Ensemble mean and spread: 300 hPa geopotential height",
        "medium-z300-mean-spread",
    )
    medium_t10_mean_spread = (
        "Ensemble mean for 10 hPa temperature and geopotential",
        "medium-t10-mean-spread",
    )
    medium_multi_efi = ("Multi-parameter EFI during last 24 hours", "medium-multi-efi")
    efi2web_2t = ("EFI 2 m temperature", "efi2web_2t")
    efi2web_2tmin = ("EFI 2 m minimum temperature", "efi2web_2tmin")
    efi2web_2tmax = ("EFI 2 m maximum temperature", "efi2web_2tmax")
    efi2web_10fg = ("EFI wind gust", "efi2web_10fg")
    efi2web_10ff = ("EFI wind speed", "efi2web_10ff")
    efi2web_tp = ("EFI precipitation", "efi2web_tp")
    efi2web_hsttmax = ("EFI significant wave height", "efi2web_hsttmax")
    efi2web_sf = ("EFI snow fall", "efi2web_sf")
    efi2web_cape = ("EFI CAPE", "efi2web_cape")
    efi2web_capeshear = ("EFI CAPE shear", "efi2web_capeshear")
    efi2web_wvf = ("EFI water vapour flux", "efi2web_wvf")
    medium_trajectories = ("Ensemble trajectories", "medium-trajectories")
    pb_tpr = ("Probabilities: multi-day precipitation", "pb_tpr")
    medium_ens_fzrain = (
        "Probabilities: freezing rain during last 6 hours",
        "medium-ens-fzrain",
    )
    medium_point_rain_prob = (
        "Probabilities: point rainfall during last 12 hours",
        "medium-point-rain-prob",
    )
    medium_ens_tp = (
        "Probabilities: precipitation (total / large scale / convective) during last 6 hours",
        "medium-ens-tp",
    )
    medium_ens_snow = ("Probabilities: snowfall during last 6 hours", "medium-ens-snow")
    medium_ens_rain_rate = (
        "Probabilities: precipitation rate (total / large scale / convective / snowfall) during last 6 hours",
        "medium-ens-rain-rate",
    )
    medium_ens_visibility = ("Probabilities: visibility", "medium-ens-visibility")
    medium_ens_mn2t = (
        "Probabilities: minimum 2 m temperature during  last 6 hours",
        "medium-ens-mn2t",
    )
    medium_ens_lightning = (
        "Probabilities: lightning flash density",
        "medium-ens-lightning",
    )
    medium_ens_mx2t = (
        "Probabilities: maximum 2 m temperature, last 6 hours",
        "medium-ens-mx2t",
    )
    medium_ens_2tdew = ("Probabilities: 2 m dew point temperature", "medium-ens-2tdew")
    medium_ens_cape_shear = ("Probabilities: CAPE shear", "medium-ens-cape-shear")
    medium_ens_cape = ("Probabilities: MUCAPE", "medium-ens-cape")
    medium_ens_wind = ("Probabilities: 100 m wind speed", "medium-ens-wind")
    medium_2t_probability = (
        "Probabilities: 2 m temperature below 0ºC",
        "medium-2t-probability",
    )
    medium_ens_ptype = (
        "Probabilities: most probable precipitation type",
        "medium-ens-ptype",
    )
    medium_10m_probability = (
        "Probabilities: 10 m wind speed",
        "medium-10m-probability",
    )
    medium_wg_probability = (
        "Probabilities:  maximum of 10 m wind gust, last 24 hours",
        "medium-wg-probability",
    )
    medium_tp_probability = (
        "Probabilities:  total precipitation, last 24 hours",
        "medium-tp-probability",
    )
    medium_swh_probability = (
        "Probabilities: significant wave height",
        "medium-swh-probability",
    )
    medium_mwp_probability = (
        "Probabilities: mean wave period",
        "medium-mwp-probability",
    )
    medium_2t_long_probability = (
        "Probabilities: 2 m temperature  < 0ºC (day 10-15)",
        "medium-2t-long-probability",
    )
    medium_tp_long_probability = (
        "Probabilities: total precipitation (day 10-15)",
        "medium-tp-long-probability",
    )
    medium_wg_long_probability = (
        "Probabilities: 10 m wind gust (day 10-15)",
        "medium-wg-long-probability",
    )
    medium_swh_long_probability = (
        "Probabilities:  significant wave height (day 10-15)",
        "medium-swh-long-probability",
    )
    medium_mwp_long_probability = (
        "Probabilities:  mean wave period (day 10-15)",
        "medium-mwp-long-probability",
    )
    medium_ens_tp_2t = (
        "Combined probabilities 2 m temperature < 0ºC and total precipitation > 1mm",
        "medium-ens-tp-2t",
    )
    medium_ens_tp_ws = (
        "Combined probabilities wind speed > 10 m s-1 and total precipitation > 1 mm",
        "medium-ens-tp-ws",
    )
    medium_ens_sf_wg = (
        "Combined probabilities wind gust > 10 m s-1 and total snowfall > 1 mm",
        "medium-ens-sf-wg",
    )
    medium_ens_wave_ws = (
        "Combined probabilities wind speed > 10  m s-1   and significant wave height > 2 m",
        "medium-ens-wave-ws",
    )
    medium_2t_anomaly = (
        "2 m temperature deviation / anomaly, last 24 hours",
        "medium-2t-anomaly",
    )
    medium_tp_anomaly = (
        "Total precipitation deviation / anomaly, last 24 hours",
        "medium-tp-anomaly",
    )
    medium_ws_anomaly = (
        "10 m wind speed deviation / anomaly, last 24 hours",
        "medium-ws-anomaly",
    )
    medium_sf_anomaly = (
        "Total snowfall deviation / anomaly, last 24 hours",
        "medium-sf-anomaly",
    )
    opencharts_meteogram = ("ENS Meteograms", "opencharts_meteogram")
    visibility_meteogram_aviation = (
        "ENS visibility meteogram",
        "visibility_meteogram_aviation",
    )
    visibility_meteogram_public = (
        "ENS visibility meteogram (Public ranges)",
        "visibility_meteogram_public",
    )
    opencharts_ptype_meteogram = (
        "Precipitation type meteogram",
        "opencharts_ptype_meteogram",
    )
    visibility_meteogram = ("(New!) ENS visibility meteogram", "visibility_meteogram")
    opencharts_efi_cdf_meteogram = ("EFI/CDF", "opencharts_efi-cdf-meteogram")
    opencharts_vertical_profile_meteogram = (
        "Vertical profiles",
        "opencharts_vertical-profile-meteogram",
    )
    opencharts_extended_meteogram = (
        "ENS extended meteograms",
        "opencharts_extended_meteogram",
    )
    cluster_plot_legA = ("Cluster scenario", "cluster_plot_legA")
    enplot = ("Postage stamp charts", "enplot")
    medium_tc_genesis = (
        "Tropical cyclone activity (Including genesis)",
        "medium-tc-genesis",
    )
    cyclone = ("Tropical cyclone", "cyclone")
    plwww_m_hr_ccafreach_ts_hs = (
        "Lead time of anomaly correlation coefficient  scores (ACC) of 500 hPa height   falls to  80%",
        "plwww_m_hr_ccafreach_ts_hs",
    )
    plwww_m_eps_t850crpssreach_ts = (
        "Lead time of continuous ranked probability skill score (CRPSS) of 850 hPa temperature forecasts falls to 25%",
        "plwww_m_eps_t850crpssreach_ts",
    )
    plwww_m_eps_tpcrpssreach_ts = (
        "Lead time of  continuous ranked probability skill score  (CRPSS) of 24 hour precipitation forecasts falls to 10%",
        "plwww_m_eps_tpcrpssreach_ts",
    )
    plwww_m_hr_seepsreach_ts_hs = (
        "Lead time of stable equitable error in probability space (1-SEEPS) of 24 hour precipitation reaching a threshold",
        "plwww_m_hr_seepsreach_ts_hs",
    )
    plwww_m_efi_roc_hs = (
        "Relative operating characteristics (ROC) skill score of the extreme forecast index (EFI) for three parameters",
        "plwww_m_efi_roc_hs",
    )
    plwww_y_tc = ("Errors of tropical cyclone forecasts", "plwww_y_tc")
    plwww_m_eps_2tcrpsfracv_ts_hs = (
        "Fraction of CRPS value > 5°C for 2 m temperature errors",
        "plwww_m_eps_2tcrpsfracv_ts_hs",
    )
    plwww_y_enfh_2trpssd_ts_hs = (
        "Ranked probability skill score (RPSS) of 2 m temperature (week 3; day 15-21)",
        "plwww_y_enfh_2trpssd_ts_hs",
    )
    plwww_m_hr_ccafreachmulti_ts = (
        "Lead time of anomaly correlation coefficient (ACC) scores reaching multiple thresholds",
        "plwww_m_hr_ccafreachmulti_ts",
    )
    plwww_m_hr_ccaf_adrian_ts = (
        "Lead time of anomaly correlation coefficient (ACC) reaching multiple thresholds ( High resolution (HRES) 500 hPa height forecasts)",
        "plwww_m_hr_ccaf_adrian_ts",
    )
    plwww_m_hr_wp_ts = (
        "Verification of the high-resolution (HRES) forecast of surface parameters",
        "plwww_m_hr_wp_ts",
    )
    plwww_m_hr_ccafreach_ts = (
        "Lead time of ACC reaching a threshold",
        "plwww_m_hr_ccafreach_ts",
    )
    plwww_d_hr_cfpf_ts = (
        "Time series of scores of the HRES forecast, ENS control and ENS members",
        "plwww_d_hr_cfpf_ts",
    )
    plwww_m_hr_seepsreach_ts = (
        "Lead time of 1-SEEPS of 24-h precipitation reaching a threshold",
        "plwww_m_hr_seepsreach_ts",
    )
    plwww_m_eps_wp_brier_ts = (
        "Brier skill score (BSS) of weather parameters",
        "plwww_m_eps_wp_brier_ts",
    )
    plwww_m_efi_roc = ("ROC skill score of EFI", "plwww_m_efi_roc")
    plwww_m_eps_2tcrpsfracv_ts = (
        "Fraction of large 2m temperature errors",
        "plwww_m_eps_2tcrpsfracv_ts",
    )
    plwww_y_enfh_2trpssd_ts = (
        "RPS of 2m temperature of extended range forecast",
        "plwww_y_enfh_2trpssd_ts",
    )
    plwww_m_wmo_mean_europe = (
        "Monthly WMO scores over Europe (against radiosondes)",
        "plwww_m_wmo_mean_europe",
    )
    plwww_m_wmo_mean_extratr = (
        "Monthly WMO scores over the extra-tropics (against analysis)",
        "plwww_m_wmo_mean_extratr",
    )
    plwww_m_wmo_mean_tropics = (
        "Monthly WMO scores over the tropics (against radiosondes)",
        "plwww_m_wmo_mean_tropics",
    )
    plwww_w_other_ts_mslp = (
        "Verification of other centres (mean sea level pressure)",
        "plwww_w_other_ts_mslp",
    )
    plwww_w_other_ts_upperair = (
        "Verification of other centres (upper air)",
        "plwww_w_other_ts_upperair",
    )
    plwww_3m_ens_tigge_spreldiag = (
        "Spread reliability diagram for ENS forecasts by TIGGE centres",
        "plwww_3m_ens_tigge_spreldiag",
    )
    plwww_3m_ens_tigge_upper_mean = (
        "Continuous ranked probability skill scores (CRPSS) of forecasts of upper-air parameters by TIGGE centres.",
        "plwww_3m_ens_tigge_upper_mean",
    )
    plwww_3m_ens_tigge_wp_mean = (
        "Skill scores of forecasts of weather parameters by TIGGE centres",
        "plwww_3m_ens_tigge_wp_mean",
    )
    plwww_m_waves_ts = (
        "Ocean waves and 10 m wind verification against analysis",
        "plwww_m_waves_ts",
    )
    plwww_3m_ens_waves_mean = (
        "Skill scores of ENS forecasts of ocean waves",
        "plwww_3m_ens_waves_mean",
    )
    extended_anomaly_uv = (
        "Winds at various levels: Weekly mean anomalies",
        "extended-anomaly-uv",
    )
    extended_anomaly_2t = (
        "2 m temperature: Weekly mean anomalies",
        "extended-anomaly-2t",
    )
    extended_anomaly_t = (
        "Surface temperature: Weekly mean anomalies",
        "extended-anomaly-t",
    )
    extended_anomaly_tp = (
        "Precipitation: Weekly mean anomalies",
        "extended-anomaly-tp",
    )
    extended_anomaly_mslp = (
        "Mean sea level pressure: Weekly mean anomalies",
        "extended-anomaly-mslp",
    )
    extended_anomaly_multi_param = (
        "Multiparam: Weekly mean anomalies",
        "extended-anomaly-multi-param",
    )
    extended_anomaly_z500 = (
        "500 hPa height: Weekly mean anomalies",
        "extended-anomaly-z500",
    )
    extended_anomaly_10t = (
        "10hPa temperature: Weekly mean anomalies",
        "extended-anomaly-10t",
    )
    extended_pdist_2t = (
        "2 m temperature: Probability distribution",
        "extended-pdist-2t",
    )
    extended_pdist_mslp = (
        "Mean sea level pressure: Probability distribution",
        "extended-pdist-mslp",
    )
    extended_pdist_rain = (
        "Precipitation: Probability distribution",
        "extended-pdist-rain",
    )
    extended_pdist_t = (
        "Surface temperature: Probability distribution",
        "extended-pdist-t",
    )
    extended_prob_anom_2t = (
        "2 m temperature: Probability of weekly anomaly > 0",
        "extended-prob-anom-2t",
    )
    extended_prob_anom_tp = (
        "Precipitation: Probability of weekly anomaly > 0",
        "extended-prob-anom-tp",
    )
    extended_prob_anom_t = (
        "Surface temperature: Probability of weekly anomaly > 0",
        "extended-prob-anom-t",
    )
    extended_prob_anom_mslp = (
        "Mean sea level pressure: Probability of weekly anomaly > 0",
        "extended-prob-anom-mslp",
    )
    extended_efi_sot_2t = (
        "EFI of 2 m temperature - Extended range forecast",
        "extended-efi-sot-2t",
    )
    extended_efi_sot_tp = (
        "EFI precipitation - Extended range forecast",
        "extended-efi-sot-tp",
    )
    mofc_multi_eps_family_plumes = (
        "Monthly forecast plumes - Extended range forecast",
        "mofc_multi_eps_family_plumes",
    )
    mofc_multi_eps_family_stamps = (
        "Postage stamp charts - Extended range forecast",
        "mofc_multi_eps_family_stamps",
    )
    mofc_multi_tcyc_family_forecast = (
        "Tropical storm probabilities - Extended range forecast",
        "mofc_multi_tcyc_family_forecast",
    )
    mofc_multi_tcyc_family_frequency = (
        "Tropical storm frequency - Extended range forecast",
        "mofc_multi_tcyc_family_frequency",
    )
    mofc_multi_eps_family_regime = (
        "Weather regime clusters - Extended range forecast",
        "mofc_multi_eps_family_regime",
    )
    extended_regime_probabilities = (
        "Weather regimes probabilities - Extended range forecast",
        "extended-regime-probabilities",
    )
    mofc_multi_eps_family_mean_flow = (
        "Large scale mean flow - Extended range forecast",
        "mofc_multi_eps_family_mean_flow",
    )
    mofc_multi_eps_family_hovmoller = (
        "Time-longitudes diagram - Extended range forecast",
        "mofc_multi_eps_family_hovmoller",
    )
    mofc_multi_mjo_family_index = (
        "Madden-Julian Oscillation (MJO) index - Extended range forecast",
        "mofc_multi_mjo_family_index",
    )
    mofc_multi_mjo_family_time_longitudes = (
        "Time-longitudes sections - Extended range forecast",
        "mofc_multi_mjo_family_time_longitudes",
    )
    mofc_multi_mjo_family_time_longitudes_stamps = (
        "Time-longitudes sections of individual members - Extended range forecast",
        "mofc_multi_mjo_family_time_longitudes_stamps",
    )
    extended_zonal_mean_zonal_wind = (
        "Mean zonal wind at 10 hPa - Extended range forecast",
        "extended-zonal-mean-zonal-wind",
    )
    extended_2dim_pdf = (
        "Two-dimensional Probability Distribution Functions (PDF) - Extended range forecast",
        "extended-2dim-pdf",
    )
    mofc_multi_verification_anomaly_family_flowz500 = (
        "Northern Hemisphere 500 hPa height anomaly",
        "mofc_multi_verification_anomaly_family_flowz500",
    )
    mofc_multi_verification_anomaly_family_vanomaly = (
        "Global and regional 500 hPa height anomaly",
        "mofc_multi_verification_anomaly_family_vanomaly",
    )
    mofc_multi_verification_scores_family_alldaily = (
        "Daily Scores",
        "mofc_multi_verification_scores_family_alldaily",
    )
    mofc_multi_verification_scores_family_allweekly = (
        "Weekly Scores",
        "mofc_multi_verification_scores_family_allweekly",
    )
    mofc_multi_verification_probability_family_reliability = (
        "Reliability",
        "mofc_multi_verification_probability_family_reliability",
    )
    mofc_multi_verification_probability_family_roc = (
        "Relative Operating Characteristics (ROC)",
        "mofc_multi_verification_probability_family_roc",
    )
    mofc_multi_verification_probability_family_rocmap = (
        "Map of Relative Operating Characteristics (ROC) - ROCMAP",
        "mofc_multi_verification_probability_family_rocmap",
    )
    mofc_multi_verification_probability_family_value = (
        "Value ",
        "mofc_multi_verification_probability_family_value",
    )
    mofc_multi_verification_probability_family_rpssmap = (
        "RPSSMAP - Map of Ranked Probability Skill Score.",
        "mofc_multi_verification_probability_family_rpssmap",
    )
    seasonal_system5_standard_2mtm = (
        "2m Temperature Anomaly – SEAS5",
        "seasonal_system5_standard_2mtm",
    )
    seasonal_system5_anomaly_correlation_2mtm = (
        "2m Temperature - Anomaly Correlation Coefficient – SEAS5",
        "seasonal_system5_anomaly_correlation_2mtm",
    )
    seasonal_system5_roc_skill_score_2mtm = (
        "2m Temperature – ROC Skill Scores – SEAS5",
        "seasonal_system5_roc_skill_score_2mtm",
    )
    seasonal_system5_verification_roc_2mtm = (
        "2m Temperature – ROC Diagram – SEAS5",
        "seasonal_system5_verification_roc_2mtm",
    )
    seasonal_system5_verification_reliability_2mtm = (
        "2m Temperature – Reliability Diagram – SEAS5",
        "seasonal_system5_verification_reliability_2mtm",
    )
    seasonal_system5_standard_z500 = (
        "500 hPa Geopotential Anomaly – SEAS5",
        "seasonal_system5_standard_z500",
    )
    seasonal_system5_anomaly_correlation_z500 = (
        "500 hPa Geopotential - Anomaly Correlation Coefficient – SEAS5",
        "seasonal_system5_anomaly_correlation_z500",
    )
    seasonal_system5_roc_skill_score_z500 = (
        "500 hPa Geopotential – ROC Skill Scores – SEAS5",
        "seasonal_system5_roc_skill_score_z500",
    )
    seasonal_system5_verification_roc_z500 = (
        "500 hPa Geopotential – ROC Diagram – SEAS5",
        "seasonal_system5_verification_roc_z500",
    )
    seasonal_system5_verification_reliability_z500 = (
        "500 hPa Geopotential – Reliability Diagram – SEAS5",
        "seasonal_system5_verification_reliability_z500",
    )
    seasonal_system5_standard_t850 = (
        "850 hPa Temperature – SEAS5",
        "seasonal_system5_standard_t850",
    )
    seasonal_system5_anomaly_correlation_t850 = (
        "850 hPa Temperature - Anomaly Correlation Coefficient – SEAS5",
        "seasonal_system5_anomaly_correlation_t850",
    )
    seasonal_system5_roc_skill_score_t850 = (
        "850 hPa Temperature – ROC Skill Scores – SEAS5",
        "seasonal_system5_roc_skill_score_t850",
    )
    seasonal_system5_verification_roc_t850 = (
        "850 hPa Temperature – ROC Diagram – SEAS5",
        "seasonal_system5_verification_roc_t850",
    )
    seasonal_system5_verification_reliability_t850 = (
        "850 hPa Temperature – Reliability Diagram – SEAS5",
        "seasonal_system5_verification_reliability_t850",
    )
    seasonal_system5_standard_mslp = (
        "Mean Sea Level Pressure – SEAS5",
        "seasonal_system5_standard_mslp",
    )
    seasonal_system5_anomaly_correlation_mslp = (
        "Mean Sea Level Pressure - Anomaly Correlation Coefficient – SEAS5",
        "seasonal_system5_anomaly_correlation_mslp",
    )
    seasonal_system5_roc_skill_score_mslp = (
        "Mean Sea Level Pressure – ROC Skill Scores – SEAS5",
        "seasonal_system5_roc_skill_score_mslp",
    )
    seasonal_system5_verification_roc_mslp = (
        "Mean Sea Level Pressure – ROC Diagram – SEAS5",
        "seasonal_system5_verification_roc_mslp",
    )
    seasonal_system5_verification_reliability_mslp = (
        "Mean Sea Level Pressure – Reliability Diagram – SEAS5",
        "seasonal_system5_verification_reliability_mslp",
    )
    seasonal_system5_standard_rain = (
        "Precipitation – SEAS5",
        "seasonal_system5_standard_rain",
    )
    seasonal_system5_anomaly_correlation_rain = (
        "Precipitation - Anomaly Correlation Coefficient – SEAS5",
        "seasonal_system5_anomaly_correlation_rain",
    )
    seasonal_system5_roc_skill_score_rain = (
        "Precipitation – ROC Skill Scores – SEAS5",
        "seasonal_system5_roc_skill_score_rain",
    )
    seasonal_system5_verification_roc_rain = (
        "Precipitation – ROC Diagram – SEAS5",
        "seasonal_system5_verification_roc_rain",
    )
    seasonal_system5_verification_reliability_rain = (
        "Precipitation – Reliability Diagram – SEAS5",
        "seasonal_system5_verification_reliability_rain",
    )
    seasonal_system5_standard_ssto = (
        "Sea Surface Temperature – SEAS5",
        "seasonal_system5_standard_ssto",
    )
    seasonal_system5_anomaly_correlation_ssto = (
        "Sea Surface Temperature - Anomaly Correlation Coefficient – SEAS5",
        "seasonal_system5_anomaly_correlation_ssto",
    )
    seasonal_system5_roc_skill_score_ssto = (
        "Sea Surface Temperature – ROC Skill Scores – SEAS5",
        "seasonal_system5_roc_skill_score_ssto",
    )
    seasonal_system5_verification_roc_ssto = (
        "Sea Surface Temperature – ROC Diagram – SEAS5",
        "seasonal_system5_verification_roc_ssto",
    )
    seasonal_system5_verification_reliability_ssto = (
        "Sea Surface Temperature – Reliability Diagram – SEAS5",
        "seasonal_system5_verification_reliability_ssto",
    )
    seasonal_system5_climagrams_2mt = (
        "2  m Temperature Area Averages – Long Range Forecast – SEAS5",
        "seasonal_system5_climagrams_2mt",
    )
    seasonal_system5_climagrams_monsoon = (
        "Monsoon Indices – Long Range Forecast – SEAS5",
        "seasonal_system5_climagrams_monsoon",
    )
    seasonal_system5_climagrams_precipitation = (
        "Precipitation Area Averages – Long Range Forecast – SEAS5",
        "seasonal_system5_climagrams_precipitation",
    )
    seasonal_system5_climagrams_sst = (
        "Sea Surface Temperature Area Averages – Long Range Forecast – SEAS5",
        "seasonal_system5_climagrams_sst",
    )
    seasonal_system5_climagrams_teleconnection = (
        "Teleconnection Indices – Long Range Forecast – SEAS5",
        "seasonal_system5_climagrams_teleconnection",
    )
    seasonal_system5_tstorm_ace_verification = (
        "Accumulated Cyclone Energy (Verification) – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_ace_verification",
    )
    seasonal_system5_tstorm_ace = (
        "Accumulated Cyclone Energy – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_ace",
    )
    seasonal_system5_tstorm_hurricane_frequency = (
        "Hurricane/Typhoon Frequency – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_hurricane_frequency",
    )
    seasonal_system5_tstorm_hurricane_verification = (
        "Hurricane/Typhoon Number (Verification) – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_hurricane_verification",
    )
    seasonal_system5_tstorm_density_anomaly = (
        "Tropical Storm Density Anomaly – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_density_anomaly",
    )
    seasonal_system5_tstorm_frequency = (
        "Tropical Storm Frequency – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_frequency",
    )
    seasonal_system5_tstorm_verification = (
        "Tropical Storm Number (Verification) – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_verification",
    )
    seasonal_system5_tstorm_density_standard = (
        "Tropical Storm Standardised Density – Long Range Forecast – SEAS5",
        "seasonal_system5_tstorm_density_standard",
    )
    seasonal_system5_nino_annual_verification = (
        "Nino Annual Plumes (Verification) – Long Range Forecast – SEAS5",
        "seasonal_system5_nino_annual_verification",
    )
    seasonal_system5_nino_verification = (
        "Nino Plumes (Verification) – Long Range Forecast – SEAS5",
        "seasonal_system5_nino_verification",
    )
    seasonal_system5_nino_plumes = (
        "Nino Plumes – Long Range Forecast – SEAS5",
        "seasonal_system5_nino_plumes",
    )
    seasonal_system5_nino_annual_plumes = (
        "Niño Annual Plumes – Long Range Forecast – SEAS5",
        "seasonal_system5_nino_annual_plumes",
    )
    mofc_multi_anomaly = (
        "Weekly anomaly - Extended range forecast",
        "mofc_multi_anomaly",
    )
    mofc_multi_probability_anomaly = (
        "Weekly probability anomaly - Extended range forecast",
        "mofc_multi_probability_anomaly",
    )
    mofc_multi_tercile = (
        "Weekly terciles - Extended range forecast",
        "mofc_multi_tercile",
    )
    mofc_multi_eps_family_multiparameter = (
        "Multiparameter outlook - Extended range forecast",
        "mofc_multi_eps_family_multiparameter",
    )
    fire_activity = ("Fire activity analyses", "fire-activity")
    aerosol_forecasts = ("Aerosol forecasts", "aerosol-forecasts")
    carbon_dioxide_forecasts = ("Carbon Dioxide forecasts", "carbon-dioxide-forecasts")
    carbon_monoxide_forecasts = (
        "Carbon Monoxide forecasts",
        "carbon-monoxide-forecasts",
    )
    formaldehyde_forecasts = ("Formaldehyde forecasts", "formaldehyde-forecasts")
    methane_forecasts = ("Methane forecasts", "methane-forecasts")
    nitrogen_dioxide_forecasts = (
        "Nitrogen Dioxide forecasts",
        "nitrogen-dioxide-forecasts",
    )
    ozone_forecasts = ("Ozone forecasts", "ozone-forecasts")
    particulate_matter_forecasts = (
        "Particulate matter forecasts",
        "particulate-matter-forecasts",
    )
    sulphur_dioxide_forecasts = (
        "Sulphur Dioxide forecasts",
        "sulphur-dioxide-forecasts",
    )
    uvindex_forecasts = ("Uv index forecasts", "uvindex-forecasts")
    tc_overview = ("Tropical cyclone overview map", "tc_overview")
    tc_plumes = ("Tropical cyclone plumes", "tc_plumes")
    tc_strike_probability = (
        "Tropical cyclone strike probability",
        "tc_strike_probability",
    )
    tc_track = ("Tropical cyclone track", "tc_track")
    tc_verification = ("Tropical cyclone verification", "tc_verification")


st_model = SentenceTransformer("all-MiniLM-L6-v2")
all_prods = list()
for prod in Products:
    all_prods.append(st_model.encode(prod.value[0]))

all_prods = np.array(all_prods)

nn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
nn.fit(all_prods)
