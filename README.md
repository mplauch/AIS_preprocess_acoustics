

Workflow:

Run the scripts in the following order:

1_Generate_mmsi-list_files.py: Extract one record per MMSI from the raw AIS CSV, standardize vessel types, and calculate the median valid vessel length per MMSI. 


2_Compare_mmsi_databases_V3.py: Compare each MMSI against Vessel_Database.csv. Use database values for matching MMSIs and flag missing vessels for lookup.

3_Scrub_by_mmsi_New.py: Look up missing MMSIs on MyShipTracking, standardize usable vessel information, and merge it into the flagged list.


4_Correct_raw_AIS.py: Apply the merged vessel information to the original AIS CSV by MMSI, processing the file in chunks.

5_Preprocess_ais.py: Spatially filter, clean, quality-check, and interpolate vessel tracks into daily Parquet files.





zone_config.yaml
Salish Sea zone configuration, used by the preprocess script to denote the zone bounds, the target timestep between AIS points, the maximum ship speeds and AIS point gaps (gaps in an otherwise continuous time series for a single ship), and maximum hours stitched from the day before or day after if a ship continuously covers multiple days. 
