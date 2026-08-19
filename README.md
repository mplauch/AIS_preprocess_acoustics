### Preprocess_AIS.py 
script that converts the single raw AIS csv for the year (or any time period) to daily parquet files that the model uses. Includes filtering and interpolation steps to clean up the raw AIS. Definitely not perfect but works for now, more than likely the filtering is super overkill and needs to be toned down (future problem)

### zone_config.yaml
Salish Sea zone configuration, used by the preprocess script to denote the zone bounds, the target timestep between AIS points, the maximum ship speeds and AIS point gaps (gaps in an otherwise continuous time series for a single ship), and maximum hours stitched from the day before or day after if a ship continuously covers multiple days. 

### To Run:
```
python Preprocess_ais.py --input data/raw/AIS_2025.csv \\
                         --output data/ais/ \\
                         --config config.yaml
```
