#!/usr/bin/env python3
"""
Simple script that generates mmsi-length list files as csvs

INPUT:
Raw csv file, can be corrected or uncorrected

OUTPUT:
csv files with one row per mmsi from the Raw file
"""

import pandas as pd

# ============================== CONFIG ==============================
# Inputted raw csv (corrected or uncorrected)
CSV_INPUT = "/Users/maximilianlauch/Documents/AIS_Data/Raw/2023_raw/January-December2023_A_5min_VesselAISdata.csv"

# The list of mmsi's (ships) in the raw file, as a csv
CSV_OUTPUT = "/Users/maximilianlauch/Documents/AIS/AIS_Diagnostic_Checks/mmsi-length_database/mmsi-length_2023_uncorrected.csv"

# Fields in the output csv and fields to extract from raw csv
FIELDS = ['Name', 'Type', 'mmsi', 'Length', 'Width', 'Draught']

# Whether you want to print the unique ship types that will be saved to different lists
PRINT_TYPES=True
# =====================================================================


def load_and_preprocess_ais(csv_input, fields):
    """
    - load and correct some of the Type names in the raw AIS file
    - Get's the median of the lengths for each mmsi and then rewrites
        the corresponding lengths as that median

    :param csv_input:
    :param fields:
    :return: preprocessed csv input
    """
    ais_raw = pd.read_csv(csv_input, header=0, usecols=fields)

    ais_raw.Type = ais_raw.Type.map({'Containerships': 'Containership', 'Ferry': 'Ferry', 'Fishing': 'Fishing',
                                     'Fishing Vessel': 'Fishing', 'Naval Vessel': 'Naval', 'Tanker': 'Tanker',
                                     'Government/Research': 'Government', 'Vehicles Carrier': 'VehicleCarrier',
                                     'Tug': 'Tug', 'Passenger': 'Passenger', 'Passenger Ship': 'Passenger',
                                     'Bulker': 'Cargo',
                                     'Recreational Vessel': 'Recreational', 'Other': 'Other', 'T': 'Tug'})

    # Ensure Length is numeric
    ais_raw['Length'] = pd.to_numeric(ais_raw['Length'], errors='coerce')

    # Compute median length per MMSI using only valid values
    median_length_per_mmsi = (
        ais_raw.loc[ais_raw['Length'].notna() & (ais_raw['Length'] > 0)]
        .groupby('mmsi')['Length']
        .median()
    )

    # Map the median back onto every row
    ais_raw['Length'] = ais_raw['mmsi'].map(median_length_per_mmsi)

    return ais_raw


def generate_unique_mmsi_df(ais_in, print_types=True):
    """
    Collapses raw dataset to have just one row per mmsi

    Optionally:
        Print the types that are found in the file


    :param ais_in:
    :param print_types:
    :return: df with one row per mmsi from the raw AIS file
    """
    ais_unique = (
        ais_in
        .drop_duplicates(subset='mmsi')
        .reset_index(drop=True)
    )

    unique_types = ais_unique['Type'].astype(str).unique()

    if print_types:
        for t in sorted(unique_types):
            print(f"  {t}")

    return ais_unique


def save_unique_mmsi_df(ais_in, out_path):
    """
    Save the unique-mmsi dataframe to csv file

    :param ais_in:
    :param out_path:
    """
    # Save unique AIS
    ais_in.to_csv(
        out_path,
        index=False)


def main():
    AIS_RAW = load_and_preprocess_ais(CSV_INPUT, FIELDS)
    AIS_UNIQUE = generate_unique_mmsi_df(AIS_RAW, PRINT_TYPES)

    save_unique_mmsi_df(AIS_UNIQUE, CSV_OUTPUT)


if __name__ == "__main__":
    main()
