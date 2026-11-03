import pandas as pd

CSV_INPUT = "/gpfs/fs7/dfo/meq/work/mal007/DATA/modelling/regional/SalishSea/traffic/AIS/Full_CSVS/January-December_A_VesselAISdata_2015.csv"

# The list of mmsi in the raw file, as a csv
CSV_OUTPUT = "/gpfs/fs7/dfo/meq/work/doz001/AIS_Preprocessing/mmsi-length_2015_uncorrected.csv"

# Fields in the output csv and fields to extract from raw csv
FIELDS = ['Name', 'Type', 'mmsi', 'Length', 'Width', 'Draught']


def load_and_preprocess_ais(csv_input, fields):
    ais_raw = pd.read_csv(csv_input, header=0, usecols=fields)

    ais_raw.Type = ais_raw.Type.map({'Containerships': 'Containership',
                                     'Containership': 'Containership',
                                     'Naval': 'Naval',
                                     'Cargo': 'Cargo',
                                     'Research': 'Government',
                                     'Government': 'Government',
                                     'VehiclesCarrier': 'VehicleCarrier',
                                     'Ferry': 'Ferry', 'Fishing': 'Fishing',
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


def generate_unique_mmsi_df(ais_in, out_path, print_types=True):

    ais_unique = (
        ais_in
        .drop_duplicates(subset='mmsi')
        .reset_index(drop=True))

    unique_types = ais_unique['Type'].astype(str).unique()

    if print_types:
        for t in sorted(unique_types):
            print(f"  {t}")

    ais_unique.to_csv(out_path,index=False)



def main():
    AIS_RAW = load_and_preprocess_ais(CSV_INPUT, FIELDS)
    generate_unique_mmsi_df(AIS_RAW, CSV_OUTPUT)



if __name__ == "__main__":
    main()
