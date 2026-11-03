import pandas as pd

RAW_CSV = "/gpfs/fs7/dfo/meq/work/mal007/DATA/modelling/regional/SalishSea/traffic/AIS/Full_CSVS/January-December_A_VesselAISdata_2015.csv"
MERGED_CSV =  "/gpfs/fs7/dfo/meq/work/doz001/AIS_Preprocessing/2015_flagged_merged.csv"
OUT_CSV = "/gpfs/fs7/dfo/meq/work/doz001/AIS_Preprocessing/Fixed_AIS_2015_Corrected.csv"

CHUNKSIZE = 2_000_000

FIELDS = ["Type", "Name", "Length", "Width", "Draught"]
USE_FLAGS = ["ok", "scraped"]


def load_corrections():
    m = pd.read_csv(MERGED_CSV)
    m = m[m["flag"].isin(USE_FLAGS)].copy()
    m["mmsi"] = pd.to_numeric(m["mmsi"], errors="coerce").round().astype("Int64")
    return m.dropna(subset=["mmsi"]).drop_duplicates("mmsi").set_index("mmsi")[FIELDS]


def main():
    corr = load_corrections()
    print(f"{len(corr)} ships to correct")

    first = True
    n_rows = 0
    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNKSIZE, dtype={"Type": "string", "Name": "string"}):
        key = pd.to_numeric(chunk["mmsi"], errors="coerce").round().astype("Int64")
        new = corr.reindex(key.values)
        new.index = chunk.index
        for f in FIELDS:
            chunk[f] = new[f].where(new[f].notna(), chunk[f])

        chunk.to_csv(OUT_CSV, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(chunk)
        print(f"  {n_rows:,} rows written")

    print(f"Done -> {OUT_CSV}")


if __name__ == "__main__":
    main()