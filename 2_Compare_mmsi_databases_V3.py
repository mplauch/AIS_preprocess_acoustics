"""
Flag each ship in the input list against the database.
in database, Type and Name agree      -> flag "ok"
in database, Type/Name disagree       -> flag "ok", row replaced with the database data
not in the database at all            -> flag "to_scrub", row left as in the input
"""

import pandas as pd

DATABASE_FILE = "C:/Users/ZHANGD/Desktop/AIS_preprocess_acoustics/AIS_Corrections/Vessel_Database.csv"
Input = "C:/Users/ZHANGD/Desktop/AIS_preprocess_acoustics/output/mmsi-length_2015_uncorrected.csv"
Output = "C:/Users/ZHANGD/Desktop/AIS_preprocess_acoustics/output/2015_flagged.csv"


MMSI_COL = "mmsi"
TYPE_COL = "Type"
NAME_COL = "Name"
DB_COLS = [MMSI_COL, TYPE_COL, NAME_COL, "Length", "Width", "Draught"]


KEY = "_key"
RAW, DB = "_raw", "_db"


def norm_name(s):
    out = (s.astype("string").str.upper().str.replace("@", "", regex=False)
             .str.replace(r"\s+", " ", regex=True).str.strip())
    return out.replace("", pd.NA)

def norm_type(s):
    out = (s.astype("string").str.strip().str.upper().str.replace(r"\.0+$", "", regex=True))
    return out.replace("", pd.NA)


def load(path):
    df = pd.read_csv(path)

    df = df[DB_COLS].copy()
    df[MMSI_COL] = pd.to_numeric(df[MMSI_COL], errors="coerce").astype("Int64")
    n_bad = df[MMSI_COL].isna().sum()
    if n_bad:
        print(f"  WARNING: {path}: dropping {n_bad} rows with NaN MMSI")
        df = df.dropna(subset=[MMSI_COL])

    n_dup = df[MMSI_COL].duplicated().sum()
    if n_dup:
        print(f"  WARNING: {path}: {n_dup} duplicate MMSIs, keeping first occurrence")
        df = df.drop_duplicates(subset=MMSI_COL, keep="first")

    # Normalised keys only used for comparison, original values are preserved for output
    df[TYPE_COL + KEY] = norm_type(df[TYPE_COL])
    df[NAME_COL + KEY] = norm_name(df[NAME_COL])
    return df.reset_index(drop=True)


def pct(n, total):
    return f"{n:>6} ({100 * n / total:5.1f}%)" if total else f"{n:>6}"


def flag_ships(raw, db):
    merged = raw.merge(db, on=MMSI_COL, how="left", suffixes=(RAW, DB), indicator=True)
    in_db = merged["_merge"] == "both"

    # NA on either side counts as a non-match
    type_ok = (merged[TYPE_COL + KEY + RAW] == merged[TYPE_COL + KEY + DB]).fillna(False).astype(bool)
    name_ok = (merged[NAME_COL + KEY + RAW] == merged[NAME_COL + KEY + DB]).fillna(False).astype(bool)

    n_valid = int((in_db & type_ok & name_ok).sum())
    n_fixed = int((in_db & ~(type_ok & name_ok)).sum())
    n_scrub = int((~in_db).sum())

    out = pd.DataFrame({MMSI_COL: merged[MMSI_COL]})
    for c in DB_COLS[1:]:
        out[c] = merged[c + RAW].where(~in_db, merged[c + DB])
    out["flag"] = "to_scrub"
    out.loc[in_db, "flag"] = "ok"

    n = len(raw)
    print(f"Total Input: {n:>6}")
    print(f"Valid      : {pct(n_valid, n)}")
    print(f"Fixed      : {pct(n_fixed, n)}")
    print(f"To_scrub   : {pct(n_scrub, n)}")

    return out


def main():
    print(f"Loading database: {DATABASE_FILE}")
    db = load(DATABASE_FILE)

    print(f"\nLoading {Input}")
    raw = load(Input)
    flagged = flag_ships(raw, db)

    flagged.to_csv(Output, index=False)
    print(f"\n  Wrote {len(flagged)} ships -> {Output}")


if __name__ == "__main__":
    main()