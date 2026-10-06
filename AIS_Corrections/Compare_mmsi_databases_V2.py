#!/usr/bin/env python3
"""
compare_mmsi_databases.py

Compare a corrected MMSI list file (e.g. 2025) against one or more lists
uncorrected yearly MMSI databases (one row per unique MMSI).
-     (lists made from Generate_mmsi-list_files.py)

For each uncorrected year it reports:
  - MMSIs missing from the corrected database (still in need of correction)
  - MMSIs present in both (overlapping)
  - Within the overlap: ship type match, ship name match, and the combined
    categories (both match / type only / name only / neither)
  - For the missing ships: whether they have any value for length, and whether it falls
    within TYPE_LENGTH_BOUNDS for their type

Outputs per year (in OUTDIR):
  <year>_missing_from_corrected.csv   ships still need correction by hand
  <year>_overlap_comparison.csv       side-by-side overlap
  <year>_type_name_mismatches.csv     overlap ships whose type and/or name differ b/w the two lists
  <year>_missing_length_check.csv     missing ships with a length_status flag

Per-type MMSI lists (CSV with OUT_COLS), split into _1/_2/_Full above LIST_SPLIT ships:
(split is only for large lists to be used in the scrubbing scripts)
  LISTS_DIR_ALL     all missing ships
  LISTS_DIR_FAILED  missing ships that FAIL the length check

Clean lists (CSV with OUT_COLS only) in CLEAN_DIR:
  <year>_overlap_corrected.csv               overlap ships, corrected values
  <year>_type_name_mismatches_corrected.csv  mismatched overlap ships, corrected values
  <year>_missing_passed_length.csv           missing ships that PASS the length check,
                                             uncorrected values
"""
from pathlib import Path
import sys

import pandas as pd

# ================================ CONFIG ================================
WORKDIR = Path("/Users/maximilianlauch/Documents/AIS/AIS_Diagnostic_Checks/mmsi-length_database")

CORRECTED_FILE = "mmsi-length_database_2025.csv"

# filenames for uncorrected or lists to be checked
UNCORRECTED_FILES = {
    "2023": "mmsi-length_2023_uncorrected.csv",
    # "2024": "mmsi-length_2024_uncorrected.csv",
    # "2022": "mmsi-length_2022_uncorrected.csv",
}

# Column names (must be the same in all files)
MMSI_COL = "mmsi"
TYPE_COL = "Type"
NAME_COL = "Name"
LENGTH_COL = "Length"
WIDTH_COL = "Width"

# Columns written to the per-type lists and the clean lists
OUT_COLS = [MMSI_COL, TYPE_COL, NAME_COL, LENGTH_COL, WIDTH_COL]

# Empirically derived length ranges (m) per ship type, used to check the missing ships
TYPE_LENGTH_BOUNDS = {
    "Tanker":         {"min": 37,  "max": 299},
    "Tug":            {"min": 7,   "max": 90},
    "Fishing":        {"min": 8,   "max": 150},
    "Ferry":          {"min": 34,  "max": 168},
    "Government":     {"min": 5,   "max": 160},
    "Passenger":      {"min": 15,  "max": 350},
    "VehicleCarrier": {"min": 160, "max": 300},
    "Naval":          {"min": 8,   "max": 335},
    "Cargo":          {"min": 24,  "max": 337},
    "Containership":  {"min": 180, "max": 370},
}

OUTDIR = WORKDIR / "2023_comparison_output"
WRITE_CSVS = True
PRINT_MISMATCHES = False  # print the type/name mismatch table to the console

# Length statuses that count as FAILING the check (everything else passes).
# Options: "too_short", "too_long", "no_length", "no_bounds_for_type"
LENGTH_FAIL_STATUSES = ("too_short", "too_long", "no_length", "no_bounds_for_type")

# Output directories (relative to OUTDIR); {label} is replaced by the year
LISTS_DIR_ALL = "{label}_uncorrected_mmsi_lists"       # per-type lists, all missing
LISTS_DIR_FAILED = "{label}_failed_length_mmsi_lists"  # per-type lists, missing + failed
CLEAN_DIR = "{label}_clean_lists"                      # corrected overlap + passed missing
LIST_SPLIT = 600  # types with more ships than this get split into _1 / _2 (+ _Full)
# ========================================================================

UNC, COR = "_unc", "_cor"  # suffixes for side-by-side columns (unc for uncorrected, cor for corrected)

LENGTH_STATUSES = ("in_bounds", "too_short", "too_long", "no_bounds_for_type", "no_length")


def norm_name(s):
    """Normalise ship names: uppercase, strip AIS '@' padding, collapse whitespace."""
    out = (s.astype("string")
             .str.upper()
             .str.replace("@", "", regex=False)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())
    return out.replace("", pd.NA)


def norm_type(s):
    """Normalise ship types so 70, 70.0 and ' 70 ' (or 'cargo'/'CARGO') compare equal."""
    out = (s.astype("string")
             .str.strip()
             .str.upper()
             .str.replace(r"\.0+$", "", regex=True))
    return out.replace("", pd.NA)


def load(path):
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")
    df = pd.read_csv(path)

    missing_cols = [c for c in OUT_COLS if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: {path.name} is missing columns {missing_cols}. "
                 f"Available: {list(df.columns)}")

    df[MMSI_COL] = pd.to_numeric(df[MMSI_COL], errors="coerce").astype("Int64")
    n_bad = df[MMSI_COL].isna().sum()
    if n_bad:
        print(f"  WARNING: {path.name}: dropping {n_bad} rows with invalid MMSI")
        df = df.dropna(subset=[MMSI_COL])

    n_dup = df[MMSI_COL].duplicated().sum()
    if n_dup:
        print(f"  WARNING: {path.name}: {n_dup} duplicate MMSIs, keeping first occurrence")
        df = df.drop_duplicates(subset=MMSI_COL, keep="first")

    return df


def pct(n, total):
    return f"{n:>6} ({100 * n / total:5.1f}%)" if total else f"{n:>6}"


def compare(corrected, uncorrected, label):
    in_cor = uncorrected[MMSI_COL].isin(corrected[MMSI_COL])
    missing = uncorrected[~in_cor].copy()

    # Inner merge keeps every column from both files side by side (lengths, breadths, etc.)
    overlap = uncorrected.merge(corrected, on=MMSI_COL, how="inner", suffixes=(UNC, COR))

    t_unc, t_cor = norm_type(overlap[TYPE_COL + UNC]), norm_type(overlap[TYPE_COL + COR])
    n_unc, n_cor = norm_name(overlap[NAME_COL + UNC]), norm_name(overlap[NAME_COL + COR])

    # A blank/NaN on either side counts as a non-match
    overlap["type_match"] = (t_unc == t_cor).fillna(False).astype(bool)
    overlap["name_match"] = (n_unc == n_cor).fillna(False).astype(bool)

    tm, nm = overlap["type_match"], overlap["name_match"]
    overlap["match_category"] = "neither"
    overlap.loc[tm & nm, "match_category"] = "both"
    overlap.loc[tm & ~nm, "match_category"] = "type_only"
    overlap.loc[~tm & nm, "match_category"] = "name_only"

    n_ov = len(overlap)
    cats = overlap["match_category"].value_counts()
    print(f"\n=== {label} (uncorrected) vs {CORRECTED_FILE} ===")
    print(f"Uncorrected unique MMSI : {len(uncorrected):>6}")
    print(f"Corrected unique MMSI   : {len(corrected):>6}")
    print(f"Missing from corrected  : {pct(len(missing), len(uncorrected))}")
    print(f"Overlap                 : {pct(n_ov, len(uncorrected))}")
    print("  Separately (% of overlap):")
    print(f"    Type match          : {pct(int(tm.sum()), n_ov)}")
    print(f"    Name match          : {pct(int(nm.sum()), n_ov)}")
    print("  Together (% of overlap):")
    for cat in ("both", "type_only", "name_only", "neither"):
        print(f"    {cat:<20}: {pct(int(cats.get(cat, 0)), n_ov)}")

    # ---- Ships where type and/or name differ, side by side ----
    mm_cols = [MMSI_COL, "match_category",
               TYPE_COL + UNC, TYPE_COL + COR, NAME_COL + UNC, NAME_COL + COR]
    mismatches = (overlap.loc[overlap["match_category"] != "both", mm_cols]
                  .sort_values(["match_category", MMSI_COL])
                  .rename(columns={
                      TYPE_COL + UNC: f"{TYPE_COL}_{label}",
                      TYPE_COL + COR: f"{TYPE_COL}_corrected",
                      NAME_COL + UNC: f"{NAME_COL}_{label}",
                      NAME_COL + COR: f"{NAME_COL}_corrected",
                  }))

    if PRINT_MISMATCHES and not mismatches.empty:
        print(f"\n  Type/name mismatches ({len(mismatches)}):")
        print(mismatches.to_string(index=False))

    if WRITE_CSVS:
        f_mm = OUTDIR / f"{label}_type_name_mismatches.csv"
        OUTDIR.mkdir(parents=True, exist_ok=True)
        mismatches.to_csv(f_mm, index=False)
        print(f"  -> {f_mm}")

        # Side-by-side overlap without the match flags (those live in the mismatch file)
        flags = ["match_category", "type_match", "name_match"]
        lead = [MMSI_COL, TYPE_COL + UNC, TYPE_COL + COR, NAME_COL + UNC, NAME_COL + COR]
        rest = [c for c in overlap.columns if c not in lead + flags]
        overlap_out = overlap[lead + rest].sort_values(MMSI_COL)

        f_missing = OUTDIR / f"{label}_missing_from_corrected.csv"
        f_overlap = OUTDIR / f"{label}_overlap_comparison.csv"
        missing.to_csv(f_missing, index=False)
        overlap_out.to_csv(f_overlap, index=False)
        print(f"  -> {f_missing}")
        print(f"  -> {f_overlap}")

    return missing, overlap


def check_lengths(missing, label):
    """For the missing ships: is there a length at all, and does it fit the type bounds?

    length_status is one of:
      no_length           blank, non-numeric, or <= 0 (AIS uses 0 for 'not available')
      no_bounds_for_type  has a length, but its type isn't in TYPE_LENGTH_BOUNDS
      too_short / too_long / in_bounds
    """
    df = missing.copy()
    length = pd.to_numeric(df[LENGTH_COL], errors="coerce")
    types = df[TYPE_COL].astype("string").str.strip()

    lo = types.map({t: b["min"] for t, b in TYPE_LENGTH_BOUNDS.items()}).astype(float)
    hi = types.map({t: b["max"] for t, b in TYPE_LENGTH_BOUNDS.items()}).astype(float)

    has_len = length.notna() & (length > 0)

    status = pd.Series("in_bounds", index=df.index, name="length_status")
    status[~has_len] = "no_length"
    status[has_len & lo.isna()] = "no_bounds_for_type"
    status[has_len & (length < lo)] = "too_short"
    status[has_len & (length > hi)] = "too_long"

    df["length_status"] = status
    df["length_min"] = lo
    df["length_max"] = hi

    n = len(df)
    counts = status.value_counts()
    print(f"\n--- Length check on {n} missing ships ({label}) ---")
    print(f"  Has length info       : {pct(int(has_len.sum()), n)}")
    for s in LENGTH_STATUSES:
        print(f"    {s:<20}: {pct(int(counts.get(s, 0)), n)}")

    print("\n  By type:")
    table = pd.crosstab(types.fillna("NaN"), status)
    table = table.reindex(columns=[s for s in LENGTH_STATUSES if s in table.columns])
    print(table.to_string())

    if WRITE_CSVS:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        order = {s: i for i, s in enumerate(("too_short", "too_long", "no_length",
                                             "no_bounds_for_type", "in_bounds"))}
        lead = [MMSI_COL, "length_status", TYPE_COL, LENGTH_COL, "length_min", "length_max"]
        df_out = (df[lead + [c for c in df.columns if c not in lead]]
                  .assign(_o=status.map(order))
                  .sort_values(["_o", TYPE_COL, MMSI_COL])
                  .drop(columns="_o"))
        f_len = OUTDIR / f"{label}_missing_length_check.csv"
        df_out.to_csv(f_len, index=False)
        print(f"  -> {f_len}")

    return df


def write_type_lists(df, save_dir, desc):
    """One CSV (OUT_COLS) per ship type; types above LIST_SPLIT get _1, _2 and _Full files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = df[OUT_COLS].copy()
    out[TYPE_COL] = out[TYPE_COL].fillna("NaN")

    print(f"\n{desc}: {len(out)} ships -> {save_dir}")
    count_ships = 0
    for name, group in out.groupby(TYPE_COL):
        if len(group) > LIST_SPLIT:
            parts = {"_1": group.iloc[:LIST_SPLIT], "_2": group.iloc[LIST_SPLIT:], "_Full": group}
        else:
            parts = {"": group}

        for suffix, part in parts.items():
            save_file = save_dir / f"mmsi_list_{name}{suffix}.csv"
            part.to_csv(save_file, index=False)
            print(f"  Saved {save_file.name}: {len(part)} ships")
        count_ships += len(group)
    print(f"  written ships: {count_ships}")


def corrected_view(overlap_rows):
    """Overlap rows to OUT_COLS, taking the corrected (_cor) value for every column."""
    cols = {MMSI_COL: MMSI_COL, **{c + COR: c for c in OUT_COLS if c != MMSI_COL}}
    return overlap_rows[list(cols)].rename(columns=cols).sort_values(MMSI_COL)


def write_clean_lists(overlap, passed, label):
    """Corrected-only overlap/mismatch lists, plus missing ships that passed the length check."""
    save_dir = OUTDIR / CLEAN_DIR.format(label=label)
    save_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        f"{label}_overlap_corrected.csv": corrected_view(overlap),
        f"{label}_type_name_mismatches_corrected.csv":
            corrected_view(overlap[overlap["match_category"] != "both"]),
        f"{label}_missing_passed_length.csv": passed[OUT_COLS].sort_values(MMSI_COL),
    }

    print(f"\nClean lists -> {save_dir}")
    for fname, df in outputs.items():
        df.to_csv(save_dir / fname, index=False)
        print(f"  Saved {fname}: {len(df)} ships")


def main():
    print(f"Loading corrected: {CORRECTED_FILE}")
    corrected = load(WORKDIR / CORRECTED_FILE)

    for label, fname in UNCORRECTED_FILES.items():
        print(f"\nLoading uncorrected [{label}]: {fname}")
        uncorrected = load(WORKDIR / fname)
        missing, overlap = compare(corrected, uncorrected, label)

        length_check = check_lengths(missing, label)
        failed_mask = length_check["length_status"].isin(LENGTH_FAIL_STATUSES)
        failed = length_check[failed_mask]
        passed = length_check[~failed_mask]
        print(f"\n  Length check: {len(passed)} passed, {len(failed)} failed "
              f"(fail = {', '.join(LENGTH_FAIL_STATUSES)})")

        # Per-type lists (same selection/split as before, now CSV with OUT_COLS)
        write_type_lists(missing, OUTDIR / LISTS_DIR_ALL.format(label=label),
                         "All missing ships")
        write_type_lists(failed, OUTDIR / LISTS_DIR_FAILED.format(label=label),
                         "Missing ships that failed the length check")

        # Clean lists in database format
        write_clean_lists(overlap, passed, label)


if __name__ == "__main__":
    main()
