"""
Preprocess_ais.py

Raw AIS CSV --> cleaned and gridded daily Parquet files for the SPL model.

The raw AIS is already on a regular 5-min grid, so this script only removes
bad points and fills short reporting gaps (now, as of being updated 2026-11-05)

Pipeline:
  Pass 1 (per day, per ship) -> clean/YYYYMMDD.parquet
    - Load raw CSV, standardize columns, keep valid ship types, spatial filter
    - Remove GPS jumps (points implying an impossible speed in and out)
    - Choose speed per point: reported AIS sog where plausible, else position-derived
    - Remove stationary points (moored/anchored ships aren't radiating propulsion noise)
    - Split the track into segments wherever:
        * points were removed by the filter (never bridged)
        * the ship was silent longer than max_fill_minutes
        * a gap would need a straight line longer than max_fill_nm
    - Collapse length/breadth/draught to one value per ship

  Pass 2 (per day) -> interpolated/YYYYMMDD.parquet
    - Load the day's clean data plus edge records from neighbouring days, so
      tracks continue across midnight (same gap rules as Pass 1)
    - Interpolate each segment onto the model timestep grid. Only short
      reporting gaps inside a segment get filled; segments are never joined.

How to run:
    python Preprocess_ais.py --input data/raw/AIS_2025.csv \\
                             --output data/ais/ \\
                             --config config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

VALID_TYPES = [
    'Containership', 'Tanker', 'VehicleCarrier', 'Tug',
    'Passenger', 'Cargo', 'Ferry', 'Fishing', 'Naval', 'Government'
]


# ===========================================================================
# Loading and standardization
# ===========================================================================

def load_and_standardize(csv_path):
    """Load raw AIS CSV, standardize column names, keep valid ship types"""
    df = pd.read_csv(csv_path)

    col_map = {
        'timestamp': 'time', 'x': 'lon', 'y': 'lat',
        'MMSI': 'mmsi', 'Type': 'ship_type',
        'Length': 'length', 'Width': 'breadth', 'Draught': 'draught',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ship type mapping and filtering
    # type_map = {
    #     'Containerships': 'Containership', 'Ferry': 'Ferry',
    #     'Fishing Vessel': 'Fishing', 'Naval Vessel': 'Naval',
    #     'Tanker': 'Tanker', 'Government/Research': 'Government',
    #     'Vehicles Carrier': 'VehicleCarrier', 'Tug': 'Tug',
    #     'Passenger': 'Passenger', 'Passenger Ship': 'Passenger',
    #     'Bulker': 'Cargo', 'Recreational Vessel': 'Recreational',
    #     'Other': 'Other'
    # }

    if 'ship_type' in df.columns:
        df['ship_type'] = df['ship_type'].astype(str).str.strip()
        dropped = df.loc[~df['ship_type'].isin(VALID_TYPES), 'ship_type'].value_counts()
        if len(dropped):
            print("  Dropped ship types:\n" + dropped.to_string())
        df = df[df['ship_type'].isin(VALID_TYPES)]

    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)

    for col in ['mmsi', 'lat', 'lon', 'sog', 'length', 'breadth', 'draught']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def spatial_filter(df, zone):
    """Keep only points within the study area (from the zone config file)"""
    return df[
        df['lat'].between(zone['lat_min'], zone['lat_max']) &
        df['lon'].between(zone['lon_min'], zone['lon_max'])
    ].copy()


# ===========================================================================
# Geometry
# ===========================================================================

def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles"""
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_nm * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def step_distances(lat, lon):
    """Distance (in nautical miles, or nm) from each point to the next; length n-1"""
    return haversine_nm(lat[:-1], lon[:-1], lat[1:], lon[1:])


def step_speeds(lat, lon, time_vals):
    """Speed (kn) from each point to the next; length n-1"""
    dt_hours = np.diff(time_vals).astype('timedelta64[s]').astype(float) / 3600
    with np.errstate(divide='ignore', invalid='ignore'):
        return step_distances(lat, lon) / np.where(dt_hours > 0, dt_hours, np.nan)


def centered_speed(lat, lon, time_vals):
    """Position-derived speed at each point"""
    if len(lat) < 2:
        return np.full(len(lat), np.nan)
    v = step_speeds(lat, lon, time_vals)
    sog = np.empty(len(lat))
    sog[0], sog[-1] = v[0], v[-1]
    sog[1:-1] = (v[:-1] + v[1:]) / 2
    return sog


# ===========================================================================
# Pass 1: clean one ship's track for one day
# ===========================================================================

def clean_ship_track(group, config):
    """
    Remove bad points from one ship's daily track and split it into segments.
    Returns the surviving raw points with a 'segment' column and a 'sog_final'
    column, or None if nothing survives the filters
    """
    sog_min = config.get('sog_min', 0.25)
    sog_max = config.get('sog_max', 45.0)
    max_disagreement = config.get('max_disagreement', 5.0)
    max_jump_kn = config.get('max_jump_kn', 50.0)
    max_fill_minutes = config.get('max_fill_minutes', 30)
    max_fill_nm = config.get('max_fill_nm', 5.0)
    min_points = config.get('min_points', 3)

    g = (group.sort_values('time')
              .drop_duplicates(subset='time')
              .dropna(subset=['lat', 'lon'])
              .reset_index(drop=True))
    if len(g) < min_points:
        return None

    # 1. Remove GPS jumps: points implying an impossible speed both in and out
    #    (or on the way in/out for the first/last point)
    for _ in range(3):
        if len(g) < 3:
            break
        v = step_speeds(g['lat'].values, g['lon'].values, g['time'].values)
        v_in, v_out = np.r_[np.nan, v], np.r_[v, np.nan]
        bad = (((v_in > max_jump_kn) & (v_out > max_jump_kn))
               | (np.isnan(v_in) & (v_out > max_jump_kn))
               | (np.isnan(v_out) & (v_in > max_jump_kn)))
        if not bad.any():
            break
        g = g[~bad].reset_index(drop=True)
    if len(g) < min_points:
        return None

    # 2. Speed: reported AIS sog when it is in range and agrees with the
    #    positions; otherwise uses the position-derived speed
    sog_pos = centered_speed(g['lat'].values, g['lon'].values, g['time'].values)
    sog_ais = g['sog'].to_numpy(dtype=float) if 'sog' in g else np.full(len(g), np.nan)
    trust_ais = ((sog_ais >= 0) & (sog_ais <= sog_max)
                 & ((np.abs(sog_ais - sog_pos) < max_disagreement) | np.isnan(sog_pos)))
    g['sog_final'] = np.clip(np.where(trust_ais, sog_ais, sog_pos), 0, sog_max)

    # 3. Remove stationary points, records where they were
    n_raw = len(g)
    g['raw_idx'] = np.arange(n_raw)
    g = g[g['sog_final'] >= sog_min].reset_index(drop=True)
    if len(g) < min_points:
        return None

    # 4. Segment breaks
    lat, lon = g['lat'].values, g['lon'].values
    removed_between = g['raw_idx'].diff() > 1
    dt_min = g['time'].diff().dt.total_seconds() / 60
    dist = np.r_[np.nan, step_distances(lat, lon)]
    speed = np.r_[np.nan, step_speeds(lat, lon, g['time'].values)]

    g['segment'] = (removed_between
                    | (dt_min > max_fill_minutes)
                    | (dist > max_fill_nm)
                    | (speed > max_jump_kn)).cumsum()

    # 5. Flag whether the track reaches the start/end of the day's raw data with
    #    nothing removed in between; only those ends may be stitched to a
    #    neighbouring day (otherwise stitching could bridge removed points)
    g['open_start'] = g['raw_idx'] == 0
    g['open_end'] = g['raw_idx'] == n_raw - 1

    g = (g.groupby('segment').filter(lambda s: len(s) >= min_points)
          .drop(columns='raw_idx'))
    return g if len(g) else None


def collapse_dimensions(cleaned, raw_group):
    """Set length/breadth/draught to one value per ship (mode of valid raw values)"""
    cleaned = cleaned.copy()
    for col in ['length', 'breadth', 'draught']:
        valid = raw_group[col].dropna() if col in raw_group.columns else pd.Series(dtype=float)
        valid = valid[valid > 0]   # drops 0 and -99 sentinels
        cleaned[col] = valid.mode().iloc[0] if len(valid) else np.nan
    return cleaned


def process_raw_to_clean_daily(df_day, config):
    """Pass 1 for one day: clean every ship. Returns (cleaned df, n rejected)"""
    cleaned = []
    n_rejected = 0

    for mmsi, group in df_day.groupby('mmsi'):
        track = clean_ship_track(group, config)
        if track is None:
            n_rejected += 1
            continue
        track['ship_type'] = group['ship_type'].iloc[0]
        cleaned.append(collapse_dimensions(track, group))

    if not cleaned:
        return pd.DataFrame(), n_rejected
    return pd.concat(cleaned, ignore_index=True), n_rejected


# ===========================================================================
# Pass 2: cross-day stitching and gridding
# ===========================================================================

def load_with_neighbors(clean_dir, date, config):
    """
    Load a day's clean data plus the last record of the
    previous day and the first record of the next day, when they are close
    enough in time/space to fill (same rules as segment breaks in clean func).

    Only ends where the filter removed nothing (open_start / open_end) are
    stitched.
    """
    max_gap = pd.Timedelta(minutes=config.get('max_fill_minutes', 30))
    max_fill_nm = config.get('max_fill_nm', 5.0)

    target_file = clean_dir / f"{date:%Y%m%d}.parquet"
    if not target_file.exists():
        return pd.DataFrame()
    df = pd.read_parquet(target_file)

    df = df.sort_values('time')
    first_today = df.groupby('mmsi').head(1).set_index('mmsi')
    last_today = df.groupby('mmsi').tail(1).set_index('mmsi')

    def edge_records(neighbor_date, take_last):
        """Edge records from a neighbouring day that connect to today"""
        f = clean_dir / f"{neighbor_date:%Y%m%d}.parquet"
        if not f.exists():
            return []
        nb = pd.read_parquet(f).sort_values('time')
        nb = nb[nb['mmsi'].isin(first_today.index)]
        nb_edge = nb.groupby('mmsi').tail(1) if take_last else nb.groupby('mmsi').head(1)

        records = []
        for _, rec in nb_edge.iterrows():
            # previous-day record connects to today's first point, next-day to today's last
            today = first_today.loc[rec['mmsi']] if take_last else last_today.loc[rec['mmsi']]
            gap = (today['time'] - rec['time']) if take_last else (rec['time'] - today['time'])
            dist = haversine_nm(rec['lat'], rec['lon'], today['lat'], today['lon'])
            # both ends must be untouched raw edges, so nothing removed lies between them
            both_open = (rec['open_end'] and today['open_start']) if take_last \
                else (rec['open_start'] and today['open_end'])
            if both_open and pd.Timedelta(0) < gap <= max_gap and dist <= max_fill_nm:
                rec = rec.copy()
                rec['segment'] = today['segment']
                records.append(rec)
        return records

    edges = (edge_records(date - pd.Timedelta(days=1), take_last=True)
             + edge_records(date + pd.Timedelta(days=1), take_last=False))
    if edges:
        df = pd.concat([df, pd.DataFrame(edges)], ignore_index=True)
    return df


def interpolate_segment(seg, time_grid):
    """
    Interpolate one segment onto the model time grid
    """
    seg = seg.sort_values('time').drop_duplicates(subset='time')
    if len(seg) < 2:
        return None

    grid = time_grid[(time_grid >= seg['time'].min()) & (time_grid <= seg['time'].max())]
    if len(grid) == 0:
        return None

    t_obs = (seg['time'] - seg['time'].iloc[0]).dt.total_seconds().values
    t_grid = (grid - seg['time'].iloc[0]).total_seconds().values

    out = pd.DataFrame({
        'time': grid,
        'lat': np.interp(t_grid, t_obs, seg['lat'].values),
        'lon': np.interp(t_grid, t_obs, seg['lon'].values),
        'sog': np.interp(t_grid, t_obs, seg['sog_final'].values),
    })
    for col in ['mmsi', 'ship_type', 'length', 'breadth', 'draught']:
        out[col] = seg[col].iloc[0]
    return out


def process_clean_to_interpolated(df_stitched, date, config):
    """Pass 2 for one day: grid every segment of every ship"""
    step = config.get('timestep_minutes', 5)
    day_start = pd.Timestamp(date)
    time_grid = pd.date_range(day_start, day_start + pd.Timedelta(days=1),
                              freq=f'{step}min', inclusive='left')

    pieces = [interpolate_segment(seg, time_grid)
              for _, seg in df_stitched.groupby(['mmsi', 'segment'])]
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


# ===========================================================================
# Main
# ===========================================================================

def main(csv_path, output_dir, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    clean_dir = output_dir / 'clean'
    interp_dir = output_dir / 'interpolated'
    clean_dir.mkdir(parents=True, exist_ok=True)
    interp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path}...")
    df = load_and_standardize(csv_path)
    print(f"  {len(df)} records of valid ship types")
    df = spatial_filter(df, config['zone'])
    print(f"  {len(df)} records in zone")

    # Pass 1
    print("\nPass 1: cleaning raw tracks...")
    df['date'] = df['time'].dt.date
    dates_processed = []
    for date, df_day in df.groupby('date'):
        cleaned, n_rejected = process_raw_to_clean_daily(df_day.drop(columns='date'), config)
        if len(cleaned) == 0:
            print(f"  {date}: no valid ships ({n_rejected} rejected)")
            continue
        cleaned.to_parquet(clean_dir / f"{date:%Y%m%d}.parquet", index=False)
        dates_processed.append(pd.Timestamp(date))
        print(f"  {date}: {cleaned['mmsi'].nunique()} ships, {len(cleaned)} points "
              f"({n_rejected} rejected)")

    # Pass 2
    print("\nPass 2: stitching and gridding...")
    for date in dates_processed:
        df_stitched = load_with_neighbors(clean_dir, date, config)
        result = process_clean_to_interpolated(df_stitched, date, config)
        if len(result) == 0:
            print(f"  {date.date()}: no ships after gridding")
            continue
        result.to_parquet(interp_dir / f"{date:%Y%m%d}.parquet", index=False)
        print(f"  {date.date()}: {result['mmsi'].nunique()} ships, {len(result)} rows")

    print("\nDone.")
    print(f"  Clean files:        {clean_dir}/")
    print(f"  Interpolated files: {interp_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess raw AIS CSV into cleaned, gridded daily Parquet files.')
    parser.add_argument('--input', required=True, help='Path to raw AIS CSV')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.input, args.output, args.config)
