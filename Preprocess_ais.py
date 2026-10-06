"""
Preprocess_ais.py

Large raw CSV --> cleaned, interpolated daily Parquet files on a regular time grid.

interpolates onto a fine regular grid FIRST, then filters on the regularly-spaced data, then
re-interpolates to fill gaps. This produces stable speed estimates
regardless of the AIS reporting interval (since the raw AIS can be really messy,
overkill if there is cleaner AIS inputted).

Pipeline:
  Pass 1 (per day):
    - Load raw CSV, standardize columns, spatial filter (only take points in zone)
    - Per ship: segment track --> per segment: interpolate to 1-min grid -->
      compute speed --> iteratively filter --> re-interpolate --> recompute
    - Collapse scalar/constant ship dimensions (length/breadth/draught) to single values
    - Save cleaned daily Parquet (1-min resolution by default for 1st pass)

  Pass 2 (per day):
    - Load cleaned data with neighbor-day edge records (cross-day stitching, in case ships travel overnight or have
      consistent, consecutive points that span more than one day)
    - Re-resolve sog on stitched data
    - Interpolate onto the model timestep grid (typically use 5-min but can input any)
    - Note: script never interpolates across observation gaps > max_interp_gap to avoid weird jumps

How to run:
    python preprocess_ais.py --input data/raw/AIS_2025.csv \\
                             --output data/ais/ \\
                             --config config.yaml
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ===========================================================================
# Loading and standardization
# ===========================================================================

def load_and_standardize(csv_path):
    """Load raw AIS CSV and standardize column names"""
    df = pd.read_csv(csv_path)

    col_map = {
        'timestamp': 'time', 'x': 'lon', 'y': 'lat',
        'MMSI': 'mmsi', 'Type': 'ship_type',
        'Length': 'length', 'Width': 'breadth',
        'Draught': 'draught',
        'sog': 'sog', 'cog': 'cog', 'heading': 'heading'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ship type mapping and filtering
    type_map = {
        'Containerships': 'Containership', 'Ferry': 'Ferry',
        'Fishing Vessel': 'Fishing', 'Naval Vessel': 'Naval',
        'Tanker': 'Tanker', 'Government/Research': 'Government',
        'Vehicles Carrier': 'VehicleCarrier', 'Tug': 'Tug',
        'Passenger': 'Passenger', 'Passenger Ship': 'Passenger',
        'Bulker': 'Cargo', 'Recreational Vessel': 'Recreational',
        'Other': 'Other'
    }
    valid_types = [
        'Containership', 'Tanker', 'VehicleCarrier', 'Tug',
        'Passenger', 'Cargo', 'Ferry', 'Fishing', 'Naval', 'Government'
    ]

    if 'ship_type' in df.columns:
        df['ship_type'] = df['ship_type'].map(type_map)
        df = df[df['ship_type'].isin(valid_types)]

    # df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)

    try:
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    except Exception:
        df['time'] = pd.to_datetime(df['time'], utc=True)

    for col in ['mmsi', 'lat', 'lon', 'sog', 'length', 'breadth', 'draught']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def spatial_filter(df, config):
    """Keep only points within the study area zone bounds found in config file"""
    z = config['zone']
    return df[
        (df['lat'] >= z['lat_min']) & (df['lat'] <= z['lat_max']) &
        (df['lon'] >= z['lon_min']) & (df['lon'] <= z['lon_max'])
    ].copy()


# ===========================================================================
# Geometry tools
# ===========================================================================

def haversine_nm(lat1, lon1, lat2, lon2):
    """Haversine distance in nautical miles"""
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_nm * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_segment_speeds(lat, lon, time_vals):
    """
    Compute speed (knots) between consecutive points
    """
    dt_hours = np.diff(time_vals).astype('timedelta64[s]').astype(float) / 3600
    dist_nm = haversine_nm(lat[:-1], lon[:-1], lat[1:], lon[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        speed = dist_nm / np.where(dt_hours > 0, dt_hours, np.nan)
    return speed


def compute_centered_speed(lat, lon, time_vals):
    """
    Compute centered speed
    """
    if len(lat) < 2:
        return np.full(len(lat), np.nan)

    seg_speeds = compute_segment_speeds(lat, lon, time_vals)

    sog = np.empty(len(lat))
    sog[0] = seg_speeds[0]
    sog[-1] = seg_speeds[-1]
    if len(sog) > 2:
        sog[1:-1] = (seg_speeds[:-1] + seg_speeds[1:]) / 2

    return sog


# ===========================================================================
# Track segmentation/separation
# ===========================================================================

def segment_track(group, max_gap_minutes=30, max_jump_kn=40):
    """
    Split a ship's track into continuous segments (to separate between trips like
    if the ship was sitting at port)

    A new segment starts when:
      - The time gap b/w points exceeds max_gap_minutes
      - Implied speed exceeds max_jump_kn (GPS jump or duplicate MMSI)

    Returns list of df, one per segment
    """
    group = group.sort_values('time').reset_index(drop=True)

    if len(group) < 2:
        return [group]

    dt_minutes = np.diff(group['time'].values).astype('timedelta64[s]').astype(float) / 60
    implied_speed = compute_segment_speeds(
        group['lat'].values, group['lon'].values, group['time'].values)

    splits = np.where(
        (dt_minutes > max_gap_minutes) | (implied_speed > max_jump_kn)
    )[0] + 1

    segments = []
    prev = 0
    for s in splits:
        seg = group.iloc[prev:s].reset_index(drop=True)
        if len(seg) >= 2:
            segments.append(seg)
        prev = s
    seg = group.iloc[prev:].reset_index(drop=True)
    if len(seg) >= 2:
        segments.append(seg)

    return segments


# ===========================================================================
# Quality filtering (interpolate-first approach)
# ===========================================================================

def quality_filter_segment(seg, min_points=3,
                           sog_min=0.05, sog_max=30.0,
                           aog_max=50.0, max_iterations=10,
                           fine_interval_sec=60,
                           sog_trust_min=3.0, sog_trust_max=30.0,
                           max_sog_disagreement=5.0):
    """
    quality_filter_segment does the following:
      1. Removes isolated position jumps from the raw data
      2. Interpolates onto a fine regular grid (default 1-min)
      3. Computes speed on the regular grid
      4. Resolves sog (reported from AIS originally vs position-derived)
      5. Iteratively remove speed/acceleration outliers
      6. Re-interpolate to fill gaps left by removed points
      7. Recompute and verify speeds
      8. Final clamp as safety net

    Returns a df on the finer grid, or None if segment is discarded
    """
    if len(seg) < min_points:
        return None

    seg = seg.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)
    if len(seg) < min_points:
        return None

    # Remove isolated position jumps from raw data
    if len(seg) >= 3:
        implied = compute_segment_speeds(
            seg['lat'].values, seg['lon'].values, seg['time'].values)
        is_jump = np.zeros(len(seg), dtype=bool)
        for i in range(1, len(seg) - 1):
            if implied[i - 1] > 40 and implied[i] > 40:
                is_jump[i] = True
        if len(implied) > 0 and implied[-1] > 40:
            is_jump[-1] = True
        if len(implied) > 0 and implied[0] > 40:
            is_jump[0] = True

        seg = seg[~is_jump].reset_index(drop=True)
        if len(seg) < min_points:
            return None

    # Step 1: Interpolate onto fine regular grid
    t_min = seg['time'].min()
    t_max = seg['time'].max()

    fine_grid = pd.date_range(
        t_min.ceil(f'{fine_interval_sec}s'),
        t_max.floor(f'{fine_interval_sec}s'),
        freq=f'{fine_interval_sec}s'
    )
    if len(fine_grid) < min_points:
        return None

    t_obs = (seg['time'] - t_min).dt.total_seconds().values.astype(float)
    t_fine = (fine_grid - t_min).total_seconds().values.astype(float)

    lat_fine = np.interp(t_fine, t_obs, seg['lat'].values)
    lon_fine = np.interp(t_fine, t_obs, seg['lon'].values)

    # Carry AIS sog onto the fine grid
    if 'sog' in seg.columns and seg['sog'].notna().any():
        sog_ais_fine = np.interp(t_fine, t_obs, seg['sog'].fillna(0).values)
    else:
        sog_ais_fine = np.full(len(fine_grid), np.nan)

    work = pd.DataFrame({
        'time': fine_grid,
        'lat': lat_fine,
        'lon': lon_fine,
        'sog_ais': sog_ais_fine,
    })

    # Step 2: Compute position-derived speed
    work['sog_pos'] = compute_centered_speed(
        work['lat'].values, work['lon'].values, work['time'].values)

    # Step 3: Resolve sog
    sog_pos = work['sog_pos']
    sog_ais = work['sog_ais']

    trust_ais = (
        (sog_ais >= sog_trust_min) &
        (sog_ais <= sog_trust_max) &
        (((sog_ais - sog_pos).abs() < max_sog_disagreement) | sog_pos.isna())
    )
    work['sog_final'] = np.where(trust_ais, sog_ais, sog_pos)

    # Acceleration
    dt_h = fine_interval_sec / 3600
    work['aog_final'] = np.gradient(work['sog_final'].values) / dt_h

    # Initial stats for statistical outlier detection
    sog_mean_init = work['sog_final'].mean()
    sog_std_init = work['sog_final'].std()

    # Step 4: Iteratively filter and re-interpolate
    for _ in range(max_iterations):
        sog = work['sog_final']
        aog = work['aog_final']

        # Apply speed bounds
        cond_speed = (sog >= sog_min) & (sog <= sog_max)

        # Apply acceleration bounds
        cond_accel = (aog.abs() <= aog_max) | aog.isna()

        # Statistical outliers (using initial stats from Step 3)
        if sog_std_init > 0:
            cond_stat = (sog - sog_mean_init).abs() <= 3 * sog_std_init
        else:
            cond_stat = pd.Series(True, index=work.index)

        mask = cond_speed & cond_accel & cond_stat

        if mask.all():
            break

        work = work[mask].reset_index(drop=True)
        if len(work) < min_points:
            return None

        # Re-interpolate to fill gaps left by filtered/removed points
        t_surv = (work['time'] - work['time'].iloc[0]).dt.total_seconds().values.astype(float)
        t_refill = np.arange(0, t_surv[-1] + fine_interval_sec, fine_interval_sec)
        refill_times = work['time'].iloc[0] + pd.to_timedelta(t_refill, unit='s')

        work = pd.DataFrame({
            'time': refill_times,
            'lat': np.interp(t_refill, t_surv, work['lat'].values),
            'lon': np.interp(t_refill, t_surv, work['lon'].values),
        })

        # Recompute sog on the refilled grid
        work['sog_pos'] = compute_centered_speed(
            work['lat'].values, work['lon'].values, work['time'].values)
        work['sog_final'] = work['sog_pos']
        work['aog_final'] = np.gradient(work['sog_final'].values) / dt_h

    if len(work) < min_points:
        return None

    # Final safety filter
    work['sog_final'] = work['sog_final'].clip(lower=0.0, upper=sog_max)

    return work


def quality_filter_ship(group, min_points_total=3, **kwargs):
    """
    Full quality pipeline for one ship's daily data.

    Quality filter ship does the following:
    1. Segments the track at gaps and coordinate jumps
    2. Filters each segment independently with quality_filter_segment
    3. Recombines surviving segments
    """
    segments = segment_track(group)

    filtered_segments = []
    for seg in segments:
        result = quality_filter_segment(seg, **kwargs)
        if result is not None:
            filtered_segments.append(result)

    if not filtered_segments:
        return None

    combined = pd.concat(filtered_segments, ignore_index=True)
    if len(combined) < min_points_total:
        return None

    return combined


# ===========================================================================
# Dimension collapsing
# ===========================================================================

def collapse_dimensions(group, original_group):
    """
    Collapse length/breadth/draught to single best value per ship.
    """
    group = group.copy()
    for col in ['length', 'breadth', 'draught']:
        if col in original_group.columns:
            valid = original_group[col].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                mode_vals = valid.mode()
                group[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else valid.median()
            else:
                group[col] = np.nan
        else:
            group[col] = np.nan
    return group


# ===========================================================================
# Cross-day stitching
# ===========================================================================

def load_with_neighbors(parquet_dir, date, max_stitch_hours=6):
    """
    Load a day's cleaned data plus edge records from neighboring days,
    this is to ensure continuous interpolation if a ship is travelling over
    more than one day in one trip (emulates MATLAB quality control pipeline
    by Florian Aulanier)

    For each ship in the target day:
      - From previous day: last record if within max_stitch_hours
      - From next day: first record if within max_stitch_hours
    """
    parquet_dir = Path(parquet_dir)
    target_file = parquet_dir / f"{date:%Y%m%d}.parquet"

    if not target_file.exists():
        return pd.DataFrame()

    df = pd.read_parquet(target_file)
    df['time'] = pd.to_datetime(df['time'])
    target_mmsis = set(df['mmsi'].unique())
    max_gap = pd.Timedelta(hours=max_stitch_hours)
    edge_records = []

    # Previous day
    prev_file = parquet_dir / f"{date - pd.Timedelta(days=1):%Y%m%d}.parquet"
    if prev_file.exists():
        prev_df = pd.read_parquet(prev_file)
        prev_df['time'] = pd.to_datetime(prev_df['time'])
        for mmsi in target_mmsis:
            ship_prev = prev_df[prev_df['mmsi'] == mmsi]
            if len(ship_prev) == 0:
                continue
            last_record = ship_prev.loc[ship_prev['time'].idxmax()]
            ship_today = df[df['mmsi'] == mmsi]
            earliest_today = ship_today['time'].min()
            gap = earliest_today - last_record['time']
            if pd.Timedelta(0) < gap <= max_gap:
                edge_records.append(last_record)

    # Next day
    next_file = parquet_dir / f"{date + pd.Timedelta(days=1):%Y%m%d}.parquet"
    if next_file.exists():
        next_df = pd.read_parquet(next_file)
        next_df['time'] = pd.to_datetime(next_df['time'])
        for mmsi in target_mmsis:
            ship_next = next_df[next_df['mmsi'] == mmsi]
            if len(ship_next) == 0:
                continue
            first_record = ship_next.loc[ship_next['time'].idxmin()]
            ship_today = df[df['mmsi'] == mmsi]
            latest_today = ship_today['time'].max()
            gap = first_record['time'] - latest_today
            if pd.Timedelta(0) < gap <= max_gap:
                edge_records.append(first_record)

    if edge_records:
        edge_df = pd.DataFrame(edge_records)
        df = pd.concat([df, edge_df], ignore_index=True)

    return df


# ===========================================================================
# Gap-aware interpolation onto model timestep grid
# ===========================================================================

def interpolate_ship_track(group, time_grid, day_start, day_end,
                           max_gap=pd.Timedelta(minutes=60),
                           sog_max=30.0):
    """
    Interpolate a ship's track (already on the fine grid) onto
    the coarser model timestep grid

    Only length_draught timesteps that fall between two observations
    separated by less than max_gap
    """
    group = group.sort_values('time').drop_duplicates(subset='time')

    if len(group) < 2:
        return None

    valid_times = time_grid[
        (time_grid >= day_start) &
        (time_grid < day_end) &
        (time_grid >= group['time'].min()) &
        (time_grid <= group['time'].max())
    ]
    if len(valid_times) == 0:
        return None

    t0 = group['time'].iloc[0]
    t_obs = (group['time'] - t0).dt.total_seconds().values.astype(float)
    t_interp = (valid_times - t0).total_seconds().values.astype(float)

    lat_interp = np.interp(t_interp, t_obs, group['lat'].values)
    lon_interp = np.interp(t_interp, t_obs, group['lon'].values)
    sog_interp = np.interp(t_interp, t_obs, group['sog_final'].values)

    # Gap-aware masking
    obs_times = group['time'].values
    max_gap_ns = max_gap.total_seconds() * 1e9
    valid_mask = np.zeros(len(valid_times), dtype=bool)

    for i, vt in enumerate(valid_times.values):
        before_mask = obs_times <= vt
        after_mask = obs_times >= vt

        if not before_mask.any() or not after_mask.any():
            continue

        t_before = obs_times[before_mask][-1]
        t_after = obs_times[after_mask][0]
        bracket_gap = float((t_after - t_before).astype('timedelta64[ns]'))

        if bracket_gap <= max_gap_ns:
            valid_mask[i] = True

    if not valid_mask.any():
        return None

    # Static fields
    static = {}
    for col in ['mmsi', 'ship_type', 'length', 'breadth', 'draught']:
        if col in group.columns:
            static[col] = group[col].iloc[0]

    result = pd.DataFrame({
        'time': valid_times[valid_mask],
        'lat': lat_interp[valid_mask],
        'lon': lon_interp[valid_mask],
        'sog': np.clip(sog_interp[valid_mask], 0.0, sog_max),
        **{k: v for k, v in static.items()}
    })

    return result


# ===========================================================================
# Processing pipeline
# ===========================================================================

def process_raw_to_clean_daily(df_day, config):
    """
    Pass 1: Per ship
    Segment, interpolate to fine grid, quality filter,
    collapse dimensions. Output is cleaned data on fine 1-min grid
    """
    cleaned = []
    n_rejected = 0

    for mmsi, group in df_day.groupby('mmsi'):
        # Quality filter
        filtered = quality_filter_ship(group.copy())
        if filtered is None:
            n_rejected += 1
            continue

        # Add static metadata from the original raw data
        filtered['mmsi'] = mmsi
        if 'ship_type' in group.columns:
            filtered['ship_type'] = group['ship_type'].iloc[0]
        filtered = collapse_dimensions(filtered, group)

        cleaned.append(filtered)

    if not cleaned:
        return pd.DataFrame(), n_rejected
    return pd.concat(cleaned, ignore_index=True), n_rejected


def resolve_sog_for_stitched(group, sog_trust_min=3.0, sog_trust_max=30.0,
                              max_disagreement=5.0, sog_max=30.0):
    """
    Re-resolve sog (speed over ground) for stitched data (cleaned + neighbor edge records)

    Ensure consistent speed values across day boundaries
    """
    group = group.copy()

    sog_pos = compute_centered_speed(
        group['lat'].values, group['lon'].values, group['time'].values)
    group['sog_pos'] = sog_pos

    if 'sog_final' in group.columns:
        existing = group['sog_final']
        trust_existing = (
            (existing >= sog_trust_min) &
            (existing <= sog_trust_max) &
            (((existing - sog_pos).abs() < max_disagreement) | np.isnan(sog_pos))
        )
        group['sog_final'] = np.where(trust_existing, existing, sog_pos)
    else:
        group['sog_final'] = sog_pos

    # Clamp
    group['sog_final'] = group['sog_final'].clip(lower=0.0, upper=sog_max)

    return group


def process_clean_to_interpolated(df_with_neighbors, date, config):
    """
    Pass 2: Interpolates cleaned data onto the regular model time grid (e.g. 5-min)
    """
    timestep_min = config.get('timestep_minutes', 5)
    max_gap_min = config.get('max_interp_gap_minutes', 60)
    sog_max = config.get('sog_max', 30.0)

    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    time_grid = pd.date_range(
        day_start,
        day_end - pd.Timedelta(minutes=timestep_min),
        freq=f'{timestep_min}min'
    )
    max_gap = pd.Timedelta(minutes=max_gap_min)

    interpolated = []
    for mmsi, group in df_with_neighbors.groupby('mmsi'):
        interp = interpolate_ship_track(
            group, time_grid, day_start, day_end,
            max_gap=max_gap, sog_max=sog_max)
        if interp is not None:
            interpolated.append(interp)

    if not interpolated:
        return pd.DataFrame()
    return pd.concat(interpolated, ignore_index=True)


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

    # Pass 1: Quality filtering
    print(f"Loading {csv_path}...")
    df = load_and_standardize(csv_path)
    print(f"  {len(df)} raw records")

    df = spatial_filter(df, config)
    print(f"  {len(df)} records in zone")

    df['date'] = df['time'].dt.date
    dates_processed = []

    print("\nPass 1: Quality filtering (interpolate-first)...")
    for date, df_day in df.groupby('date'):
        cleaned, n_rejected = process_raw_to_clean_daily(
            df_day.drop(columns='date'), config)
        if len(cleaned) == 0:
            print(f"  {date}: no valid ships ({n_rejected} rejected)")
            continue

        n_ships = cleaned['mmsi'].nunique()
        out_path = clean_dir / f"{date:%Y%m%d}.parquet"
        cleaned.to_parquet(out_path, index=False)
        dates_processed.append(pd.Timestamp(date))
        print(f"  {date}: {n_ships} ships, {len(cleaned)} records "
              f"({n_rejected} rejected)")

    # Pass 2: Cross-day stitching + interpolation
    print("\nPass 2: Stitching and interpolation...")
    for date in dates_processed:
        df_stitched = load_with_neighbors(
            clean_dir, date,
            max_stitch_hours=config.get('max_stitch_hours', 6)
        )
        if len(df_stitched) == 0:
            continue

        # Re-resolve sog for stitched data
        stitched_groups = []
        for mmsi, group in df_stitched.groupby('mmsi'):
            group = (group.sort_values('time')
                     .drop_duplicates(subset='time')
                     .reset_index(drop=True))
            if len(group) >= 2:
                group = resolve_sog_for_stitched(group)
                stitched_groups.append(group)

        if not stitched_groups:
            continue

        df_stitched = pd.concat(stitched_groups, ignore_index=True)

        result = process_clean_to_interpolated(df_stitched, date, config)
        if len(result) == 0:
            print(f"  {date.date()}: no interpolated ships")
            continue

        n_ships = result['mmsi'].nunique()
        n_timesteps = result['time'].nunique()
        out_path = interp_dir / f"{date:%Y%m%d}.parquet"
        result.to_parquet(out_path, index=False)
        print(f"  {date.date()}: {n_ships} ships x {n_timesteps} timesteps "
              f"= {len(result)} rows")

    print("\nDone.")
    print(f"  Cleaned files (1-min):   {clean_dir}/")
    print(f"  Interpolated files:      {interp_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess raw AIS CSV into cleaned, interpolated '
                    'Parquet files.')
    parser.add_argument('--input', required=True, help='Path to raw AIS CSV')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.input, args.output, args.config)
