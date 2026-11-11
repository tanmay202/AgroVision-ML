"""
AgroVision Model 1: In-Season Weather Preprocessing Utility
===========================================================
Production-grade, reusable preprocessing utility for caller-supplied daily
weather observations. Transforms daily NASA POWER or meteorological records
into the exact 10 in-season aggregated features required by the Model 1 Delta ML models.

Key Agronomic & Leakage-Safety Guarantees:
1. Strict Cutoff Enforcement:
   - Autumn (Aus):  Sown May 1,  Cutoff July 31 (~92 days window)
   - Winter (Aman): Sown June 1, Cutoff September 30 (~122 days window)
   - Summer (Boro): Sown Nov 1 (prior year), Cutoff February 28 (~120 days window)
2. Zero Future-Date Leakage:
   - Any observation dated after the season cutoff is strictly rejected/excluded.
3. Explicit Handling of Missing Observations:
   - Tracks observed days vs expected window days, flags incomplete series.
4. Input Normalization:
   - Accepts pandas DataFrame or list-of-dictionaries.
   - Standardizes NASA POWER parameter names (PRECTOTCORR, T2M, T2M_MIN, T2M_MAX, RH2M, ALLSKY_SFC_SW_DWN, WS2M)
     as well as common meteorological aliases.
   - Sanitizes missing sentinels (-999), invalid dates, non-numeric strings, and infinities.
5. Zero Live API Calls:
   - Operates strictly on caller-supplied data; weather retrieval and inference remain separate.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Canonical in-season feature names expected by Model 1 pipelines (exact ordering)
INSEASON_FEATURE_COLUMNS = [
    'InSeason_Precip_Cum',
    'InSeason_Precip_Mean',
    'InSeason_Precip_Max',
    'InSeason_T2M_Mean',
    'InSeason_T2M_Min',
    'InSeason_T2M_Max',
    'InSeason_T2M_Range',
    'InSeason_RH2M_Mean',
    'InSeason_Solar_Mean',
    'InSeason_Wind_Mean',
]

# Season cutoff specifications for West Bengal rice (verified against build_inseason_features.py)
SEASON_CONFIG = {
    'Autumn': {
        'aliases': ['autumn', 'aus', 'pre-kharif'],
        'start_month': 5, 'start_day': 1,
        'cutoff_month': 7, 'cutoff_day': 31,
        'year_offset': 0,
        'expected_days': 92,
    },
    'Winter': {
        'aliases': ['winter', 'aman', 'kharif'],
        'start_month': 6, 'start_day': 1,
        'cutoff_month': 9, 'cutoff_day': 30,
        'year_offset': 0,
        'expected_days': 122,
    },
    'Summer': {
        'aliases': ['summer', 'boro', 'rabi'],
        'start_month': 11, 'start_day': 1,
        'cutoff_month': 2, 'cutoff_day': 28,
        'year_offset': -1,
        'expected_days': 120,
    },
}

# Aliases for meteorological variables
PARAM_ALIASES = {
    'PRECTOTCORR': ['prectotcorr', 'precip', 'precipitation', 'rainfall', 'rain'],
    'T2M': ['t2m', 'temp', 'temperature', 'mean_temp', 't2m_mean'],
    'T2M_MIN': ['t2m_min', 'temp_min', 'min_temp', 't_min', 'min_temperature'],
    'T2M_MAX': ['t2m_max', 'temp_max', 'max_temp', 't_max', 'max_temperature'],
    'RH2M': ['rh2m', 'humidity', 'relative_humidity', 'rh', 'rh2m_mean'],
    'ALLSKY_SFC_SW_DWN': ['allsky_sfc_sw_dwn', 'solar', 'solar_mean', 'radiation', 'solar_radiation', 'insolation', 'sw_dwn'],
    'WS2M': ['ws2m', 'wind', 'wind_speed', 'wind_mean', 'ws_2m'],
}

# Canonical districts in West Bengal
VALID_DISTRICTS = [
    'Alipurduar', 'Bankura', 'Bardhaman', 'Birbhum', 'Cooch Behar',
    'Dakshin Dinajpur', 'Darjeeling', 'Hooghly', 'Howrah',
    'Jalpaiguri', 'Jhargram', 'Kalimpong', 'Maldah', 'Murshidabad',
    'Nadia', 'North 24 Parganas', 'Paschim Bardhaman',
    'Paschim Medinipur', 'Purba Bardhaman', 'Purba Medinipur',
    'Purulia', 'South 24 Parganas', 'Uttar Dinajpur'
]

DISTRICT_ALIASES = {
    '24 paraganas north': 'North 24 Parganas',
    '24 parganas north': 'North 24 Parganas',
    'north 24 parganas': 'North 24 Parganas',
    'north 24-parganas': 'North 24 Parganas',
    'north 24 pargana': 'North 24 Parganas',
    'north 24 paraganas': 'North 24 Parganas',
    '24 paraganas south': 'South 24 Parganas',
    '24 parganas south': 'South 24 Parganas',
    'south 24 parganas': 'South 24 Parganas',
    'south 24-parganas': 'South 24 Parganas',
    'south 24 pargana': 'South 24 Parganas',
    'south 24 paraganas': 'South 24 Parganas',
    'dinajpur dakshin': 'Dakshin Dinajpur',
    'dakshin dinajpur': 'Dakshin Dinajpur',
    'south dinajpur': 'Dakshin Dinajpur',
    'dinajpur uttar': 'Uttar Dinajpur',
    'uttar dinajpur': 'Uttar Dinajpur',
    'north dinajpur': 'Uttar Dinajpur',
    'coochbehar': 'Cooch Behar',
    'cooch behar': 'Cooch Behar',
    'koch bihar': 'Cooch Behar',
    'medinipur west': 'Paschim Medinipur',
    'paschim medinipur': 'Paschim Medinipur',
    'west midnapore': 'Paschim Medinipur',
    'midnapore west': 'Paschim Medinipur',
    'paschim midnapore': 'Paschim Medinipur',
    'medinipur east': 'Purba Medinipur',
    'purba medinipur': 'Purba Medinipur',
    'east midnapore': 'Purba Medinipur',
    'midnapore east': 'Purba Medinipur',
    'purba midnapore': 'Purba Medinipur',
    'alipurduar': 'Alipurduar',
    'alipur duar': 'Alipurduar',
    'bardhaman': 'Bardhaman',
    'burdwan': 'Bardhaman',
    'paschim bardhaman': 'Paschim Bardhaman',
    'west bardhaman': 'Paschim Bardhaman',
    'bardhaman west': 'Paschim Bardhaman',
    'west burdwan': 'Paschim Bardhaman',
    'burdwan west': 'Paschim Bardhaman',
    'purba bardhaman': 'Purba Bardhaman',
    'east bardhaman': 'Purba Bardhaman',
    'bardhaman east': 'Purba Bardhaman',
    'east burdwan': 'Purba Bardhaman',
    'burdwan east': 'Purba Bardhaman',
    'bankura': 'Bankura',
    'birbhum': 'Birbhum',
    'darjeeling': 'Darjeeling',
    'hooghly': 'Hooghly',
    'hugli': 'Hooghly',
    'howrah': 'Howrah',
    'haora': 'Howrah',
    'jalpaiguri': 'Jalpaiguri',
    'jhargram': 'Jhargram',
    'kalimpong': 'Kalimpong',
    'maldah': 'Maldah',
    'malda': 'Maldah',
    'murshidabad': 'Murshidabad',
    'nadia': 'Nadia',
    'purulia': 'Purulia'
}


def normalize_district_name(val: Any) -> str:
    """Standardizes district names across casing and regional spelling variations."""
    if val is None or pd.isna(val):
        return "Unknown"
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null", "unknown", "?"):
        return "Unknown"
    clean = s.lower().replace("-", " ")
    if clean in DISTRICT_ALIASES:
        return DISTRICT_ALIASES[clean]
    title = s.title()
    if title in VALID_DISTRICTS:
        return title
    return s


def normalize_season_name(val: Any) -> str:
    """Standardizes season names to Autumn, Winter, or Summer."""
    if val is None or pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower()
    for canonical, cfg in SEASON_CONFIG.items():
        if s == canonical.lower() or s in cfg['aliases']:
            return canonical
    return s.title()


def get_season_window(year: int, season: str) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """
    Returns (start_date, cutoff_date, expected_days) for a given crop year and season.

    Autumn (Aus):  May 1, year to July 31, year (92 days)
    Winter (Aman): June 1, year to Sep 30, year (122 days)
    Summer (Boro): Nov 1, year-1 to Feb 28, year (120 days)
    """
    canon_season = normalize_season_name(season)
    if canon_season not in SEASON_CONFIG:
        raise ValueError(f"Unknown season '{season}'. Expected one of {list(SEASON_CONFIG.keys())}")

    cfg = SEASON_CONFIG[canon_season]
    start_year = year + cfg['year_offset']
    cutoff_year = year

    start_date = pd.Timestamp(datetime(start_year, cfg['start_month'], cfg['start_day']))
    cutoff_date = pd.Timestamp(datetime(cutoff_year, cfg['cutoff_month'], cfg['cutoff_day'], 23, 59, 59))
    return start_date, cutoff_date, cfg['expected_days']


def _find_column(df_columns: List[str], target_param: str) -> Optional[str]:
    """Finds matching column in DataFrame using canonical name or aliases."""
    # 1. Exact match (case-insensitive)
    lower_cols = {c.lower(): c for c in df_columns}
    if target_param.lower() in lower_cols:
        return lower_cols[target_param.lower()]

    # 2. Alias match
    aliases = PARAM_ALIASES.get(target_param, [])
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]

    return None


def preprocess_inseason_weather(
    daily_weather: Union[pd.DataFrame, List[Dict[str, Any]]],
    district: str,
    year: int,
    season: str,
    max_allowed_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, Any]:
    """
    Preprocess caller-supplied daily weather observations into the exact 10
    Model 1 in-season weather features up to the season-specific cutoff.

    Parameters
    ----------
    daily_weather : pd.DataFrame or list of dicts
        Daily weather records. Must contain date information and meteorological variables
        (PRECTOTCORR, T2M, T2M_MIN, T2M_MAX, RH2M, ALLSKY_SFC_SW_DWN, WS2M or recognized aliases).
    district : str
        District name (e.g. 'Hooghly', 'Purba Bardhaman').
    year : int
        Crop prediction year (e.g. 2018).
    season : str
        Crop season ('Autumn', 'Winter', 'Summer' or agronomic aliases 'Aus', 'Aman', 'Boro').
    max_allowed_date : str or pd.Timestamp, optional
        Optional additional temporal barrier. Observations after this date are excluded.

    Returns
    -------
    dict with structure:
        - 'features': Dict[str, float] with 10 InSeason_* keys
        - 'is_complete': bool (True if all 10 features are valid non-NaN numbers)
        - 'metadata': Dict[str, Any] with audit details:
            * district, year, season
            * season_start, cutoff_date
            * expected_days, observed_days, missing_days, completeness_ratio
            * excluded_post_cutoff_count, excluded_pre_season_count
            * missing_features (list of missing feature names)
        - 'warnings': List[str]
    """
    warnings: List[str] = []

    # 1. Validate and normalize district & season
    canon_district = normalize_district_name(district)
    canon_season = normalize_season_name(season)

    if canon_season not in SEASON_CONFIG:
        raise ValueError(f"Invalid season '{season}'. Expected Autumn (Aus), Winter (Aman), or Summer (Boro).")

    try:
        year_int = int(year)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid crop year '{year}'. Expected integer.")

    start_date, cutoff_date, expected_days = get_season_window(year_int, canon_season)

    # Effective cutoff cannot exceed max_allowed_date if provided
    effective_cutoff = cutoff_date
    if max_allowed_date is not None:
        try:
            user_max = pd.to_datetime(max_allowed_date)
            if hasattr(user_max, 'tz') and user_max.tz is not None:
                user_max = user_max.tz_localize(None)
            if user_max.hour == 0 and user_max.minute == 0 and user_max.second == 0:
                user_max = user_max.replace(hour=23, minute=59, second=59)
            if user_max < effective_cutoff:
                effective_cutoff = user_max
                warnings.append(f"Effective cutoff constrained to user-specified barrier: {user_max.date()}")
        except Exception:
            warnings.append(f"Could not parse max_allowed_date '{max_allowed_date}'; using season cutoff.")

    # 2. Ingest daily weather into DataFrame
    if isinstance(daily_weather, list):
        if len(daily_weather) == 0:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(daily_weather)
    elif isinstance(daily_weather, pd.DataFrame):
        df = daily_weather.copy()
    else:
        raise TypeError(f"daily_weather must be a pandas DataFrame or list of dicts, got {type(daily_weather).__name__}")

    # Empty input handling
    if len(df) == 0:
        empty_features = {col: np.nan for col in INSEASON_FEATURE_COLUMNS}
        return {
            "features": empty_features,
            "is_complete": False,
            "metadata": {
                "district": canon_district,
                "year": year_int,
                "season": canon_season,
                "season_start": start_date.strftime("%Y-%m-%d"),
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "expected_days": expected_days,
                "observed_days": 0,
                "missing_days": expected_days,
                "completeness_ratio": 0.0,
                "excluded_post_cutoff_count": 0,
                "excluded_pre_season_count": 0,
                "missing_features": list(INSEASON_FEATURE_COLUMNS),
            },
            "warnings": ["No daily weather observations provided (empty input)."],
        }

    # 3. Resolve Date column
    date_col = None
    for cand in ['Date', 'date', 'DATE', 'timestamp', 'datetime', 'Reported Date', 'observation_date']:
        if cand in df.columns:
            date_col = cand
            break

    if date_col is None:
        raise ValueError(f"daily_weather must contain a date column ('Date', 'date', etc.). Found: {list(df.columns)}")

    # Parse dates safely
    df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    if hasattr(df['parsed_date'].dt, 'tz') and df['parsed_date'].dt.tz is not None:
        df['parsed_date'] = df['parsed_date'].dt.tz_localize(None)
    invalid_dates_count = int(df['parsed_date'].isna().sum())
    if invalid_dates_count > 0:
        warnings.append(f"Found {invalid_dates_count} observation(s) with invalid or unparseable dates; excluded.")
        df = df.dropna(subset=['parsed_date']).copy()

    # Normalize parsed_date to midnight for daily calendar aggregation
    df['parsed_date'] = df['parsed_date'].dt.normalize()

    # Filter by District if DataFrame contains a district column
    dist_col = None
    for cand in ['District', 'district', 'District Name', 'district_name']:
        if cand in df.columns:
            dist_col = cand
            break

    if dist_col is not None:
        norm_series = df[dist_col].apply(normalize_district_name)
        dist_mask = norm_series == canon_district
        if not dist_mask.any():
            warnings.append(
                f"No rows matching district '{canon_district}' found in weather dataset '{dist_col}' column."
            )
        df = df.loc[dist_mask].copy()

    # 4. Filter strictly by season window and cutoff
    post_cutoff_mask = df['parsed_date'] > effective_cutoff
    excluded_post_cutoff_count = int(post_cutoff_mask.sum())
    if excluded_post_cutoff_count > 0:
        warnings.append(
            f"Leakage protection: Excluded {excluded_post_cutoff_count} observation(s) recorded after cutoff "
            f"({effective_cutoff.strftime('%Y-%m-%d')})."
        )

    pre_season_mask = df['parsed_date'] < start_date
    excluded_pre_season_count = int(pre_season_mask.sum())
    if excluded_pre_season_count > 0:
        warnings.append(
            f"Excluded {excluded_pre_season_count} observation(s) recorded before season start "
            f"({start_date.strftime('%Y-%m-%d')})."
        )

    # Valid window subset
    window_mask = (df['parsed_date'] >= start_date) & (df['parsed_date'] <= effective_cutoff)
    subset = df.loc[window_mask].copy()

    # Deduplicate multiple observations per day if present (take mean)
    if subset['parsed_date'].duplicated().any():
        warnings.append("Duplicate date entries found in weather records; daily observations averaged.")

    # 5. Extract and sanitize meteorological parameters
    sanitized_data: Dict[str, pd.Series] = {}
    missing_params: List[str] = []

    for param in ['PRECTOTCORR', 'T2M', 'T2M_MIN', 'T2M_MAX', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M']:
        col = _find_column(subset.columns.tolist(), param)
        if col is None:
            missing_params.append(param)
            sanitized_data[param] = pd.Series(np.nan, index=subset.index)
            continue

        raw_series = subset[col]
        if isinstance(raw_series, pd.DataFrame):
            raw_series = raw_series.iloc[:, 0]
        raw_series = pd.to_numeric(raw_series, errors='coerce')

        # Convert NASA POWER sentinel -999 to NaN
        raw_series = raw_series.replace([-999, -999.0], np.nan)

        # Replace infinite values with NaN
        raw_series = raw_series.replace([np.inf, -np.inf], np.nan)

        # Agronomic plausibility sanitization
        if param == 'PRECTOTCORR':
            raw_series = raw_series.where(raw_series >= 0, np.nan)
        elif param == 'RH2M':
            raw_series = raw_series.where((raw_series >= 0) & (raw_series <= 100), np.nan)
        elif param in ('ALLSKY_SFC_SW_DWN', 'WS2M'):
            raw_series = raw_series.where(raw_series >= 0, np.nan)
        elif param in ('T2M', 'T2M_MIN', 'T2M_MAX'):
            raw_series = raw_series.where((raw_series >= -50.0) & (raw_series <= 65.0), np.nan)

        sanitized_data[param] = raw_series

    # Physical consistency check: T2M_MIN should not exceed T2M_MAX
    if 'T2M_MIN' in sanitized_data and 'T2M_MAX' in sanitized_data:
        t_min_s = sanitized_data['T2M_MIN']
        t_max_s = sanitized_data['T2M_MAX']
        inconsistent = (t_min_s > t_max_s) & t_min_s.notna() & t_max_s.notna()
        if inconsistent.any():
            warnings.append(
                f"Physical inconsistency: T2M_MIN exceeds T2M_MAX on {int(inconsistent.sum())} observation(s)."
            )

    # If duplicate dates exist, group by parsed_date and take daily mean
    df_clean = pd.DataFrame(sanitized_data, index=subset.index)
    df_clean['Date'] = subset['parsed_date']
    if len(df_clean) > 0:
        df_daily = df_clean.groupby('Date').mean().reset_index()
    else:
        df_daily = df_clean

    observed_days = len(df_daily)
    missing_days = max(0, expected_days - observed_days)
    completeness_ratio = round(observed_days / expected_days, 4) if expected_days > 0 else 0.0

    if observed_days == 0:
        empty_features = {col: np.nan for col in INSEASON_FEATURE_COLUMNS}
        return {
            "features": empty_features,
            "is_complete": False,
            "metadata": {
                "district": canon_district,
                "year": year_int,
                "season": canon_season,
                "season_start": start_date.strftime("%Y-%m-%d"),
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "expected_days": expected_days,
                "observed_days": 0,
                "missing_days": expected_days,
                "completeness_ratio": 0.0,
                "excluded_post_cutoff_count": excluded_post_cutoff_count,
                "excluded_pre_season_count": excluded_pre_season_count,
                "missing_features": list(INSEASON_FEATURE_COLUMNS),
            },
            "warnings": warnings + [
                f"No weather observations available between {start_date.date()} and {effective_cutoff.date()}."
            ],
        }

    if observed_days < expected_days:
        warnings.append(
            f"Observation series incomplete: {observed_days}/{expected_days} days present "
            f"({completeness_ratio * 100.0:.1f}% temporal coverage)."
        )

    # 6. Compute exact 10 aggregated features (formulas identical to build_inseason_features.py)
    p_cum = df_daily['PRECTOTCORR'].sum(min_count=1)
    p_mean = df_daily['PRECTOTCORR'].mean()
    p_max = df_daily['PRECTOTCORR'].max()
    t_mean = df_daily['T2M'].mean()
    t_min = df_daily['T2M_MIN'].min()
    t_max = df_daily['T2M_MAX'].max()
    t_range = t_max - t_min if (pd.notna(t_max) and pd.notna(t_min)) else np.nan
    rh_mean = df_daily['RH2M'].mean()
    solar_mean = df_daily['ALLSKY_SFC_SW_DWN'].mean()
    wind_mean = df_daily['WS2M'].mean()

    def _val(x):
        if pd.isna(x) or np.isinf(x):
            return np.nan
        return round(float(x), 4)

    features = {
        'InSeason_Precip_Cum': _val(p_cum),
        'InSeason_Precip_Mean': _val(p_mean),
        'InSeason_Precip_Max': _val(p_max),
        'InSeason_T2M_Mean': _val(t_mean),
        'InSeason_T2M_Min': _val(t_min),
        'InSeason_T2M_Max': _val(t_max),
        'InSeason_T2M_Range': _val(t_range),
        'InSeason_RH2M_Mean': _val(rh_mean),
        'InSeason_Solar_Mean': _val(solar_mean),
        'InSeason_Wind_Mean': _val(wind_mean),
    }

    missing_feature_cols = [k for k, v in features.items() if pd.isna(v)]
    is_complete = (len(missing_feature_cols) == 0 and observed_days > 0)

    if missing_feature_cols:
        warnings.append(f"Incomplete aggregated features: missing values in {missing_feature_cols}")

    metadata = {
        "district": canon_district,
        "year": year_int,
        "season": canon_season,
        "season_start": start_date.strftime("%Y-%m-%d"),
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "effective_cutoff": effective_cutoff.strftime("%Y-%m-%d"),
        "expected_days": expected_days,
        "observed_days": observed_days,
        "missing_days": missing_days,
        "completeness_ratio": completeness_ratio,
        "max_observation_date": df_daily['Date'].max().strftime("%Y-%m-%d") if len(df_daily) > 0 else None,
        "excluded_post_cutoff_count": excluded_post_cutoff_count,
        "excluded_pre_season_count": excluded_pre_season_count,
        "missing_features": missing_feature_cols,
    }

    return {
        "features": features,
        "is_complete": is_complete,
        "metadata": metadata,
        "warnings": warnings,
    }
