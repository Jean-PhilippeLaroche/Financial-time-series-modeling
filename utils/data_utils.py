"""
data_utils.py - 5-minute forward return prediction with time features
"""

import os
import pandas as pd
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np
import warnings
import sqlite3
from scripts.time_feature_engineering import TimeFeatureEngineer, get_recommended_time_features

import sys
sys.path.insert(0, os.path.dirname(__file__))


import data_pipeline


warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_stock_csv(ticker, data_dir=None):
    """
    Load stock CSV into a DataFrame, auto-locating 'data/raw' if not specified.
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        found = False
        for _ in range(3):
            candidate = os.path.join(base_dir, "data", "raw")
            if os.path.exists(candidate):
                data_dir = candidate
                found = True
                break
            base_dir = os.path.dirname(base_dir)
        if not found:
            logging.error("Could not find 'data/raw' folder in project tree.")
            return None

    file_path = os.path.join(data_dir, f"{ticker}.csv")

    if not os.path.exists(file_path):
        logging.error(f"CSV file not found for {ticker}: {file_path}")
        return None

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Rows loaded:            {len(df)}")
    print(f"Database:               .csv file")

    expected_cols = ["open", "high", "low", "close", "volume"]
    for col in expected_cols:
        if col not in df.columns:
            logging.warning(f"Column {col} missing in {ticker} data")

    return df


def load_stock_sqlite(ticker, db_dir=None):
    """
    Load stock data from SQLite into a DataFrame.
    """
    if db_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        found = False

        for _ in range(3):
            processed_dir = os.path.join(base_dir, "data", "processed")
            db_candidate = os.path.join(processed_dir, "data.db")

            if os.path.exists(db_candidate):
                db_path = db_candidate
                found = True
                break

            base_dir = os.path.dirname(base_dir)

        if not found:
            logging.error("Could not find 'data/processed/data.db' in project tree.")
            return None

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logging.error(f"Failed to connect to SQLite database: {e}")
        return None

    try:
        query = f"SELECT * FROM {ticker}"
        df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    except Exception as e:
        logging.error(f"Failed to load {ticker} from SQLite: {e}")
        return None
    finally:
        conn.close()

    if df.empty:
        logging.warning(f"No data found for {ticker} in database")
        return None

    df.set_index("timestamp", inplace=True)

    print(f"Rows loaded:            {len(df)}")
    print(f"Database:               SQLite")

    expected_cols = ["open", "high", "low", "close", "volume"]
    for col in expected_cols:
        if col not in df.columns:
            logging.warning(f"Column {col} missing in {ticker} data")

    return df


def filter_regular_hours_only(df):
    """
    Keep only regular market hours: 9:30 AM - 4:00 PM EST
    Works with timestamp as index.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    time = df.index.time

    market_open = pd.to_datetime('09:30', format='%H:%M').time()
    market_close = pd.to_datetime('16:00', format='%H:%M').time()

    mask = (time >= market_open) & (time <= market_close)

    print(f"Original data:          {len(df):,} rows")
    print(f"Regular hours:          {mask.sum():,} rows ({mask.sum() / len(df) * 100:.1f}%)")
    print(f"Removed:                {(~mask).sum():,} rows ({(~mask).sum() / len(df) * 100:.1f}%)")

    return df[mask]


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def compute_rsi(df, period=14, column="close"):
    """
    Compute Relative Strength Index (RSI).
    """
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    rsi = RSIIndicator(df[column], window=period).rsi()
    return rsi


def compute_atr(df, period=14):
    """
    Compute Average True Range (ATR) - measures volatility.
    """
    df = df.copy()

    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logging.error("ATR requires high, low, close columns")
        return None

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_volume_ratio(df, period=20):
    """
    Compute volume relative to its moving average.
    """
    df = df.copy()

    if 'volume' not in df.columns:
        logging.error("Volume column not found")
        return None

    avg_volume = df['volume'].rolling(window=period).mean()
    volume_ratio = df['volume'] / avg_volume

    return volume_ratio


def compute_macd_histogram(df, column="close", fast=12, slow=26, signal=9):
    """
    Compute MACD histogram (MACD - Signal).
    """
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    macd_indicator = MACD(df[column], window_slow=slow, window_fast=fast, window_sign=signal)

    macd_df = pd.DataFrame({
        "MACD_Histogram": macd_indicator.macd_diff()
    }, index=df.index)

    return macd_df


def compute_sma_deviation(df, period=20, column="close"):
    """
    Compute deviation from SMA as a percentage.
    """
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    sma = df[column].rolling(window=period).mean()
    deviation = ((df[column] - sma) / sma) * 100

    return deviation


def add_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26,
                   macd_signal=9, sma_period=20, atr_period=14,
                   price_column="close"):
    """
    Add technical indicators optimized for transformer models.
    """
    df = df.copy()

    # Momentum
    df["RSI"] = compute_rsi(df, period=rsi_period, column=price_column)

    # Trend
    macd_df = compute_macd_histogram(df, column=price_column, fast=macd_fast,
                                     slow=macd_slow, signal=macd_signal)
    if macd_df is not None:
        df = pd.concat([df, macd_df], axis=1)

    # Price position
    df["SMA_Deviation"] = compute_sma_deviation(df, period=sma_period,
                                                column=price_column)

    # Volatility
    df["ATR"] = compute_atr(df, period=atr_period)

    # Volume (normalized)
    df["Volume_Ratio"] = compute_volume_ratio(df, period=20)

    print(f"Indicators added:       RSI({rsi_period}), "
          f"MACD_Histogram({macd_fast},{macd_slow},{macd_signal}), "
          f"SMA_Deviation({sma_period}), ATR({atr_period}), Volume_Ratio(20)")

    return df


def add_time_features(df, minimal=True, market_open='09:30', market_close='16:00'):
    """
    Add time-based features for return prediction.

    Args:
        df: DataFrame with datetime index
        minimal: If True, add only essential time features (7 features)
                 If False, add comprehensive time features (25 features)
        market_open: Market opening time (HH:MM format)
        market_close: Market closing time (HH:MM format)

    Returns:
        DataFrame with added time features
    """
    df = df.copy()

    # Initialize time feature engineer
    engineer = TimeFeatureEngineer(
        market_open=market_open,
        market_close=market_close,
        use_cyclic=True
    )

    # Add time features
    # The engineer expects a 'timestamp' column, but our df has timestamp as index
    df_temp = df.reset_index()
    df_temp = df_temp.rename(columns={df_temp.columns[0]: 'timestamp'})

    # Add features
    df_with_time = engineer.add_all_time_features(
        df_temp,
        timestamp_col='timestamp',
        minimal=minimal
    )

    # Set timestamp back as index
    df_with_time = df_with_time.set_index('timestamp')

    # Get feature names for logging
    time_feature_names = engineer.get_feature_names(minimal=minimal)

    print(f"Time features added:    {len(time_feature_names)} features "
          f"({'minimal' if minimal else 'comprehensive'} set)")

    return df_with_time


def clean_data(df):
    """
    Clean the stock data DataFrame by handling missing values and ensuring numeric types.
    """
    df = df.copy()

    # Forward-fill missing data
    df = df.ffill()

    # Drop remaining NaNs if any
    df.dropna(inplace=True)

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Rows after cleaning:    {len(df)}")
    return df


# ============================================================================
# SEQUENCE CREATION WITH FORWARD RETURNS
# ============================================================================

def create_sequences_with_forward_returns(df, feature_columns, target_column="close",
                                          window_size=20, forward_bars=5, raw_close=None):
    """
    Convert DataFrame into sequences for AI training, predicting forward returns.

    Args:
        df: DataFrame with scaled features
        feature_columns: List of feature column names
        target_column: Column to compute forward return from (must be in SCALED df)
        window_size: Lookback period
        forward_bars: Prediction horizon
        raw_close: UNSCALED close prices (Series) - REQUIRED for correct return calculation

    Returns:
        X: np.array of shape (samples, window_size, features)
        y: np.array of shape (samples,) containing forward returns
    """
    df = df.copy()

    if raw_close is None:
        raise ValueError("raw_close must be provided! Cannot calculate returns on scaled prices.")

    # Align raw_close with df index
    raw_close_aligned = raw_close.loc[df.index]

    # Calculate forward return on RAW prices
    df['forward_return'] = (raw_close_aligned.shift(-forward_bars) / raw_close_aligned) - 1

    # Convert close to returns if it's in features (also on RAW prices)
    feature_columns_model = []
    for f in feature_columns:
        if f == 'close':
            # Calculate close returns on RAW prices
            df['close_return'] = raw_close_aligned.pct_change()
            feature_columns_model.append('close_return')
        else:
            feature_columns_model.append(f)

    # Drop NaN
    df = df.dropna()

    # Extract arrays
    data = df[feature_columns_model].values
    target = df['forward_return'].values

    # Create sequences
    X, y = [], []
    max_idx = len(df) - window_size - forward_bars + 1

    for i in range(max_idx):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size - 1])

    X = np.array(X)
    y = np.array(y)

    return X, y


def create_sequences(df, feature_columns, target_column="close", window_size=20):
    """
    LEGACY FUNCTION - Use create_sequences_with_forward_returns().

    Kept for backward compatibility but will convert close to returns automatically.
    """
    logging.warning("Using legacy create_sequences(). Consider using create_sequences_with_forward_returns().")

    # Default to 1-bar forward return
    return create_sequences_with_forward_returns(
        df, feature_columns, target_column, window_size, forward_bars=1
    )


def prepare_data_for_ai(
        ticker,
        data_dir=None,
        feature_columns=None,
        target_column="close",
        SQLite=False,
        window_size=20,
        forward_bars=5,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        sma_period=20,
        atr_period=14,
        start_idx=None,
        end_idx=None,
        scaler=None,
        add_time_features_flag=True,
        time_features_minimal=True
):
    """
    Full pipeline to prepare stock data for AI training with forward return prediction.

    Now predicts forward returns instead of next close price.

    Args:
        forward_bars: Number of bars ahead to predict (e.g., 5 = predict 5-minute return)
        add_time_features_flag: Whether to add time features (default: True)
        time_features_minimal: If True, add minimal time features (7 features)
                               If False, add comprehensive time features (25 features)

    Returns:
        X: Input sequences
        y: Target forward returns
        scaler: Fitted scaler
    """

    # 1) Load data
    if SQLite:
        whole_df = load_stock_sqlite(ticker, data_dir)
        if whole_df is None:
            logging.error(f"SQLite load failed for {ticker}.")
            return None, None, None
    else:
        whole_df = load_stock_csv(ticker, data_dir)
        if whole_df is None:
            logging.error(f"CSV not found for {ticker}.")
            return None, None, None

    # 2) Filter regular hours
    whole_df = filter_regular_hours_only(whole_df)

    # 3) Add indicators
    whole_df = add_indicators(
        whole_df,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period,
        atr_period=atr_period,
        price_column=target_column
    )

    # 4) Add time features (NEW)
    if add_time_features_flag:
        whole_df = add_time_features(whole_df, minimal=time_features_minimal)

    whole_df = clean_data(whole_df)

    # 5) Slice if needed
    s = start_idx if start_idx is not None else 0
    e = end_idx if end_idx is not None else len(whole_df)
    df = whole_df.iloc[s:e].copy()

    # 6) Select features
    if feature_columns is None:
        # Default feature columns
        feature_columns = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]

        # Add time features if they were added
        if add_time_features_flag:
            time_feature_names = get_recommended_time_features(
                use_case='minimal' if time_features_minimal else 'standard'
            )
            feature_columns.extend(time_feature_names)

        # Filter to only columns that exist
        feature_columns = [c for c in feature_columns if c in df.columns]

    # 7) Scale features (BEFORE converting close to returns)
    # We scale the raw features, then sequence creation handles return conversion
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_columns])
        print("Scaler:                 created & fitted MinMaxScaler.")
    else:
        print("Scaler:                 using provided")

    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df[feature_columns])

    # 8) Create sequences with forward returns
    # Need to pass raw close for return calculation
    raw_close = df[target_column].copy()

    X, y = create_sequences_with_forward_returns(
        df_scaled,
        feature_columns,
        target_column=target_column,
        window_size=window_size,
        forward_bars=forward_bars,
        raw_close=raw_close
    )

    print(f"Sequences prepared:     {X.shape[0]:,} sequences")
    print(f"Target:                 {forward_bars}-bar forward returns")
    print(f"Feature shape:          {X.shape}")
    print(f"Target stats:           mean={y.mean():.6f}, std={y.std():.6f}")

    return X, y, scaler


# ============================================================================
# HELPER FUNCTION FOR MAIN.PY
# ============================================================================

def prepare_sequences_from_df(df, feature_columns, window_size=20, forward_bars=5, scaler=None):
    """
    Simplified function for main.py to use after loading/filtering/adding indicators.
    """
    # Select features
    if feature_columns is None:
        feature_columns = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
        feature_columns = [c for c in feature_columns if c in df.columns]

    # SAVE RAW CLOSE BEFORE SCALING
    raw_close = df['close'].copy()

    # Scale
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_columns])
        print("Scaler:                 created & fitted MinMaxScaler")
    else:
        print("Scaler:                 using provided MinMaxScaler")

    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df[feature_columns])

    # Create sequences
    X, y = create_sequences_with_forward_returns(
        df_scaled,
        feature_columns,
        target_column="close",
        window_size=window_size,
        forward_bars=forward_bars,
        raw_close=raw_close
    )

    print(f"Sequences:              {X.shape[0]:,} samples")
    print(f"Target:                 {forward_bars}-bar forward returns")
    print(f"Return stats:           mean={y.mean():.6f}, std={y.std():.6f}")

    return X, y, scaler


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("DATA UTILS TEST")
    print("=" * 70)

    ticker = "MSFT"

    # Test 1: Basic functionality (without time features)
    print("\n[TEST 1] Testing basic forward returns functionality...")
    df = load_stock_sqlite(ticker)
    df = filter_regular_hours_only(df)
    df = add_indicators(df)
    df = clean_data(df)

    for forward_bars in [1, 5, 10]:
        print(f"\n--- Testing {forward_bars}-bar forward returns (no time features) ---")

        X, y, scaler = prepare_sequences_from_df(
            df.iloc[:10000],
            feature_columns=["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"],
            window_size=20,
            forward_bars=forward_bars,
            scaler=None
        )

        assert X is not None, "X is None"
        assert y is not None, "y is None"
        assert X.shape[0] == y.shape[0], "X and y length mismatch"
        assert not np.isnan(X).any(), "X contains NaN"
        assert not np.isnan(y).any(), "y contains NaN"
        assert abs(y.mean()) < 0.01, f"Return mean suspiciously high: {y.mean():.6f}"

        print(f"{forward_bars}-bar forward returns passed")

    # Test 2: With time features (minimal)
    print("\n[TEST 2] Testing with minimal time features...")
    df = load_stock_sqlite(ticker)
    df = filter_regular_hours_only(df)
    df = add_indicators(df)
    df = add_time_features(df, minimal=True)
    df = clean_data(df)

    engineer = TimeFeatureEngineer()
    time_feature_names = engineer.get_feature_names(minimal=True)

    feature_cols = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
    feature_cols.extend(time_feature_names)

    X, y, scaler = prepare_sequences_from_df(
        df.iloc[:10000],
        feature_columns=feature_cols,
        window_size=20,
        forward_bars=5,
        scaler=None
    )

    print(f"Feature count:          {X.shape[2]} features (6 technical + {len(time_feature_names)} time)")
    print(f"Expected features:      {len(feature_cols)}")
    assert X.shape[2] == len(
        time_feature_names) + 6, f"Feature count mismatch: {X.shape[2]} vs {len(time_feature_names) + 6}"
    assert not np.isnan(X).any(), "X contains NaN with time features"
    print("Minimal time features test passed")

    # Test 3: With comprehensive time features
    print("\n[TEST 3] Testing with comprehensive time features...")
    df = load_stock_sqlite(ticker)
    df = filter_regular_hours_only(df)
    df = add_indicators(df)
    df = add_time_features(df, minimal=False)
    df = clean_data(df)

    time_feature_names_full = engineer.get_feature_names(minimal=False)
    feature_cols_full = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
    feature_cols_full.extend(time_feature_names_full)

    X, y, scaler = prepare_sequences_from_df(
        df.iloc[:10000],
        feature_columns=feature_cols_full,
        window_size=20,
        forward_bars=5,
        scaler=None
    )

    print(f"Feature count:          {X.shape[2]} features (6 technical + {len(time_feature_names_full)} time)")
    assert X.shape[2] == len(time_feature_names_full) + 6, "Full feature count mismatch"
    assert not np.isnan(X).any(), "X contains NaN with full time features"
    print("Comprehensive time features test passed")

    # Test 4: Test prepare_data_for_ai with time features
    print("\n[TEST 4] Testing prepare_data_for_ai() with time features...")
    X, y, scaler = prepare_data_for_ai(
        ticker="MSFT",
        SQLite=True,
        window_size=20,
        forward_bars=5,
        start_idx=0,
        end_idx=10000,
        add_time_features_flag=True,
        time_features_minimal=True
    )

    assert X is not None, "prepare_data_for_ai returned None"
    assert X.shape[2] > 6, "Time features not added in prepare_data_for_ai"
    print(f"prepare_data_for_ai() with time features passed ({X.shape[2]} features)")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)