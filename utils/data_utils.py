import os
import pandas as pd
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
import sqlite3
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)

def load_stock_csv(ticker, data_dir=None):
    """
    Load stock CSV into a DataFrame, auto-locating 'data/raw' if not specified.

    Args:
        ticker (str): Stock ticker symbol.
        data_dir (str, optional): Directory where CSV files are stored.

    Returns:
        pd.DataFrame: DataFrame with stock data, indexed by date.
    """
    import os
    import logging
    import pandas as pd

    # Auto-locate data/raw
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        found = False
        # Walk up 3 levels to find project root
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

    # Ensure columns we need exist
    expected_cols = ["open", "high", "low", "close", "volume"]
    for col in expected_cols:
        if col not in df.columns:
            logging.warning(f"Column {col} missing in {ticker} data")

    return df


def load_stock_sqlite(ticker, db_dir=None):
    """
    Load stock data from SQLite into a DataFrame.

    Args:
        ticker (str): Stock ticker symbol (table name).
        db_dir (str, optional): Directory containing data.db.

    Returns:
        pd.DataFrame: DataFrame with stock data, indexed by date.
    """

    # Auto-locate data/processed and data.db
    if db_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        found = False

        # Walk up 3 levels to find project root
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


    # Connect to SQLite
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logging.error(f"Failed to connect to SQLite database: {e}")
        return None

    # Load table
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

    # Match CSV loader behavior
    df.set_index("timestamp", inplace=True)

    print(f"Rows loaded:            {len(df)}")
    print(f"Database:               SQLite")

    # Ensure columns we need exist
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

    # Use the index (which is the timestamp)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Extract time from index
    time = df.index.time

    # Define market hours
    market_open = pd.to_datetime('09:30', format='%H:%M').time()
    market_close = pd.to_datetime('16:00', format='%H:%M').time()

    # Filter
    mask = (time >= market_open) & (time <= market_close)

    print(f"Original data:          {len(df):,} rows")
    print(f"Regular hours:          {mask.sum():,} rows ({mask.sum() / len(df) * 100:.1f}%)")
    print(f"Removed:                {(~mask).sum():,} rows ({(~mask).sum() / len(df) * 100:.1f}%)")

    return df[mask]


def compute_rsi(df, period=14, column="close"):
    """
    Compute Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): Stock data.
        period (int): Lookback period for RSI.
        column (str): Column name to compute RSI on.

    Returns:
        pd.Series: RSI values.
    """

    # copying the dataframe so that it doesn't modify the original one
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    # Ensure numeric
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    rsi = RSIIndicator(df[column], window=period).rsi()
    return rsi


def compute_atr(df, period=14):
    """
    Compute Average True Range (ATR) - measures volatility.

    Returns:
        pd.Series: ATR values
    """
    df = df.copy()

    # Ensure we have OHLC data
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logging.error("ATR requires high, low, close columns")
        return None

    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR is EMA of True Range
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_volume_ratio(df, period=20):
    """
    Compute volume relative to its moving average.

    Returns:
        pd.Series: Current volume / Average volume
    """
    df = df.copy()

    if 'volume' not in df.columns:
        logging.error("Volume column not found")
        return None

    avg_volume = df['volume'].rolling(window=period).mean()

    # Ratio > 1 = above average volume, < 1 = below average
    volume_ratio = df['volume'] / avg_volume

    return volume_ratio


def compute_macd_histogram(df, column="close", fast=12, slow=26, signal=9):
    """
    Compute MACD histogram (MACD - Signal).

    The histogram represents the momentum divergence, which is more
    informative than MACD and Signal separately.

    Returns:
        pd.DataFrame: Column ['MACD_Histogram']
    """
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    macd_indicator = MACD(df[column], window_slow=slow, window_fast=fast, window_sign=signal)

    # Return only the histogram (difference between MACD and Signal)
    macd_df = pd.DataFrame({
        "MACD_Histogram": macd_indicator.macd_diff()  # This is MACD - Signal
    }, index=df.index)

    return macd_df


def compute_sma_deviation(df, period=20, column="close"):
    """
    Compute deviation from SMA as a percentage.
    Tells you if price is above/below its moving average.

    Returns:
        pd.Series: (close - SMA) / SMA * 100
    """
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    sma = df[column].rolling(window=period).mean()
    # Return percentage deviation
    deviation = ((df[column] - sma) / sma) * 100

    return deviation


def add_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26,
                   macd_signal=9, sma_period=20, atr_period=14,
                   price_column="close"):
    """
    Add technical indicators optimized for transformer models.

    Indicators:
    - RSI: Momentum oscillator
    - MACD_Histogram: Trend strength/direction
    - SMA_Deviation: Price position relative to MA
    - ATR: Volatility measure
    - Volume_Ratio: Relative volume activity

    Returns:
        pd.DataFrame: Original DataFrame with indicators
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

    # Volume (normalized, instead of raw)
    df["Volume_Ratio"] = compute_volume_ratio(df, period=20)

    print(f"Indicators added:       RSI({rsi_period}), "
          f"MACD_Histogram({macd_fast},{macd_slow},{macd_signal}), "
          f"SMA_Deviation({sma_period}), ATR({atr_period}), Volume_Ratio(20)")

    return df


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


def create_sequences(df, feature_columns, target_column="close", window_size=20):
    """
    Convert DataFrame into sequences for AI training.

    Returns:
        X: np.array of shape (samples, window_size, features)
        y: np.array of shape (samples,)
    """
    data = df[feature_columns].values
    target = df[target_column].values

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size])

    X = np.array(X)
    y = np.array(y)

    return X, y

def prepare_data_for_ai(
    ticker,
    data_dir=None,
    feature_columns=None,
    target_column="close",
    SQLite=False,
    window_size=20,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    sma_period=20,
    atr_period=14,
    start_idx=None,
    end_idx=None,
    scaler=None
):
    """
    Full pipeline to prepare stock data for AI training or validation.
    """

    if SQLite is False:
        # 1) Load full raw dataframe
        whole_df = load_stock_csv(ticker, data_dir)
        if whole_df is None:
            logging.error(f"CSV not found for {ticker}.")
            return None, None, None
    elif SQLite:
        # 1) Load full raw dataframe
        whole_df = load_stock_sqlite(ticker, data_dir)
        if whole_df is None:
            logging.error(f"CSV not found for {ticker}.")
            return None, None, None

    # 2) Remove extended hours for better accuracy
    whole_df = filter_regular_hours_only(whole_df)

    # 3) Compute indicators on full data to avoid leakage
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

    whole_df = clean_data(whole_df)

    # 4) Slice after indicator computation
    s = start_idx if start_idx is not None else 0
    e = end_idx if end_idx is not None else len(whole_df)
    df = whole_df.iloc[s:e].copy()

    # 5) Select features
    if feature_columns is None:
        feature_columns = ["close","RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
        feature_columns = [c for c in feature_columns if c in df.columns]

    # 6) Scale features
    if scaler is None:

        # Training phase: create and fit new scaler
        #scaler = MinMaxScaler()
        #scaler.fit(df[feature_columns])
        #("Scaler:                 created & fitted MinMaxScaler")

        # Training phase: create and fit new different scaler
        scaler = RobustScaler()
        scaler.fit(df[feature_columns])
        print("Scaler:                 created & fitted RobustScaler")

    else:
        # Validation phase: use existing scaler (no fitting)
        print("Scaler:                 using provided")

    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df[feature_columns])

    # 7) Create sequences
    X, y = create_sequences(df_scaled, feature_columns,
                            target_column=target_column,
                            window_size=window_size)

    print(f"Fully prepared {X.shape[0]} sequences for {ticker}")
    return X, y, scaler


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Example ticker
    ticker = "MSFT"

    print("=" * 70)
    print("DATA UTILS TEST SUITE")
    print("=" * 70)

    # -----------------------------
    # Test 1: Load CSV
    # -----------------------------
    print("\n[TEST 1] Loading CSV...")
    df_csv = load_stock_csv(ticker)
    assert df_csv is not None, "Failed to load CSV"
    assert len(df_csv) > 0, "CSV loaded but empty"
    assert df_csv.index.name == 'timestamp' or isinstance(df_csv.index, pd.DatetimeIndex), \
        "Index should be datetime"
    print("load_stock_csv passed")

    # -----------------------------
    # Test 2: Load SQLite
    # -----------------------------
    print("\n[TEST 2] Loading SQLite...")
    df_sqlite = load_stock_sqlite(ticker)
    assert df_sqlite is not None, "Failed to load database"
    assert len(df_sqlite) > 0, "Database loaded but empty"
    assert isinstance(df_sqlite.index, pd.DatetimeIndex), "Index should be datetime"
    print("load_stock_sqlite passed")

    # Use SQLite data for remaining tests
    df = df_sqlite.copy()

    # -----------------------------
    # Test 3: Filter regular hours
    # -----------------------------
    print("\n[TEST 3] Filtering regular hours...")
    df_filtered = filter_regular_hours_only(df)
    assert len(df_filtered) > 0, "All data removed by filter"
    assert len(df_filtered) < len(df), "Filter didn't remove any data (suspicious)"

    # Check that times are within market hours
    times = df_filtered.index.time
    market_open = pd.to_datetime('09:30', format='%H:%M').time()
    market_close = pd.to_datetime('16:00', format='%H:%M').time()
    assert all((t >= market_open) and (t <= market_close) for t in times[:100]), \
        "Some times outside market hours"
    print("filter_regular_hours_only passed")

    df = df_filtered  # Use filtered data going forward

    # -----------------------------
    # Test 4: Individual indicators
    # -----------------------------
    print("\n[TEST 4] Testing individual indicators...")

    # RSI
    rsi = compute_rsi(df, period=14, column="close")
    assert rsi is not None, "RSI computation failed"
    assert not rsi.isnull().all(), "RSI is all NaN"
    assert rsi.dropna().max() <= 100 and rsi.dropna().min() >= 0, "RSI values out of range [0, 100]"
    print("RSI computed correctly")

    # ATR
    atr = compute_atr(df, period=14)
    assert atr is not None, "ATR computation failed"
    assert not atr.isnull().all(), "ATR is all NaN"
    assert (atr.dropna() >= 0).all(), "ATR should be non-negative"  # ← FIX: dropna() first
    print("ATR computed correctly")

    # Volume Ratio
    vol_ratio = compute_volume_ratio(df, period=20)
    assert vol_ratio is not None, "Volume ratio computation failed"
    assert not vol_ratio.isnull().all(), "Volume ratio is all NaN"
    assert (vol_ratio.dropna() > 0).all(), "Volume ratio should be positive"  # ← Also add dropna()
    print("Volume Ratio computed correctly")

    # MACD Histogram
    macd_df = compute_macd_histogram(df, column="close")
    assert macd_df is not None, "MACD computation failed"
    assert "MACD_Histogram" in macd_df.columns, "MACD_Histogram column missing"
    assert not macd_df["MACD_Histogram"].isnull().all(), "MACD is all NaN"
    print("MACD Histogram computed correctly")

    # SMA Deviation
    sma_dev = compute_sma_deviation(df, period=20, column="close")
    assert sma_dev is not None, "SMA deviation computation failed"
    assert not sma_dev.isnull().all(), "SMA deviation is all NaN"
    print("SMA Deviation computed correctly")

    # -----------------------------
    # Test 5: Add all indicators
    # -----------------------------
    print("\n[TEST 5] Adding all indicators...")
    df_ind = add_indicators(df)
    expected_indicators = ["RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
    for col in expected_indicators:
        assert col in df_ind.columns, f"{col} not added"
    print("add_indicators passed")

    # -----------------------------
    # Test 6: Clean data
    # -----------------------------
    print("\n[TEST 6] Cleaning data...")
    df_clean = clean_data(df_ind)
    assert df_clean.isnull().sum().sum() == 0, "Data still contains NaNs"
    assert len(df_clean) > 0, "All data removed during cleaning"
    print("clean_data passed")

    # -----------------------------
    # Test 7: Scale features
    # -----------------------------
    print("\n[TEST 7] Scaling features...")
    features = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
    features = [f for f in features if f in df_clean.columns]

    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaler.fit(df_clean[features])
    df_scaled = df_clean.copy()
    df_scaled[features] = scaler.transform(df_clean[features])

    assert df_scaled[features].isnull().sum().sum() == 0, "Scaled features contain NaNs"
    print("Scaling passed")

    # -----------------------------
    # Test 8: Create sequences
    # -----------------------------
    print("\n[TEST 8] Creating sequences...")
    window_size = 20
    X, y = create_sequences(df_scaled, feature_columns=features,
                            target_column="close", window_size=window_size)
    assert X.shape[0] == y.shape[0], "Mismatch between X and y"
    assert X.shape[1] == window_size, f"Window size incorrect, expected {window_size}, got {X.shape[1]}"
    assert X.shape[2] == len(features), f"Feature count incorrect, expected {len(features)}, got {X.shape[2]}"
    assert X.shape[0] == len(df_scaled) - window_size, "Incorrect number of sequences"
    print(f"create_sequences passed (shape: {X.shape})")

    # -----------------------------
    # Test 9: Full pipeline (training)
    # -----------------------------
    print("\n[TEST 9] Testing full pipeline (training mode)...")
    X_train, y_train, scaler_train = prepare_data_for_ai(
        ticker,
        SQLite=True,
        window_size=60,
        start_idx=None,
        end_idx=int(len(df) * 0.8)  # First 80% for training
    )
    assert X_train is not None, "Pipeline returned None for X"
    assert y_train is not None, "Pipeline returned None for y"
    assert scaler_train is not None, "Pipeline returned None for scaler"
    assert X_train.shape[0] > 0, "Pipeline returned empty X"
    assert y_train.shape[0] > 0, "Pipeline returned empty y"
    assert X_train.shape[0] == y_train.shape[0], "X and y size mismatch"
    print(f"prepare_data_for_ai (training) passed")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Features: {X_train.shape[2]}")

    # -----------------------------
    # Test 10: Full pipeline (validation)
    # -----------------------------
    print("\n[TEST 10] Testing full pipeline (validation mode)...")
    X_val, y_val, _ = prepare_data_for_ai(
        ticker,
        SQLite=True,
        window_size=60,
        start_idx=int(len(df) * 0.7),
        end_idx=int(len(df) * 0.85),  # 70-85% for validation
        scaler=scaler_train  # Reuse scaler from training
    )
    assert X_val is not None, "Pipeline returned None for X_val"
    assert y_val is not None, "Pipeline returned None for y_val"
    assert X_val.shape[0] > 0, "Pipeline returned empty X_val"
    assert X_val.shape[2] == X_train.shape[2], "Feature count mismatch between train and val"
    print(f"prepare_data_for_ai (validation) passed")
    print(f"  Validation samples: {X_val.shape[0]:,}")

    # -----------------------------
    # Test 11: Data quality checks
    # -----------------------------
    print("\n[TEST 11] Data quality checks...")

    # Check for NaN/Inf
    assert not np.isnan(X_train).any(), "X_train contains NaN"
    assert not np.isnan(y_train).any(), "y_train contains NaN"
    assert not np.isinf(X_train).any(), "X_train contains Inf"
    assert not np.isinf(y_train).any(), "y_train contains Inf"

    # Check shapes are reasonable
    assert X_train.shape[1] > 0, "Window size is 0"
    assert X_train.shape[2] > 0, "No features in X"

    # Check data ranges (scaled data should be reasonable)
    assert X_train.min() > -100, "Scaled values suspiciously low"
    assert X_train.max() < 100, "Scaled values suspiciously high"

    print("Data quality checks passed")

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print(f"\nData Summary:")
    print(f"  Total samples (filtered): {len(df):,}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Features: {X_train.shape[2]}")
    print(f"  Window size: {X_train.shape[1]}")
    print(f"  Feature list: {features}")
    print(f"\nScaler type: RobustScaler")
    print(f"Database: SQLite")
    print("\n" + "=" * 70)