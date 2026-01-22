"""
Time Feature Engineering for Intraday Trading Models (Returns-Based)

This module provides time-based features specifically designed for return prediction
in intraday trading models. Features capture volatility regimes, liquidity patterns,
and market microstructure effects rather than absolute timestamps.

Key Design Principles:
- Features describe "what kind of market conditions exist now" not "what time is it"
- Optimized for 5-minute return prediction horizons
- Captures volatility clustering and regime shifts throughout the trading day
- Uses continuous features over discrete bins where possible (better for transformers)
"""

import numpy as np
import pandas as pd


class TimeFeatureEngineer:
    """
    Extracts and engineers time-based features for return prediction models.

    Market microstructure patterns this captures:
    - Opening auction (9:30-10:00): Very high volatility, price discovery, 2.5x normal vol
    - Morning session (10:00-11:30): High activity, trend establishment, 1.3x vol
    - Lunch lull (11:30-13:30): Lower volume and volatility, mean reversion, 0.5x vol
    - Afternoon session (13:30-15:00): Moderate activity, 0.9x vol
    - Closing auction (15:00-16:00): Very high volatility, rebalancing, 2.0x vol

    These volatility multipliers are observed patterns in equity markets that I found.
    """

    def __init__(self, market_open='09:30', market_close='16:00', use_cyclic=True):
        """
        Args:
            market_open: Market opening time (HH:MM format)
            market_close: Market closing time (HH:MM format)
            use_cyclic: Use cyclic (sin/cos) encoding for periodic features (recommended)
        """
        self.market_open = pd.to_datetime(market_open, format='%H:%M').time()
        self.market_close = pd.to_datetime(market_close, format='%H:%M').time()
        self.use_cyclic = use_cyclic

        # Calculate total trading minutes
        open_minutes = self.market_open.hour * 60 + self.market_open.minute
        close_minutes = self.market_close.hour * 60 + self.market_close.minute
        self.total_trading_minutes = close_minutes - open_minutes

    def add_all_time_features(self, df, timestamp_col='timestamp', return_col='close_return', minimal=False):
        """
        Add time features optimized for return prediction.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            return_col: Name of return column (for validation, not used in feature engineering)
            minimal: If True, only add essential features (recommended for initial testing)

        Returns:
            DataFrame with added time features
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Core features for return prediction
        if self.use_cyclic:
            df = self._add_cyclic_features(df, timestamp_col, minimal=minimal)

        # Volatility regime features
        df = self._add_volatility_regime_features(df, timestamp_col, minimal=minimal)

        # Return-specific temporal features
        df = self._add_return_specific_time_features(df, timestamp_col, minimal=minimal)

        # Liquidity patterns (only in full mode)
        if not minimal:
            df = self._add_liquidity_time_features(df, timestamp_col)
            df = self._add_day_of_week_features(df, timestamp_col)

        return df

    def _add_cyclic_features(self, df, timestamp_col, minimal=False):
        """
        Add cyclic (sine/cosine) encodings for periodic time patterns.

        Cyclic encoding preserves the circular nature of time and provides smooth,
        continuous representations that transformers can learn from effectively.

        For returns: Hour-of-day matters (volatility patterns), but minute-of-hour
        matters less at 5-min horizons, so we optionally exclude it in minimal mode.
        """
        hour = df[timestamp_col].dt.hour
        minute = df[timestamp_col].dt.minute

        # Hour of day (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        if not minimal:
            # Minute of hour (60-minute cycle)
            df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
            df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

            # Intraday position (captures full trading day as one cycle)
            # Useful for capturing symmetric open/close patterns
            minutes_since_midnight = hour * 60 + minute
            intraday_progress = minutes_since_midnight / (24 * 60)
            df['intraday_sin'] = np.sin(2 * np.pi * intraday_progress)
            df['intraday_cos'] = np.cos(2 * np.pi * intraday_progress)

        # Day of week (7-day cycle) - captures weekly patterns (e.g., Monday reversals, Friday positioning)
        day_of_week = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
        df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

        return df

    def _add_volatility_regime_features(self, df, timestamp_col, minimal=False):
        """
        Add features capturing expected volatility regimes throughout the trading day.

        Returns have different magnitudes during
        different market sessions. A 0.1% move at 9:35 AM is "normal", but the same
        move at 12:30 PM is unusual.
        """
        hour = df[timestamp_col].dt.hour
        minute = df[timestamp_col].dt.minute

        # Calculate minutes since market open (0 = 9:30, 390 = 4:00 PM)
        open_hour = self.market_open.hour
        open_minute = self.market_open.minute
        minutes_since_open = (hour - open_hour) * 60 + (minute - open_minute)
        minutes_since_open = np.clip(minutes_since_open, 0, self.total_trading_minutes)

        # Discrete volatility regime multipliers (1.0 = normal volatility)
        time_vol_multiplier = np.ones(len(df))

        # Opening auction (first 15 min): 2.5x normal volatility
        opening_mask = minutes_since_open < 15
        time_vol_multiplier[opening_mask] = 2.5

        # Early morning (15-60 min): 1.5x volatility
        early_mask = (minutes_since_open >= 15) & (minutes_since_open < 60)
        time_vol_multiplier[early_mask] = 1.5

        # Morning continuation (60-120 min): 1.2x volatility
        morning_mask = (minutes_since_open >= 60) & (minutes_since_open < 120)
        time_vol_multiplier[morning_mask] = 1.2

        # Lunch lull (120-240 min): 0.5x volatility
        lunch_mask = (minutes_since_open >= 120) & (minutes_since_open < 240)
        time_vol_multiplier[lunch_mask] = 0.5

        # Afternoon (240-330 min): 0.9x volatility
        afternoon_mask = (minutes_since_open >= 240) & (minutes_since_open < 330)
        time_vol_multiplier[afternoon_mask] = 0.9

        # Closing auction (last 60 min): 2.0x volatility
        closing_mask = minutes_since_open >= 330
        time_vol_multiplier[closing_mask] = 2.0

        df['expected_volatility_regime'] = time_vol_multiplier

        if not minimal:
            # Continuous volatility profile
            # Sum of Gaussian peaks at open/close + dip at lunch
            # This provides smoother gradients for the model to learn from
            df['volatility_curve'] = (
                    2.5 * np.exp(-((minutes_since_open - 0) ** 2) / (30 ** 2)) +  # Opening spike
                    2.0 * np.exp(-((minutes_since_open - 390) ** 2) / (45 ** 2)) +  # Closing spike
                    0.5 * np.exp(-((minutes_since_open - 195) ** 2) / (90 ** 2)) +  # Lunch dip (center)
                    0.8  # Baseline
            )

            # Smooth transition function between regimes (sigmoid-based)
            # Helps model understand we're transitioning between regimes
            df['regime_transition'] = 1 / (1 + np.exp(-0.05 * (minutes_since_open - 195)))

        return df

    def _add_return_specific_time_features(self, df, timestamp_col, minimal=False):
        """
        Time features specifically designed for return prediction.

        Key insight: Returns behave differently based on proximity to market open/close.
        """
        hour = df[timestamp_col].dt.hour
        minute = df[timestamp_col].dt.minute

        # Calculate minutes since market open
        open_hour = self.market_open.hour
        open_minute = self.market_open.minute
        minutes_since_open = (hour - open_hour) * 60 + (minute - open_minute)
        minutes_since_open = np.clip(minutes_since_open, 0, self.total_trading_minutes)

        # Proximity to open (1.0 at open, decays exponentially)
        # Half-life of ~60 minutes (volatility effects fade over first hour)
        df['proximity_to_open'] = np.exp(-minutes_since_open / 60)

        # Proximity to close (0.0 at open, rises exponentially toward close)
        # Half-life of ~60 minutes from close
        minutes_to_close = self.total_trading_minutes - minutes_since_open
        df['proximity_to_close'] = np.exp(-minutes_to_close / 60)

        if not minimal:
            # Binary flags for extreme periods (first/last 30 minutes)
            # These periods have qualitatively different return characteristics
            df['is_first_30min'] = (minutes_since_open < 30).astype(int)
            df['is_last_30min'] = (minutes_to_close < 30).astype(int)

            # Middle-of-day indicator (low at open/close, peaks at lunch)
            # Useful for mean-reversion strategies
            df['midday_indicator'] = np.exp(-((minutes_since_open - 195) ** 2) / (120 ** 2))

            # Opening strength window (first 5 bars = 25 minutes)
            # This period often sets the tone for the day
            df['is_opening_window'] = (minutes_since_open < 25).astype(int)

            # Pre-close positioning window (last 90 minutes)
            # Institutions often rebalance during this period
            df['is_pre_close'] = (minutes_to_close < 90).astype(int)

        return df

    def _add_liquidity_time_features(self, df, timestamp_col):
        """
        Features capturing expected liquidity patterns throughout the day.

        Lower liquidity → higher price impact from same order flow → larger returns.
        This helps the model understand when returns might be "exaggerated" due to
        market microstructure rather than true information.
        """
        hour = df[timestamp_col].dt.hour
        minute = df[timestamp_col].dt.minute

        # Calculate minutes since market open
        open_hour = self.market_open.hour
        open_minute = self.market_open.minute
        minutes_since_open = (hour - open_hour) * 60 + (minute - open_minute)
        minutes_since_open = np.clip(minutes_since_open, 0, self.total_trading_minutes)

        # Expected relative liquidity (1.0 = normal, <1.0 = low, >1.0 = high)
        liquidity = np.ones(len(df))

        # Opening minutes: Very high liquidity (high volume)
        opening_mask = minutes_since_open < 30
        liquidity[opening_mask] = 1.6

        # Morning: Above-average liquidity
        morning_mask = (minutes_since_open >= 30) & (minutes_since_open < 120)
        liquidity[morning_mask] = 1.2

        # Lunch: Low liquidity (low volume, wide spreads)
        lunch_mask = (minutes_since_open >= 120) & (minutes_since_open < 240)
        liquidity[lunch_mask] = 0.6

        # Afternoon: Below-average liquidity
        afternoon_mask = (minutes_since_open >= 240) & (minutes_since_open < 330)
        liquidity[afternoon_mask] = 0.85

        # Closing: High liquidity (rebalancing, high volume)
        closing_mask = minutes_since_open >= 330
        liquidity[closing_mask] = 1.5

        df['expected_liquidity'] = liquidity

        # Inverse liquidity (higher = more price impact expected)
        df['liquidity_impact'] = 1.0 / (liquidity + 0.1)  # Add 0.1 to avoid division issues

        return df

    def _add_day_of_week_features(self, df, timestamp_col):
        """
        Day-of-week specific features.

        Market patterns vary by day:
        - Monday: Often shows weekend news effects, reversals
        - Tuesday-Thursday: Normal trading patterns
        - Friday: Position squaring, lower volume, weekend risk aversion

        Note: Cyclic encoding (sin/cos) is already added, these are supplements.
        """
        day_of_week = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6

        # Binary flags for days with distinct patterns
        df['is_monday'] = (day_of_week == 0).astype(int)
        df['is_friday'] = (day_of_week == 4).astype(int)

        # Week position indicators
        df['is_week_start'] = (day_of_week <= 1).astype(int)  # Mon-Tue: momentum continuation
        df['is_week_end'] = (day_of_week >= 3).astype(int)  # Thu-Fri: risk reduction

        return df

    def get_feature_names(self, minimal=False):
        """
        Get list of all feature names that will be created.

        Args:
            minimal: If True, return only minimal feature set

        Returns:
            List of feature names
        """
        if minimal:
            features = [
                # Cyclic encodings (essential periodic patterns)
                'hour_sin', 'hour_cos',
                'day_of_week_sin', 'day_of_week_cos',

                # Volatility regime (CRITICAL)
                'expected_volatility_regime',

                # Proximity features (essential for return dynamics)
                'proximity_to_open',
                'proximity_to_close',
            ]
        else:
            features = [
                # Cyclic encodings
                'hour_sin', 'hour_cos',
                'minute_sin', 'minute_cos',
                'intraday_sin', 'intraday_cos',
                'day_of_week_sin', 'day_of_week_cos',

                # Volatility regime features
                'expected_volatility_regime',
                'volatility_curve',
                'regime_transition',

                # Return-specific temporal features
                'proximity_to_open',
                'proximity_to_close',
                'is_first_30min',
                'is_last_30min',
                'midday_indicator',
                'is_opening_window',
                'is_pre_close',

                # Liquidity features
                'expected_liquidity',
                'liquidity_impact',

                # Day of week features
                'is_monday',
                'is_friday',
                'is_week_start',
                'is_week_end',
            ]

        return features

    def get_feature_importance_groups(self):
        """
        Returns features grouped by their purpose for analysis.
        Useful for feature importance analysis and ablation studies.
        """
        return {
            'cyclic_time': ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                            'intraday_sin', 'intraday_cos'],
            'volatility_regime': ['expected_volatility_regime', 'volatility_curve',
                                  'regime_transition'],
            'proximity': ['proximity_to_open', 'proximity_to_close'],
            'binary_periods': ['is_first_30min', 'is_last_30min', 'is_opening_window',
                               'is_pre_close'],
            'liquidity': ['expected_liquidity', 'liquidity_impact'],
            'day_patterns': ['day_of_week_sin', 'day_of_week_cos', 'is_monday',
                             'is_friday', 'is_week_start', 'is_week_end'],
            'midday': ['midday_indicator'],
        }


def get_recommended_time_features(use_case='minimal'):
    """
    Get recommended time features for different use cases.

    Args:
        use_case: One of 'minimal', 'standard', 'comprehensive'
            - minimal: 7 features, fastest, good starting point
            - standard: 13 features, balanced performance/complexity
            - comprehensive: 25 features, maximum information

    Returns:
        List of recommended feature names
    """
    if use_case == 'minimal':
        # Minimal set - most important features only (7 features)
        # Start here for initial model development
        return [
            'hour_sin', 'hour_cos',  # Time of day (cyclic)
            'day_of_week_sin', 'day_of_week_cos',  # Day of week (cyclic)
            'expected_volatility_regime',  # Volatility regime (CRITICAL)
            'proximity_to_open',  # Distance from open
            'proximity_to_close',  # Distance from close
        ]

    elif use_case == 'standard':
        # Standard set - good balance (13 features)
        # Use after confirming minimal set works
        return [
            'hour_sin', 'hour_cos',
            'intraday_sin', 'intraday_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'expected_volatility_regime',
            'volatility_curve',
            'proximity_to_open',
            'proximity_to_close',
            'is_first_30min',
            'is_last_30min',
            'expected_liquidity',
        ]

    else:  # comprehensive
        # Full set - all features (25 features)
        # Use for final model or if you have lots of data
        return [
            # Cyclic encodings (6)
            'hour_sin', 'hour_cos',
            'minute_sin', 'minute_cos',
            'intraday_sin', 'intraday_cos',
            'day_of_week_sin', 'day_of_week_cos',

            # Volatility regime (3)
            'expected_volatility_regime',
            'volatility_curve',
            'regime_transition',

            # Proximity and periods (6)
            'proximity_to_open',
            'proximity_to_close',
            'is_first_30min',
            'is_last_30min',
            'is_opening_window',
            'is_pre_close',

            # Liquidity (2)
            'expected_liquidity',
            'liquidity_impact',

            # Day patterns (5)
            'midday_indicator',
            'is_monday',
            'is_friday',
            'is_week_start',
            'is_week_end',
        ]


# Example usage
if __name__ == "__main__":
    # Sample intraday data (5-minute bars)
    data = {
        'timestamp': pd.date_range('2024-01-15 09:30:00', periods=79, freq='5min'),
        'close': np.random.randn(79).cumsum() + 100,  # Random walk
        'volume': np.random.randint(1000, 50000, 79)
    }
    df = pd.DataFrame(data)

    # Calculate returns
    df['close_return'] = df['close'].pct_change()

    print("=" * 70)
    print("TIME FEATURE ENGINEERING FOR RETURN PREDICTION")
    print("=" * 70)

    # Initialize feature engineer
    engineer = TimeFeatureEngineer(market_open='09:30', market_close='16:00', use_cyclic=True)

    # Test minimal features (recommended starting point)
    print("\n[1] MINIMAL FEATURE SET (7 features)")
    print("-" * 70)
    df_minimal = engineer.add_all_time_features(df.copy(), minimal=True)
    minimal_features = engineer.get_feature_names(minimal=True)
    print(f"Features: {minimal_features}")
    print(f"\nSample data (first 5 rows):")
    print(df_minimal[['timestamp', 'close_return'] + minimal_features[:5]].head())

    # Test full features
    print("\n[2] FULL FEATURE SET (25 features)")
    print("-" * 70)
    df_full = engineer.add_all_time_features(df.copy(), minimal=False)
    full_features = engineer.get_feature_names(minimal=False)
    print(f"Total features: {len(full_features)}")

    # Show feature groups
    print("\n[3] FEATURE GROUPS")
    print("-" * 70)
    groups = engineer.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"{group_name:20s}: {len(features)} features")

    # Show volatility regime throughout the day
    print("\n[4] VOLATILITY REGIME ANALYSIS")
    print("-" * 70)
    sample_times = ['09:30', '10:00', '12:00', '14:00', '15:30']
    for time_str in sample_times:
        mask = df_full['timestamp'].dt.strftime('%H:%M') == time_str
        if mask.any():
            vol_regime = df_full.loc[mask, 'expected_volatility_regime'].iloc[0]
            prox_open = df_full.loc[mask, 'proximity_to_open'].iloc[0]
            prox_close = df_full.loc[mask, 'proximity_to_close'].iloc[0]
            print(f"{time_str} - Vol regime: {vol_regime:.2f}x | "
                  f"Prox to open: {prox_open:.3f} | Prox to close: {prox_close:.3f}")