"""
Time Feature Engineering for Intraday Trading Models

This module provides various time-based features that capture market microstructure
patterns throughout the trading day.
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
import math


class TimeFeatureEngineer:
    """
    Extracts and engineers time-based features for intraday trading models.

    Market microstructure considerations:
    - Opening auction (9:30-9:45): High volatility, price discovery
    - Morning session (9:45-11:30): Active trading, trend establishment
    - Lunch lull (11:30-13:30): Lower volume, mean reversion
    - Afternoon session (13:30-15:00): Moderate activity
    - Closing auction (15:00-16:00): High volatility, rebalancing
    """

    def __init__(self, market_open='09:30', market_close='16:00', use_cyclic=True):
        """
        Args:
            market_open: Market opening time (HH:MM format)
            market_close: Market closing time (HH:MM format)
            use_cyclic: Use cyclic encoding for time features (recommended)
        """
        self.market_open = pd.to_datetime(market_open, format='%H:%M').time()
        self.market_close = pd.to_datetime(market_close, format='%H:%M').time()
        self.use_cyclic = use_cyclic

        # Calculate total trading minutes
        open_minutes = self.market_open.hour * 60 + self.market_open.minute
        close_minutes = self.market_close.hour * 60 + self.market_close.minute
        self.total_trading_minutes = close_minutes - open_minutes

    def add_all_time_features(self, df, timestamp_col='timestamp'):
        """
        Add all time features to dataframe.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with added time features
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Basic time components
        df = self._add_basic_time_features(df, timestamp_col)

        # Market session features
        df = self._add_session_features(df, timestamp_col)

        # Cyclic encodings
        if self.use_cyclic:
            df = self._add_cyclic_features(df, timestamp_col)

        # Distance to market events
        df = self._add_event_distance_features(df, timestamp_col)

        # Day of week patterns
        df = self._add_day_of_week_features(df, timestamp_col)

        return df

    def _add_basic_time_features(self, df, timestamp_col):
        """Add basic time components."""
        df['hour'] = df[timestamp_col].dt.hour
        df['minute'] = df[timestamp_col].dt.minute
        df['day_of_week'] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month

        # Minutes since midnight (useful for intraday patterns)
        df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']

        return df

    def _add_session_features(self, df, timestamp_col):
        """
        Add market session indicators.

        Sessions:
        - Opening: 9:30-9:45 (15 min)
        - Morning: 9:45-11:30 (105 min)
        - Midday: 11:30-13:30 (120 min)
        - Afternoon: 13:30-15:00 (90 min)
        - Closing: 15:00-16:00 (60 min)
        """
        hour = df[timestamp_col].dt.hour
        minute = df[timestamp_col].dt.minute
        time_of_day = hour + minute / 60.0

        # Session indicators (one-hot style)
        df['is_opening'] = ((hour == 9) & (minute >= 30) & (minute < 45)).astype(int)
        df['is_morning'] = (((hour == 9) & (minute >= 45)) |
                            ((hour >= 10) & (hour < 11)) |
                            ((hour == 11) & (minute < 30))).astype(int)
        df['is_midday'] = (((hour == 11) & (minute >= 30)) |
                           (hour == 12) |
                           ((hour == 13) & (minute < 30))).astype(int)
        df['is_afternoon'] = (((hour == 13) & (minute >= 30)) |
                              ((hour == 14)) |
                              ((hour == 15) & (minute == 0))).astype(int)
        df['is_closing'] = ((hour == 15) & (minute > 0) | (hour == 16)).astype(int)

        # Minutes since market open
        open_minutes = self.market_open.hour * 60 + self.market_open.minute
        df['minutes_since_open'] = df['minutes_since_midnight'] - open_minutes

        # Minutes until market close
        close_minutes = self.market_close.hour * 60 + self.market_close.minute
        df['minutes_until_close'] = close_minutes - df['minutes_since_midnight']

        # Normalized position in trading day (0 to 1)
        df['trading_day_progress'] = df['minutes_since_open'] / self.total_trading_minutes
        df['trading_day_progress'] = df['trading_day_progress'].clip(0, 1)

        return df

    def _add_cyclic_features(self, df, timestamp_col):
        """
        Add cyclic (sine/cosine) encodings for time features.

        This preserves the circular nature of time (e.g., 23:59 is close to 00:00).
        Transformers benefit from this as it provides smooth, continuous representations.
        """
        # Hour of day (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Minute of hour (60-minute cycle)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # Intraday cycle (captures full trading day)
        # This is especially useful for capturing opening/closing patterns
        intraday_progress = df['minutes_since_midnight'] / (24 * 60)
        df['intraday_sin'] = np.sin(2 * np.pi * intraday_progress)
        df['intraday_cos'] = np.cos(2 * np.pi * intraday_progress)

        # Day of week (7-day cycle)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _add_event_distance_features(self, df, timestamp_col):
        """
        Add features representing distance to important market events.

        These help the model understand proximity to high-volatility periods.
        """
        hour = df[timestamp_col].dt.hour
        minute = df[timestamp_col].dt.minute
        minutes_since_midnight = hour * 60 + minute

        # Distance to market open (in minutes)
        open_minutes = self.market_open.hour * 60 + self.market_open.minute
        df['distance_to_open'] = np.abs(minutes_since_midnight - open_minutes)

        # Distance to market close (in minutes)
        close_minutes = self.market_close.hour * 60 + self.market_close.minute
        df['distance_to_close'] = np.abs(minutes_since_midnight - close_minutes)

        # Distance to lunch (typically 12:00)
        lunch_minutes = 12 * 60
        df['distance_to_lunch'] = np.abs(minutes_since_midnight - lunch_minutes)

        # Normalize distances (0 to 1)
        max_distance = 24 * 60  # Max possible distance in minutes
        df['distance_to_open_norm'] = df['distance_to_open'] / max_distance
        df['distance_to_close_norm'] = df['distance_to_close'] / max_distance
        df['distance_to_lunch_norm'] = df['distance_to_lunch'] / max_distance

        return df

    def _add_day_of_week_features(self, df, timestamp_col):
        """
        Add day-of-week specific features.

        Market patterns vary by day:
        - Monday: Often reversal after weekend
        - Tuesday-Thursday: Normal trading
        - Friday: Position squaring, lower volume
        """
        # One-hot encoding for days (if you prefer)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Weekend proximity
        df['is_week_start'] = (df['day_of_week'] <= 1).astype(int)  # Mon-Tue
        df['is_week_end'] = (df['day_of_week'] >= 3).astype(int)  # Thu-Fri

        return df

    def get_feature_names(self):
        """
        Get list of all feature names that will be created.

        Returns:
            List of feature names
        """
        features = [
            # Basic
            'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
            'minutes_since_midnight',

            # Sessions
            'is_opening', 'is_morning', 'is_midday', 'is_afternoon', 'is_closing',
            'minutes_since_open', 'minutes_until_close', 'trading_day_progress',

            # Event distances
            'distance_to_open', 'distance_to_close', 'distance_to_lunch',
            'distance_to_open_norm', 'distance_to_close_norm', 'distance_to_lunch_norm',

            # Day of week
            'is_monday', 'is_friday', 'is_week_start', 'is_week_end',
        ]

        if self.use_cyclic:
            features.extend([
                'hour_sin', 'hour_cos',
                'minute_sin', 'minute_cos',
                'intraday_sin', 'intraday_cos',
                'day_of_week_sin', 'day_of_week_cos',
                'month_sin', 'month_cos',
            ])

        return features


def get_recommended_time_features(minimal=False):
    """
    Get recommended time features for different use cases.

    Args:
        minimal: If True, return only essential features

    Returns:
        List of recommended feature names
    """
    if minimal:
        # Minimal set - most important features only
        return [
            'hour_sin', 'hour_cos',  # Time of day (cyclic)
            'minutes_since_open',  # Progress through trading day
            'minutes_until_close',  # Proximity to close
            'is_opening', 'is_closing',  # High volatility periods
            'day_of_week_sin', 'day_of_week_cos',  # Day of week (cyclic)
        ]
    else:
        # Full set - comprehensive time features
        return [
            # Cyclic encodings (smooth, continuous)
            'hour_sin', 'hour_cos',
            'minute_sin', 'minute_cos',
            'intraday_sin', 'intraday_cos',
            'day_of_week_sin', 'day_of_week_cos',

            # Trading session indicators
            'is_opening', 'is_morning', 'is_midday', 'is_afternoon', 'is_closing',

            # Progress metrics
            'minutes_since_open',
            'minutes_until_close',
            'trading_day_progress',

            # Event proximity (normalized)
            'distance_to_open_norm',
            'distance_to_close_norm',

            # Day patterns
            'is_monday', 'is_friday',
        ]


# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'timestamp': pd.date_range('2020-11-09 09:00:00', periods=10, freq='1min'),
        'close': [121.5, 121.6, 121.24, 121.09, 121.05, 121.1, 121.12, 121.24, 121.17, 121.15],
        'volume': [28716, 4100, 11497, 6216, 5112, 4078, 3593, 6276, 3024, 4533]
    }
    df = pd.DataFrame(data)

    # Initialize feature engineer
    engineer = TimeFeatureEngineer(market_open='09:30', market_close='16:00', use_cyclic=True)

    # Add all time features
    df_with_features = engineer.add_all_time_features(df)

    # Display sample
    print("Original data:")
    print(df[['timestamp', 'close']].head())

    print("\nWith time features:")
    time_features = engineer.get_feature_names()
    print(df_with_features[['timestamp'] + time_features[:5]].head())

    print(f"\nTotal time features created: {len(time_features)}")
    print("\nRecommended minimal features:")
    print(get_recommended_time_features(minimal=True))