"""
backtest.py
-----------
Proper backtesting framework for stock trading strategies.
Simulates step-by-step trading with realistic constraints:
- No lookahead bias (predict one day at a time)
- Transaction costs
- Position sizing constraints
- Comprehensive performance metrics

Key metrics calculated:
- Total return
- Expected return
- Sharpe ratio
- Max drawdown
- Win rate (% of profitable trades)
"""

import numpy as np
import pandas as pd
import logging
import torch
from scripts.train import DEVICE
import time

logging.basicConfig(level=logging.INFO)

# Colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


class Backtester:
    """
    Backtesting engine for trading strategies based on return predictions.

    UPDATED: Now works with models that predict forward returns instead of prices.
    """

    def __init__(self,
                 model,
                 scaler,
                 df,
                 feature_columns,
                 window_size=60,
                 forward_bars=5,  # NEW: prediction horizon
                 initial_balance=10000,
                 transaction_cost_pct=0.02,
                 position_size_pct=0.95,
                 max_shares_per_trade=100,
                 threshold=0.005):  # NEW: 0.5% return threshold (was price-based)
        """
        Initialize backtesting engine.

        Args:
            model: Trained PyTorch model (predicts returns, not prices!)
            scaler: Fitted RobustScaler from training
            df: DataFrame with price data and indicators
            feature_columns: List of feature column names
            window_size: Lookback window for predictions
            forward_bars: Prediction horizon (e.g., 5 = predict 5 minutes ahead)
            initial_balance: Starting cash
            transaction_cost_pct: Total transaction cost
            position_size_pct: Fraction of cash to use per trade
            max_shares_per_trade: Maximum shares per trade
            threshold: Return threshold for signals (e.g., 0.005 = 0.5% return)
        """
        self.model = model
        self.scaler = scaler
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.forward_bars = forward_bars  # NEW
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.position_size_pct = position_size_pct
        self.max_shares_per_trade = max_shares_per_trade
        self.threshold = threshold

        # State variables
        self.cash = initial_balance
        self.shares = 0
        self.portfolio_history = []
        self.trades = []

        # Set model to evaluation mode
        self.model.eval()
        self.device = DEVICE

        print(f"Backtester initialized (FORWARD RETURN MODE)")
        print(f"  Model predicts: {forward_bars}-bar forward returns")
        print(f"  Signal threshold: {threshold * 100:.2f}% return")
        print(f"  Device: {self.device}")

    def prepare_sequence(self, end_idx):
        """
        Prepare a single sequence for prediction.

        UPDATED: Now creates sequences for return-based models.

        Args:
            end_idx: End index in dataframe

        Returns:
            torch.Tensor: Sequence ready for model
        """
        start_idx = end_idx - self.window_size

        if start_idx < 0:
            raise ValueError(f"Not enough data: need {self.window_size} bars, got {end_idx}")

        # Extract window
        window_df = self.df.iloc[start_idx:end_idx].copy()

        # Save raw close for return calculation
        raw_close = window_df['close'].copy()

        # Scale features
        window_scaled = self.scaler.transform(window_df[self.feature_columns])
        window_scaled_df = pd.DataFrame(window_scaled, columns=self.feature_columns, index=window_df.index)

        # Convert close to returns
        if 'close' in self.feature_columns:
            close_returns = raw_close.pct_change()
            close_idx = self.feature_columns.index('close')
            window_scaled_df.iloc[:, close_idx] = close_returns.fillna(0).values

        # Convert to tensor: (1, window_size, features)
        sequence = torch.tensor(window_scaled_df.values, dtype=torch.float32).unsqueeze(0).to(self.device)

        return sequence

    def predict_next_return(self, end_idx):
        """
        Predict the forward return.


        Args:
            end_idx: Current timestep

        Returns:
            float: Predicted return (e.g., 0.0025 = +0.25%)
        """
        sequence = self.prepare_sequence(end_idx)

        with torch.no_grad():
            model_output = self.model(sequence)

            # Handle tuple output (model returns predictions + attention + activations)
            if isinstance(model_output, tuple):
                predicted_return = model_output[0].item()
            else:
                predicted_return = model_output.item()

        return predicted_return

    def predict_all_returns_batch(self, start_idx, end_idx, batch_size=1024):
        """
        Predict returns for all timesteps using efficient batching.


        Args:
            start_idx: Start index
            end_idx: End index
            batch_size: Batch size for GPU

        Returns:
            np.array: Predicted returns for each timestep
        """
        all_predictions = []

        # Step 1: Prepare all sequences
        num_timesteps = end_idx - start_idx
        print(f"\n{BLUE}Preparing {num_timesteps} sequences for batch prediction...{RESET}")

        sequences = []
        failed_indices = []

        count = 0
        t0 = time.time()

        for i in range(start_idx, end_idx):
            try:
                seq = self.prepare_sequence(i)
                sequences.append(seq)
                count += 1
                if count % 10000 == 0:
                    progress = (count * 100) / num_timesteps
                    print(f"    Processed {count}/{num_timesteps} sequences ({progress:.2f}%)")
            except Exception as e:
                logging.warning(f"Failed to prepare sequence at index {i}: {e}")
                sequences.append(None)
                failed_indices.append(i - start_idx)

        print(f"Time: {time.time() - t0:.2f}s")

        # Step 2: Process in batches on GPU
        t0 = time.time()
        print(f"\n{BLUE}Running batch predictions (batch_size={batch_size})...{RESET}")

        num_batches = (len(sequences) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]

            # Separate valid and invalid
            valid_indices = []
            valid_sequences = []
            for local_idx, seq in enumerate(batch_sequences):
                if seq is not None:
                    valid_indices.append(local_idx)
                    valid_sequences.append(seq)

            # Handle empty batch
            if len(valid_sequences) == 0:
                all_predictions.extend([0.0] * len(batch_sequences))
                continue

            # Stack and predict
            batch_tensor = torch.cat(valid_sequences, dim=0).to(self.device)

            with torch.no_grad():
                model_output = self.model(batch_tensor)

                # Handle tuple output
                if isinstance(model_output, tuple):
                    batch_predictions = model_output[0]
                else:
                    batch_predictions = model_output

            # Convert to numpy
            batch_predictions_np = batch_predictions.cpu().numpy().flatten()

            # Insert predictions
            batch_results = []
            pred_idx = 0
            for local_idx in range(len(batch_sequences)):
                if local_idx in valid_indices:
                    batch_results.append(float(batch_predictions_np[pred_idx]))
                    pred_idx += 1
                else:
                    batch_results.append(0.0)  # Fallback for failed sequences

            all_predictions.extend(batch_results)

            # Progress
            if batch_idx == num_batches - 1:
                print(f"    Completed {num_batches} batches")

        print(f"    Batch prediction complete: {len(all_predictions)} returns predicted")
        print(f"Time: {time.time() - t0:.2f}s\n")

        return np.array(all_predictions)

    def generate_signal(self, predicted_return, current_price):
        """
        Generate trading signal based on predicted return.

        Args:
            predicted_return: Model's predicted return (e.g., 0.0025 = +0.25%)
            current_price: Current market price (for calculating implied price)

        Returns:
            str: 'BUY', 'SELL', or 'HOLD'
        """
        # Buy if predicted return is significantly positive
        if predicted_return > self.threshold:
            return 'BUY'

        # Sell if predicted return is significantly negative
        elif predicted_return < -self.threshold:
            return 'SELL'

        else:
            return 'HOLD'

    def execute_trade(self, action, current_price, current_date):
        """
        Execute a trade with realistic constraints and costs.

        """
        if action == 'BUY' and self.cash > 0:
            max_affordable = int((self.cash * self.position_size_pct) / current_price)
            shares_to_buy = min(max_affordable, self.max_shares_per_trade)

            if shares_to_buy > 0:
                gross_cost = shares_to_buy * current_price
                transaction_fee = gross_cost * self.transaction_cost_pct
                total_cost = gross_cost + transaction_fee

                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.shares += shares_to_buy

                    self.trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'gross_cost': gross_cost,
                        'transaction_fee': transaction_fee,
                        'total_cost': total_cost
                    })

                    logging.debug(f"BUY: {shares_to_buy} shares @ ${current_price:.2f} | Fee: ${transaction_fee:.2f}")

        elif action == 'SELL' and self.shares > 0:
            shares_to_sell = self.shares
            gross_proceeds = shares_to_sell * current_price
            transaction_fee = gross_proceeds * self.transaction_cost_pct
            net_proceeds = gross_proceeds - transaction_fee

            self.cash += net_proceeds

            self.trades.append({
                'date': current_date,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'gross_proceeds': gross_proceeds,
                'transaction_fee': transaction_fee,
                'net_proceeds': net_proceeds
            })

            self.shares = 0

            logging.debug(f"SELL: {shares_to_sell} shares @ ${current_price:.2f} | Fee: ${transaction_fee:.2f}")

    def run(self, start_idx=None, end_idx=None):
        """
        Run the backtest step-by-step through the data.

        Args:
            start_idx: Start index (default: window_size)
            end_idx: End index (default: len(df))

        Returns:
            dict: Backtest results
        """
        if start_idx is None:
            start_idx = self.window_size
        if end_idx is None:
            end_idx = len(self.df)

        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}STARTING BACKTEST{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")
        print(f"Period:                 index {start_idx} to {end_idx} ({end_idx - start_idx} timesteps)")
        print(f"Initial balance:        ${self.initial_balance:,.2f}")
        print(f"Transaction cost:       {self.transaction_cost_pct * 100}%")
        print(f"Signal threshold:       {self.threshold * 100:.2f}% return")
        print(f"Prediction horizon:     {self.forward_bars} bars ahead")

        # Reset state
        self.cash = self.initial_balance
        self.shares = 0
        self.portfolio_history = []
        self.trades = []

        # Batch predict all returns at once
        all_predicted_returns = self.predict_all_returns_batch(
            start_idx,
            end_idx,
            batch_size=1024
        )

        # Process signals and execute trades
        t0 = time.time()
        print(f"{BLUE}Processing trading signals and executing trades...{RESET}")
        num_timesteps = end_idx - start_idx

        for idx, i in enumerate(range(start_idx, end_idx)):
            current_date = self.df.index[i]
            current_price = self.df.iloc[i]['close']
            predicted_return = all_predicted_returns[idx]

            # Calculate implied future price (for logging/visualization)
            predicted_price = current_price * (1 + predicted_return)

            # Generate trading signal based on predicted return
            signal = self.generate_signal(predicted_return, current_price)

            # Execute trade if signal is not HOLD
            if signal in ['BUY', 'SELL']:
                self.execute_trade(signal, current_price, current_date)

            # Update portfolio value
            portfolio_value = self.cash + (self.shares * current_price)

            self.portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'shares': self.shares,
                'current_price': current_price,
                'predicted_return': predicted_return,
                'predicted_price': predicted_price,  # Keep for visualization
                'signal': signal
            })

            # Progress logging
            if (idx + 1) % 20000 == 0:
                progress = ((idx + 1) / num_timesteps) * 100
                print(f"    Processed {idx + 1}/{num_timesteps} timesteps ({progress:.1f}%)")

        print(f"Time: {time.time() - t0:.2f}s\n")

        # Calculate metrics
        results = self.calculate_metrics()

        # Print summary
        print(f"{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}BACKTEST RESULTS{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")
        print(f"Final Portfolio Value:    ${results['final_value']:,.2f}")
        print(f"Total Return:             {results['total_return']:.2f}%")
        print(f"Expected Return (daily):  {results['expected_return']:.4f}%")
        print(f"Sharpe Ratio:             {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:             {results['max_drawdown']:.2f}%")
        print(f"Win Rate:                 {results['win_rate']:.2f}%")
        print(f"Total Trades:             {results['total_trades']}")
        print(f"Total Fees Paid:          ${results['total_fees']:,.2f}")
        print(f"{BLUE}{'-' * 70}{RESET}")
        print(f"Buy & Hold Return:        {results['buy_hold_return']:.2f}%")
        print(f"Outperformance:           {results['outperformance']:+.2f}%")
        print(f"{BLUE}{'=' * 70}{RESET}\n")

        return results

    def calculate_metrics(self):
        """
        Calculate comprehensive performance metrics.

        """
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # Basic metrics
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_balance - 1) * 100

        # Daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()

        # Expected return
        expected_return = daily_returns.mean() * 100

        # Sharpe Ratio (annualized)
        # For 1-min data: 252 days * 390 minutes = 98,280 bars per year
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(98280)
        else:
            sharpe_ratio = 0.0

        # Max Drawdown
        cumulative_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100

        # Trade statistics
        total_trades = len(self.trades)

        if total_trades > 0:
            profitable_trades = 0
            total_profit = 0

            buy_price = None
            for trade in self.trades:
                if trade['action'] == 'BUY':
                    buy_price = trade['price']
                elif trade['action'] == 'SELL' and buy_price is not None:
                    profit = (trade['price'] - buy_price) * trade['shares']
                    total_profit += profit
                    if profit > 0:
                        profitable_trades += 1
                    buy_price = None

            win_rate = (profitable_trades / (total_trades / 2)) * 100 if total_trades > 1 else 0
        else:
            win_rate = 0
            total_profit = 0

        # Buy and hold
        first_price = portfolio_df['current_price'].iloc[0]
        last_price = portfolio_df['current_price'].iloc[-1]
        buy_hold_return = (last_price / first_price - 1) * 100

        # Total fees
        total_fees = sum(trade.get('transaction_fee', 0) for trade in self.trades)

        return {
            'final_value': final_value,
            'total_return': total_return,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'portfolio_history': portfolio_df,
            'trades': trades_df
        }


# ---------------------------
# Convenience function
# ---------------------------
def run_backtest(model, scaler, df, feature_columns, window_size=60,
                 forward_bars=5, initial_balance=10000, transaction_cost_pct=0.02,
                 threshold=0.005, start_idx=None, end_idx=None):
    """
    Convenience function to run a backtest.

    Args:
        model: Trained PyTorch model (predicts returns)
        scaler: Fitted RobustScaler
        df: DataFrame with price data
        feature_columns: List of feature columns
        window_size: Lookback window
        forward_bars: Prediction horizon (e.g., 5 = 5 minutes ahead)
        initial_balance: Starting cash
        transaction_cost_pct: Transaction cost
        threshold: Return threshold for signals (e.g., 0.005 = 0.5%)
        start_idx: Optional start index
        end_idx: Optional end index

    Returns:
        dict: Backtest results
    """
    backtester = Backtester(
        model=model,
        scaler=scaler,
        df=df,
        feature_columns=feature_columns,
        window_size=window_size,
        forward_bars=forward_bars,
        initial_balance=initial_balance,
        transaction_cost_pct=transaction_cost_pct,
        threshold=threshold
    )

    return backtester.run(start_idx=start_idx, end_idx=end_idx)


# ---------------------------
# Example usage / testing
# ---------------------------
if __name__ == "__main__":

    if __name__ == "__main__":

        import time
        import json
        import logging
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import RobustScaler
        from utils.data_utils import (
            load_stock_csv,
            filter_regular_hours_only,
            add_indicators,
            clean_data
        )

        logging.basicConfig(level=logging.INFO)

        # -------------------
        # Config for the test
        # -------------------
        ticker = "AAPL"
        train_size = 0.8
        window_size = 60
        forward_bars = 5

        initial_balance = 10_000
        transaction_cost = 0.005  # 0.5%
        threshold = 0.005  # 0.5% return threshold
        epochs = 0  # just for the summary JSON

        logging.info("Step 1: Loading and preparing full dataframe...")
        df_raw = load_stock_csv(ticker)
        if df_raw is None:
            logging.error("Could not load raw CSV for ticker; exiting.")
            raise SystemExit(1)

        # Filter regular hours
        df_filtered = filter_regular_hours_only(df_raw)

        # Add indicators
        df_ind = add_indicators(df_filtered)
        df_clean = clean_data(df_ind)

        n_total = len(df_clean)
        split_idx = int(n_total * train_size)
        logging.info(f"Total cleaned rows: {n_total}, train/val split index: {split_idx}")

        # -------------------
        # Feature columns
        # -------------------
        feature_columns = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]
        feature_columns = [c for c in feature_columns if c in df_clean.columns]
        logging.info(f"Using feature columns: {feature_columns}")

        # -------------------
        # Fit scaler on train slice only to prevent leakage
        # -------------------
        train_df = df_clean.iloc[:split_idx]
        scaler = RobustScaler()
        scaler.fit(train_df[feature_columns])


        # -------------------
        # Dummy PyTorch model for testing backtest logic
        # -------------------
        class OscillatingReturnModel(nn.Module):
            """
            Oscillating model that predicts returns.

            - Takes (batch, window, features)
            - Predicts oscillating returns: +delta%, -delta%, +delta%, -delta%, ...
            - This forces BUY/SELL signals in the backtest

            Example:
              delta_return = 0.01 (1%)
              Predictions alternate: +1%, -1%, +1%, -1%, ...
            """

            def __init__(self, input_size: int, delta_return: float = 0.01):
                super().__init__()
                self.delta_return = delta_return  # Return magnitude (e.g., 0.01 = 1%)
                # Keep a tiny linear layer to mimic structure
                self.fc = nn.Linear(input_size, 1)

            def forward(self, x):
                # x shape: (batch, window, features)
                batch_size = x.size(0)
                device = x.device
                dtype = x.dtype

                # Create oscillating returns: +delta, -delta, +delta, -delta, ...
                # +delta for even indices, -delta for odd
                signs = torch.tensor(
                    [1.0 if i % 2 == 0 else -1.0 for i in range(batch_size)],
                    device=device,
                    dtype=dtype
                )

                # Predicted returns (e.g., +0.01 or -0.01)
                predicted_returns = signs * self.delta_return

                return predicted_returns.unsqueeze(1)  # Shape: (batch, 1)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OscillatingReturnModel(
            input_size=len(feature_columns),
            delta_return=0.01  # Predict ±1% returns
        ).to(device)
        model.eval()

        logging.info("Created oscillating return model (predicts ±1% returns)")

        # -------------------
        # Backtest on validation slice
        # -------------------
        logging.info("Step 6: Running backtest on validation slice of df_clean...")

        # Ensure we have enough history to build a full window
        val_start_idx = split_idx + window_size
        val_end_idx = n_total
        if val_start_idx >= val_end_idx:
            raise ValueError(
                f"Not enough data for backtest: val_start_idx={val_start_idx}, "
                f"val_end_idx={val_end_idx}"
            )

        logging.info(
            f"Backtest period: index {val_start_idx} to {val_end_idx} "
            f"({val_end_idx - val_start_idx} timesteps)"
        )

        t0 = time.time()
        backtest_results = run_backtest(
            model=model,
            scaler=scaler,
            df=df_clean,
            feature_columns=feature_columns,
            window_size=window_size,
            forward_bars=forward_bars,
            initial_balance=initial_balance,
            transaction_cost_pct=transaction_cost,
            threshold=threshold,  # Now a return threshold (0.5%)
            start_idx=val_start_idx,
            end_idx=val_end_idx
        )
        logging.info(f"Backtest finished in {time.time() - t0:.2f}s")

        # -------------------
        # Step 7: Prepare visualization data
        # -------------------
        logging.info("Step 7: Preparing visualization data...")
        portfolio_history = backtest_results['portfolio_history']

        val_dates = portfolio_history['date'].values
        actual_prices = portfolio_history['current_price'].values
        predicted_prices = portfolio_history['predicted_price'].values
        predicted_returns = portfolio_history['predicted_return'].values

        signal_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        signals = portfolio_history['signal'].map(signal_map).values

        portfolio_values = portfolio_history['portfolio_value'].values

        indicators = {}
        for ind in ["SMA_Deviation", "RSI", "MACD_Histogram", "ATR", "Volume_Ratio"]:
            if ind in df_clean.columns:
                indicators[ind] = df_clean[ind].iloc[val_start_idx:val_end_idx].values

        # -------------------
        # Step 8: Save results
        # -------------------
        logging.info("Step 8: Saving backtest results...")
        results_summary = {
            'ticker': ticker,
            'window_size': window_size,
            'forward_bars': forward_bars,
            'epochs': epochs,
            'threshold': threshold,
            'threshold_type': 'return',
            'initial_balance': initial_balance,
            'transaction_cost': transaction_cost,
            'final_value': backtest_results['final_value'],
            'total_return': backtest_results['total_return'],
            'expected_return': backtest_results['expected_return'],
            'sharpe_ratio': backtest_results['sharpe_ratio'],
            'max_drawdown': backtest_results['max_drawdown'],
            'win_rate': backtest_results['win_rate'],
            'total_trades': backtest_results['total_trades'],
            'total_fees': backtest_results['total_fees'],
            'buy_hold_return': backtest_results['buy_hold_return'],
            'outperformance': backtest_results['outperformance'],
        }

        results_file = f'backtest_results_{ticker}_return_model.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logging.info(f"Results saved to {results_file}")

        if len(backtest_results['trades']) > 0:
            trades_file = f'backtest_trades_{ticker}_return_model.csv'
            backtest_results['trades'].to_csv(trades_file, index=False)
            logging.info(f"Trade history saved to {trades_file}")
        else:
            logging.warning("No trades executed during backtest!")

        # -------------------
        # Step 9: Print detailed analysis
        # -------------------
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}DUMMY MODEL BACKTEST ANALYSIS{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")
        print(f"Model type:             Oscillating return predictor")
        print(f"Prediction pattern:     ±{model.delta_return * 100:.1f}% returns (alternating)")
        print(f"Signal threshold:       ±{threshold * 100:.2f}% return")
        print(f"\nExpected behavior:")
        print(f"  - Model predicts +1% or -1% returns alternately")
        print(f"  - Threshold is 0.5%, so both +1% and -1% exceed threshold")
        print(f"  - Should generate BUY signals on +1% predictions")
        print(f"  - Should generate SELL signals on -1% predictions")
        print(f"  - Should create lots of trades (oscillating strategy)")
        print(f"\nActual results:")
        print(f"  - Total trades: {backtest_results['total_trades']}")
        print(f"  - Win rate: {backtest_results['win_rate']:.2f}%")
        print(f"  - Total return: {backtest_results['total_return']:.2f}%")
        print(f"  - Sharpe ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"\nNote: This is a DUMMY model for testing.")
        print(f"Real models should show much more sophisticated behavior")
        print(f"{BLUE}{'=' * 70}{RESET}\n")

        logging.info("Backtest-only pipeline finished successfully")