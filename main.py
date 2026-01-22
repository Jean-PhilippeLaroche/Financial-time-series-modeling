import argparse
import logging
import atexit

from utils.data_utils import (
    prepare_data_for_ai,
    filter_regular_hours_only,
    create_sequences_with_forward_returns,
    load_stock_csv,
    add_indicators,
    clean_data, load_stock_sqlite, add_time_features
)
from scripts.time_feature_engineering import get_recommended_time_features
from utils.plot_utils import visualize_model_performance
from scripts.train import train_model, get_tensorboard_writer, launch_tensorboard, DEVICE
from scripts.backtest import run_backtest
import time

from utils.model_interpretation import main_interpretation
import torch
import numpy as np

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------
# Main orchestration
# ---------------------------
def main(
        ticker="AAPL",
        window_size=120,
        train_size=0.8,
        epochs=80,
        batch_size=256,
        lr=3e-4,
        threshold=0.02,
        initial_balance=10000,
        transaction_cost=0.0015,
        forward_bars=5,
        visualize=True,
        # Transformer hyperparameters
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        patience=20,
        lr_scheduler_patience=6,
        lr_scheduler_factor=0.5,
        model_interpret=False,
        max_norm=0.1,
        use_gradient_clipping=True
):

    # Colors
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Console outputs
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}PIPELINE START - {ticker}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")

    print(f"Window:                 {window_size} | Epochs: {epochs} | Threshold: {threshold * 100}%")
    print(f"Model:                  Transformer (d_model: {d_model} | n_head: {nhead} | num_layers: {num_layers})")

    if model_interpret:
        print("Model will be interpreted automatically after backtesting")
    if use_gradient_clipping:
        print(f"Gradient clipping:      ENABLED")
        print(f"Max norm:               {max_norm}")
    else:
        print("Gradient clipping:      DISABLED")

    # ---- Launch TensorBoard automatically ----
    tb_process = launch_tensorboard(logdir="runs", port=6006)


    # Register cleanup for TensorBoard
    if tb_process:
        def cleanup_tensorboard():
            tb_process.terminate()
            logging.info("TensorBoard stopped.")

        atexit.register(cleanup_tensorboard)


    # ============================================================
    # STEP 1: Load temp df for correct split_idx and end_idx indexes
    # ============================================================
    t0 = time.time()
    print(f"\n{BLUE}{'-' * 70}{RESET}")
    print(f"{BLUE}[Step 1] Full Data Preparation{RESET}")

    # Using SQLite database if ticker is AAPL or MSFT
    if ticker in ("AAPL", "MSFT"):
        df_raw = load_stock_sqlite(ticker)
    else:
        df_raw = load_stock_csv(ticker)

    if df_raw is None:
        logging.error("Could not load raw CSV for ticker; exiting.")
        return

    df_filtered = filter_regular_hours_only(df_raw)
    df_with_indicators = add_indicators(df_filtered)
    df_with_time = add_time_features(df_with_indicators, minimal=True)
    df_clean = clean_data(df_with_time)

    n_total = len(df_clean)
    split_idx = int(n_total * train_size)
    print(f"Train/Val split:        {split_idx} / {n_total-split_idx}")

    # Split the dataframe
    train_df = df_clean.iloc[:split_idx].copy()
    val_df = df_clean.iloc[split_idx:].copy()

    # Step 1 timer
    print(f"Time:                   {time.time() - t0:.2f}s\n")

    # ============================================================
    # STEP 2: Prepare TRAIN sequences
    # ============================================================
    t0 = time.time()
    print(f"{BLUE}{'-' * 70}{RESET}")
    print(f"{BLUE}[Step 2] Preparing training sequences{RESET}")

    # Select features - "close" will be converted to "close_return" automatically
    feature_columns = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]

    time_feature_names = get_recommended_time_features(use_case='minimal')  # or 'standard', 'comprehensive'
    feature_columns.extend(time_feature_names)

    # Filter to only columns that actually exist
    feature_columns = [c for c in feature_columns if c in train_df.columns]

    print(f"Feature columns:        {len(feature_columns)} total "
          f"(6 technical + {len(time_feature_names)} time)")

    # Save raw close BEFORE scaling
    raw_close_train = train_df['close'].copy()

    # Scale features
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns])
    print("Scaler:                 created & fitted MinMaxScaler")

    train_df_scaled = train_df.copy()
    train_df_scaled[feature_columns] = scaler.transform(train_df[feature_columns])

    # Create sequences with forward returns
    X_train, y_train = create_sequences_with_forward_returns(
        train_df_scaled,
        feature_columns,
        target_column="close",
        window_size=window_size,
        forward_bars=5,
        raw_close=raw_close_train
    )

    if X_train is None or y_train is None:
        logging.error("Training data preparation failed. Exiting.")
        return

    print(f"Training sequences:     X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Target:                 5-bar forward returns")
    print(f"Return stats:           mean={y_train.mean():.6f}, std={y_train.std():.6f}")

    # Step 2 timer
    print(f"Time:                   {time.time() - t0:.2f}s\n")

    # ============================================================
    # STEP 3: Prepare VAL sequences
    # ============================================================

    t0 = time.time()
    print(f"{BLUE}{'-' * 70}{RESET}")
    print(f"{BLUE}[Step 3] Preparing validation sequences{RESET}")

    # Save raw close BEFORE scaling
    raw_close_val = val_df['close'].copy()

    # Use the SAME scaler
    val_df_scaled = val_df.copy()
    val_df_scaled[feature_columns] = scaler.transform(val_df[feature_columns])
    print("Scaler:                 using provided MinMaxScaler")

    # Create sequences with forward returns
    X_val, y_val = create_sequences_with_forward_returns(
        val_df_scaled,
        feature_columns,
        target_column="close",
        window_size=window_size,
        forward_bars=5,
        raw_close=raw_close_val
    )

    if X_val is None or y_val is None:
        logging.error("Validation data preparation failed. Exiting.")
        return

    print(f"Validation sequences:   X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"Return stats:           mean={y_val.mean():.6f}, std={y_val.std():.6f}")

    print(f"\n{BLUE}=== DATA SPLIT VERIFICATION ==={RESET}")
    print(f"Train samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")

    # Updated verification for RETURNS
    print(f"Train return range:     [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"Val return range:       [{y_val.min():.4f}, {y_val.max():.4f}]")
    print(f"Train return mean:      {y_train.mean():.6f} ({y_train.mean() * 100:.3f}%)")
    print(f"Val return mean:        {y_val.mean():.6f} ({y_val.mean() * 100:.3f}%)")
    print(f"Train return std:       {y_train.std():.6f} ({y_train.std() * 100:.3f}%)")
    print(f"Val return std:         {y_val.std():.6f} ({y_val.std() * 100:.3f}%)")

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Train targets (y_train):")
    print(f"  Mean:   {y_train.mean():.8f}")
    print(f"  Std:    {y_train.std():.8f}")
    print(f"  Min:    {y_train.min():.8f}")
    print(f"  Max:    {y_train.max():.8f}")
    print(f"  Median: {np.median(y_train):.8f}")
    print(f"\nPercentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  {p}th: {np.percentile(y_train, p):.8f}")
    print(f"\nTarget variance: {y_train.var():.10f}")
    print(f"Non-zero targets: {np.count_nonzero(y_train)} / {len(y_train)}")
    print(f"Unique values: {len(np.unique(y_train))}")
    print("=" * 70 + "\n")

    # Distribution shift check
    shift_pct = abs(y_train.mean() - y_val.mean()) / (y_train.std() + 1e-8) * 100
    print(f"Distribution shift:     {shift_pct:.1f}%")

    # Step 3 timer
    print(f"Time:                   {time.time() - t0:.2f}s")
    print(f"{BLUE}{'-' * 70}{RESET}\n")

    # ============================================================
    # STEP 4: Start TensorBoard writer and train
    # ============================================================
    t0 = time.time()
    writer = get_tensorboard_writer()

    print(f"\n{BLUE}{'-' * 70}{RESET}")
    print(f"{BLUE}[Step 4] Training Transformer model for {epochs} epochs{RESET}")

    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        writer=writer,
        scaler=scaler,
        # Transformer-specific parameters
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        early_stopping_patience=patience,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        max_norm=max_norm,
        use_gradient_clipping=use_gradient_clipping,
        feature_columns=feature_columns
    )

    if model is None:
        logging.error("Model training failed. Exiting.")
        return

    try:
        writer.close()
    except Exception as e:
        logging.warning(f"Failed to close TensorBoard writer: {e}")

    # Step 4 timer
    print(f"Total training time:   {time.time() - t0:.2f}s | {(time.time() - t0) / 60:.2f}m")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    # ============================================================
    # STEP 5: Backtest run on validation period
    #         Backtest DF = df_clean (recomputed to avoid leakages)
    #         Period = [split_idx : n_total]
    # ============================================================

    with torch.no_grad():
        sample_X = torch.tensor(X_val[:1000], dtype=torch.float32).to(DEVICE)
        sample_preds = model(sample_X)[0].cpu().numpy()

    print(f"\nModel Prediction Statistics:")
    print(f"  Mean: {sample_preds.mean():.6f}")
    print(f"  Std:  {sample_preds.std():.6f}")
    print(f"  Min:  {sample_preds.min():.6f}")
    print(f"  Max:  {sample_preds.max():.6f}")
    print(f"  |pred| > 0.005: {(np.abs(sample_preds) > 0.005).sum()} / 1000")
    print(f"  |pred| > 0.002: {(np.abs(sample_preds) > 0.002).sum()} / 1000")

    t0 = time.time()
    print(f"{BLUE}{'-' * 70}{RESET}")
    print(f"{BLUE}[Step 5] Running backtest on validation slice of the dataframe{RESET}")

    df_clean = add_indicators(df_raw)
    df_clean = clean_data(df_clean)
    df_clean = filter_regular_hours_only(df_clean)

    feature_columns = ["close", "RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]

    time_feature_names = get_recommended_time_features(use_case='minimal')  # or 'standard', 'comprehensive'
    feature_columns.extend(time_feature_names)

    # Filter to only columns that actually exist
    feature_columns = [c for c in feature_columns if c in train_df.columns]
    print(f"Using feature columns:  {feature_columns}")

    val_start_idx = split_idx + window_size
    val_end_idx = n_total

    backtest_results = run_backtest(
        model=model,
        scaler=scaler,
        df=df_clean,
        feature_columns=feature_columns,
        window_size=window_size,
        forward_bars=forward_bars,
        initial_balance=initial_balance,
        transaction_cost_pct=transaction_cost,
        threshold=threshold,
        start_idx=val_start_idx,
        end_idx=val_end_idx
    )

    # Step 5 timer (not displayed because obsolete, custom timers in backtesting.py
    # logging.info(f"\033[91mStep 5: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 6: Prepare and generate visualization data
    # ============================================================
    if visualize:
        t0 = time.time()
        print(f"{BLUE}{'-' * 70}{RESET}")
        print(f"{BLUE}[Step 6] Generating visualizations{RESET}")

        try:
            portfolio_history = backtest_results['portfolio_history']

            signal_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}

            indicators = {}
            for ind in ["RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"]:
                if ind in df_clean.columns:
                    indicators[ind] = df_clean[ind].iloc[val_start_idx:val_end_idx].values

            visualize_model_performance(
                dates=portfolio_history['date'].values,
                actual_prices=portfolio_history['current_price'].values,
                predicted_returns=portfolio_history['predicted_return'].values,
                signals=portfolio_history['signal'].map(signal_map).values,
                portfolio_values=portfolio_history['portfolio_value'].values,
                indicators=indicators if indicators else None,
                ticker=ticker
            )
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            logging.exception("Full traceback:")

        print(f"Time: {time.time() - t0:.2f}s\n")

    # ============================================================
    # STEP 7: Save backtest results
    # ============================================================
    print(f"{BLUE}{'-' * 70}{RESET}")
    print(f"{BLUE}[Step 7] Saving backtest results{RESET}")
    results_summary = {
        'ticker': ticker,
        'window_size': window_size,
        'forward_bars': forward_bars,
        'epochs': epochs,
        'threshold': threshold,
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
        'outperformance': backtest_results['outperformance']
    }

    import json
    results_file = f'backtest_results_{ticker}.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to {results_file}")

    if len(backtest_results['trades']) > 0:
        trades_file = f'backtest_trades_{ticker}.csv'
        backtest_results['trades'].to_csv(trades_file, index=False)
        print(f"Trade history saved to {trades_file}\n")

    # ========================================================
    # STEP 8: model interpretation if selected
    # ============================================================
    if model_interpret:
        main_interpretation(ticker, train_size, window_size,
                            dim_feedforward, d_model, nhead, num_layers,
                            file="csv", input_size=X_train.shape[2])

    # ============================================================
    # STEP 9: Summary
    # ============================================================

    print(f"{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}PIPELINE SUMMARY - {ticker}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")

    print(f"Ticker:                  {ticker}")
    print(f"Training epochs:         {epochs}")
    print(f"Validation period:       {val_end_idx - val_start_idx} predictions")
    print(f"Prediction horizon:      {forward_bars} bars ahead")
    print(f"Target:                  {forward_bars}-bar forward returns")
    print(f"{BLUE}{'-' * 70}{RESET}")

    print(f"Initial Balance:         ${initial_balance:,.2f}")
    print(f"Final Portfolio Value:   ${backtest_results['final_value']:,.2f}")
    print(f"Total Return:            {backtest_results['total_return']:>6.2f}%")
    print(f"Expected Return (daily): {backtest_results['expected_return']:>6.4f}%")
    print(f"Buy & Hold Return:       {backtest_results['buy_hold_return']:>6.2f}%")
    print(f"Outperformance:          {backtest_results['outperformance']:>6.2f}%")
    print(f"{BLUE}{'-' * 70}{RESET}")

    print(f"Sharpe Ratio:            {backtest_results['sharpe_ratio']:>6.3f}")
    print(f"Max Drawdown:            {backtest_results['max_drawdown']:>6.2f}%")
    print(f"Win Rate:                {backtest_results['win_rate']:>6.2f}%")
    print(f"{BLUE}{'-' * 70}{RESET}")

    print(f"Total Trades:            {backtest_results['total_trades']}")
    print(f"Total Fees Paid:         ${backtest_results['total_fees']:,.2f}")
    print(f"Transaction Cost:        {transaction_cost * 100}%")
    print(f"Signal Threshold:        {threshold * 100:.2f}% return")
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Samples per parameter: {len(X_train) / sum(p.numel() for p in model.parameters()):.1f}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    print(f"{GREEN}Pipeline finished. Press CTRL+C to exit.{RESET}")

    # Keeping everything alive until manual stop
    from threading import Event
    Event().wait()


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Financial time series modeling AI - Train Transformer model and backtest strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument("--ticker", type=str, default="MSFT",
                        help="Stock ticker symbol")
    parser.add_argument("--window", type=int, default=120,
                        help="Window size for sequences")
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="Fraction of data for training (rest for validation)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler_patience", type=int, default=6,
                        help="Learning rate reduced after fixed number of epoch without improvement")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5,
                        help="Learning rate")

    # Transformer hyperparameters
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=1024,
                        help="Feedforward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Trading parameters
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Relative threshold for buy/sell signals (e.g., 0.02 = 2%)")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Initial balance for backtesting")
    parser.add_argument("--transaction_cost", type=float, default=0.0015,
                        help="Transaction cost as fraction (e.g., 0.02 = 2%%)")
    parser.add_argument("--forward_bars", type=int, default=5,
                        help="Prediction horizon (e.g., 5 = predict 5 minutes ahead)")

    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping after fixed number of epoch without improvement")

    # Gradient clipping parameters
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Max norm for gradient clipping (ex. 1.0)")
    parser.add_argument("--no_gradient_clipping", action="store_true",
                        help="Disable gradient clipping")

    # Visualization
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable plotting at end")
    parser.add_argument("--model_interpretation", action="store_true",
                        help="Enable model interpretation")

    args = parser.parse_args()

    main(
        ticker=args.ticker,
        window_size=args.window,
        train_size=args.train_size,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        threshold=args.threshold,
        initial_balance=args.balance,
        transaction_cost=args.transaction_cost,
        forward_bars=args.forward_bars,
        visualize=not args.no_viz,
        model_interpret=args.model_interpretation,
        # Transformer hyperparameters
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        patience=args.patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        max_norm =args.max_norm,
        use_gradient_clipping =not args.no_gradient_clipping
    )