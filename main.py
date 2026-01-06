import argparse
import logging
import atexit

from utils.data_utils import (
    prepare_data_for_ai,
    load_stock_csv,
    add_indicators,
    clean_data, load_stock_sqlite
)
from scripts.train import train_model, get_tensorboard_writer, launch_tensorboard
from scripts.evaluate import visualize_model_performance
from scripts.backtest import run_backtest
import time

from utils.model_interpretation import main_interpretation

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
        grad_clip_percentile=95,
        use_adaptive_clipping=True
):


    logging.info(f"Starting pipeline for {ticker}")
    logging.info(f"Configuration: window={window_size}, epochs={epochs}, threshold={threshold * 100}%")
    logging.info(f"Transformer config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    if model_interpret:
        logging.info("Model will be interpreted automatically after backtesting")
    if use_adaptive_clipping:
        logging.info("Using adaptive clipping")

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
    logging.info("Step 1: Loading and preparing full dataframe")

    # Using SQLite database if ticker is AAPL or MSFT
    if ticker in ("AAPL", "MSFT"):
        df_raw = load_stock_sqlite(ticker)
    else:
        df_raw = load_stock_csv(ticker)

    if df_raw is None:
        logging.error("Could not load raw CSV for ticker; exiting.")
        return

    df_tmp = add_indicators(df_raw)
    df_tmp = clean_data(df_tmp)

    n_total = len(df_tmp)
    split_idx = int(n_total * train_size)
    logging.info(f"Total cleaned rows: {n_total}, train/val split index: {split_idx}")

    # Step 1 timer
    logging.info(f"\033[91mStep 1: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 2-3: Prepare TRAIN and VAL sequences using prepare_data_for_ai
    #         Train: [0 : split_idx]
    #         Val:   [split_idx : n_total]
    # ============================================================
    t0 = time.time()
    logging.info("Step 2: Preparing training sequences")

    # Using SQLite database if ticker is AAPL or MSFT
    using_sqlite = ticker in ("AAPL", "MSFT")

    X_train, y_train, scaler = prepare_data_for_ai(
        ticker,
        data_dir=None,
        feature_columns=None,
        SQLite=using_sqlite,
        target_column="close",
        window_size=window_size,
        start_idx=0,
        end_idx=split_idx,
        # scaler=None
    )
    if X_train is None or y_train is None:
        logging.error("Training data preparation failed. Exiting.")
        return

    logging.info(f"Training sequences: X_train={X_train.shape}, y_train={y_train.shape}")

    # Step 2 timer
    logging.info(f"\033[91mStep 2: {time.time() - t0}\n\033[0m")

    t0 = time.time()
    logging.info("Step 3: Preparing validation sequences")
    X_val, y_val, _ = prepare_data_for_ai(
        ticker,
        data_dir=None,
        feature_columns=None,
        SQLite=using_sqlite,
        target_column="close",
        window_size=window_size,
        start_idx=split_idx,
        end_idx=None,
        scaler=scaler
    )
    if X_val is None or y_val is None:
        logging.error("Validation data preparation failed. Exiting.")
        return

    logging.info(f"Validation sequences: X_val={X_val.shape}, y_val={y_val.shape}")

    # Step 3 timer
    logging.info(f"\033[91mStep 3: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 4: Start TensorBoard writer and train
    # ============================================================
    t0 = time.time()
    writer = get_tensorboard_writer()

    logging.info(f"Step 4: Training Transformer model for {epochs} epochs")
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
        grad_clip_percentile=grad_clip_percentile,
        use_adaptive_clipping=use_adaptive_clipping
    )

    if model is None:
        logging.error("Model training failed. Exiting.")
        return

    try:
        writer.close()
    except Exception as e:
        logging.warning(f"Failed to close TensorBoard writer: {e}")

    # Step 4 timer
    logging.info(f"\033[91mStep 4: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 5: Backtest run on validation period
    #         Backtest DF = df_clean (recomputed to avoid leakages)
    #         Period = [split_idx : n_total]
    # ============================================================
    t0 = time.time()
    logging.info("Step 5: Running backtest on validation slice of the cleaned dataframe")

    df_clean = add_indicators(df_raw)
    df_clean = clean_data(df_clean)

    feature_columns = ["close", "volume", "RSI", "MACD", "MACD_Signal", "SMA"]
    feature_columns = [c for c in feature_columns if c in df_clean.columns]
    logging.info(f"Using feature columns: {feature_columns}")

    val_start_idx = split_idx + window_size
    val_end_idx = n_total
    logging.info(f"Backtest period: index {val_start_idx} to {val_end_idx} ({val_end_idx - val_start_idx} days)")

    backtest_results = run_backtest(
        model=model,
        scaler=scaler,
        df=df_clean,
        feature_columns=feature_columns,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost_pct=transaction_cost,
        threshold=threshold,
        start_idx=val_start_idx,
        end_idx=val_end_idx
    )

    # Step 5 timer
    logging.info(f"\033[91mStep 5: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 6: Prepare visualization data
    # ============================================================
    t0 = time.time()
    logging.info("Step 6: Preparing visualization data")
    portfolio_history = backtest_results['portfolio_history']

    val_dates = portfolio_history['date'].values
    actual_prices = portfolio_history['current_price'].values
    predicted_prices = portfolio_history['predicted_price'].values

    signal_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    signals = portfolio_history['signal'].map(signal_map).values

    portfolio_values = portfolio_history['portfolio_value'].values

    indicators = {}
    for ind in ["SMA", "RSI", "MACD", "MACD_Signal"]:
        if ind in df_clean.columns:
            indicators[ind] = df_clean[ind].iloc[val_start_idx:val_end_idx].values

    # Step 6 timer
    logging.info(f"\033[91mStep 6: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 7: Save backtest results
    # ============================================================
    t0 = time.time()
    logging.info("Step 7: Saving backtest results")
    results_summary = {
        'ticker': ticker,
        'window_size': window_size,
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
    logging.info(f"Results saved to {results_file}")

    if len(backtest_results['trades']) > 0:
        trades_file = f'backtest_trades_{ticker}.csv'
        backtest_results['trades'].to_csv(trades_file, index=False)
        logging.info(f"Trade history saved to {trades_file}")

    # Step 7 timer
    logging.info(f"\033[91mStep 7: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 8: Visualization
    # ============================================================
    t0 = time.time()
    if visualize:
        logging.info("Step 8: Generating visualizations")
        try:
            visualize_model_performance(
                dates=val_dates,
                actual_prices=actual_prices,
                predicted_prices=predicted_prices,
                signals=signals,
                portfolio_values=portfolio_values,
                indicators=indicators if indicators else None
            )
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            logging.exception("Full traceback:")

    # Step 8 timer
    logging.info(f"\033[91mStep 8: {time.time() - t0}\n\033[0m")

    # ============================================================
    # STEP 9: Summary + model interpretation if selected
    # ============================================================
    if model_interpret:
        main_interpretation(ticker, train_size, window_size,
                            dim_feedforward, d_model, nhead, num_layers,
                            file="csv", input_size=X_train.shape[2])

    logging.info("\n" + "=" * 70)
    logging.info("PIPELINE COMPLETE - SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Ticker: {ticker}")
    logging.info(f"Training epochs: {epochs}")
    logging.info(f"Validation period: {val_end_idx - val_start_idx} days")
    logging.info(f"-" * 70)
    logging.info(f"Initial Balance:        ${initial_balance:,.2f}")
    logging.info(f"Final Portfolio Value:  ${backtest_results['final_value']:,.2f}")
    logging.info(f"Total Return:           {backtest_results['total_return']:>6.2f}%")
    logging.info(f"Expected Return (daily):{backtest_results['expected_return']:>6.4f}%")
    logging.info(f"Buy & Hold Return:      {backtest_results['buy_hold_return']:>6.2f}%")
    logging.info(f"Outperformance:         {backtest_results['outperformance']:>6.2f}%")
    logging.info(f"-" * 70)
    logging.info(f"Sharpe Ratio:           {backtest_results['sharpe_ratio']:>6.3f}")
    logging.info(f"Max Drawdown:           {backtest_results['max_drawdown']:>6.2f}%")
    logging.info(f"Win Rate:               {backtest_results['win_rate']:>6.2f}%")
    logging.info(f"-" * 70)
    logging.info(f"Total Trades:           {backtest_results['total_trades']}")
    logging.info(f"Total Fees Paid:        ${backtest_results['total_fees']:,.2f}")
    logging.info(f"Transaction Cost:       {transaction_cost * 100}%")
    logging.info("=" * 70 + "\n")

    logging.info("Pipeline finished")


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
                        help="Relative threshold for buy/sell signals (e.g., 0.02 = 2%%)")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Initial balance for backtesting")
    parser.add_argument("--transaction_cost", type=float, default=0.0015,
                        help="Transaction cost as fraction (e.g., 0.02 = 2%%)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping after fixed number of epoch without improvement")

    # Gradient clipping parameters
    parser.add_argument("--grad_clip_percentile", type=float, default=95,
                        help="Percentile for adaptive gradient clipping (ex. 95)")
    parser.add_argument("--no_adaptive_clipping", action="store_true",
                        help="Disable adaptive gradient clipping")

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
        grad_clip_percentile =args.grad_clip_percentile,
        use_adaptive_clipping =not args.no_adaptive_clipping
    )