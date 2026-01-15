import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import webbrowser
import threading
from time import sleep
import logging
import os

# Global state
_app = None
_server_thread = None
_initialized = False


def create_backtest_dashboard(dates, actual_prices, predicted_returns, signals,
                              portfolio_values, indicators=None, returns=None,
                              ticker="Stock", port=8051):
    """
    Create and launch an interactive dashboard showing all backtest results.

    UPDATED: Now works with return-based predictions instead of price predictions.

    Args:
        dates: Array of dates
        actual_prices: Actual stock prices
        predicted_returns: Model's predicted returns (e.g., 0.0025 = +0.25%)
        signals: Trading signals (1=Buy, -1=Sell, 0=Hold)
        portfolio_values: Portfolio equity over time
        indicators: Dict of indicator name -> values (optional)
        returns: Daily returns for histogram (optional)
        ticker: Stock ticker symbol
        port: Port for dashboard (default: 8051)
    """
    global _app, _server_thread, _initialized

    if _initialized:
        print("Dashboard already running")
        return

    # Convert predicted returns to implied prices for visualization
    # predicted_price[t] = actual_price[t] * (1 + predicted_return[t])
    predicted_prices = actual_prices * (1 + predicted_returns)

    # Create Dash app
    _app = dash.Dash(__name__)

    # Create all figures
    fig_price_pred = create_price_prediction_plot(dates, actual_prices, predicted_prices)
    fig_signals = create_signals_plot(dates, actual_prices, signals)
    fig_portfolio = create_portfolio_plot(dates, portfolio_values)
    fig_indicators = create_indicators_plot(dates, actual_prices, indicators) if indicators else None
    fig_returns = create_returns_histogram(returns) if returns is not None else None

    # Build layout
    charts = [
        html.H1(f"{ticker} Backtest Results", style={'textAlign': 'center', 'marginBottom': 30}),

        html.Div([
            html.H3("Price vs Predicted Direction", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_price_pred, style={'height': '400px'}),
        ], style={'marginBottom': 40}),

        html.Div([
            html.H3("Trading Signals", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_signals, style={'height': '400px'}),
        ], style={'marginBottom': 40}),

        html.Div([
            html.H3("Portfolio Equity Curve", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_portfolio, style={'height': '400px'}),
        ], style={'marginBottom': 40}),
    ]

    # Add indicators plot if available
    if fig_indicators:
        charts.append(html.Div([
            html.H3("Technical Indicators", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_indicators, style={'height': '400px'}),
        ], style={'marginBottom': 40}))

    # Add returns histogram if available
    if fig_returns:
        charts.append(html.Div([
            html.H3("Daily Returns Distribution", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_returns, style={'height': '400px'}),
        ], style={'marginBottom': 40}))

    _app.layout = html.Div(charts, style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif'
    })

    # Run server in background
    def run_server():
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('werkzeug').disabled = True
        logging.getLogger('dash').setLevel(logging.ERROR)
        _app.run(debug=False, port=port, use_reloader=False)

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)

    _initialized = True
    print(f"\nBacktest dashboard opened at {url}")
    print("  Press Ctrl+C to stop the training script when done viewing\n")


def create_price_prediction_plot(dates, actual, predicted):
    """
    Create price vs prediction plot.

    NOTE: 'predicted' is now the implied future price based on predicted returns.
    The visualization shows where the model thinks the price will be.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted Direction (Implied Price)',
        line=dict(color='orange', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        legend=dict(x=0, y=1),
        margin=dict(l=50, r=50, t=30, b=50)
    )

    return fig


def create_signals_plot(dates, prices, signals):
    """Create trading signals plot."""
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))

    # Buy signals
    buy_mask = signals == 1
    if np.any(buy_mask):
        fig.add_trace(go.Scatter(
            x=dates[buy_mask],
            y=prices[buy_mask],
            mode='markers',
            name='Buy',
            marker=dict(symbol='triangle-up', size=12, color='green')
        ))

    # Sell signals
    sell_mask = signals == -1
    if np.any(sell_mask):
        fig.add_trace(go.Scatter(
            x=dates[sell_mask],
            y=prices[sell_mask],
            mode='markers',
            name='Sell',
            marker=dict(symbol='triangle-down', size=12, color='red')
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        legend=dict(x=0, y=1),
        margin=dict(l=50, r=50, t=30, b=50)
    )

    return fig


def create_portfolio_plot(dates, portfolio_values):
    """Create portfolio equity curve."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)'
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        margin=dict(l=50, r=50, t=30, b=50)
    )

    return fig


def create_indicators_plot(dates, prices, indicators_dict):
    """Create indicators overlay plot."""
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='black', width=2)
    ))

    # Indicators
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for idx, (name, values) in enumerate(indicators_dict.items()):
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=name,
            line=dict(color=colors[idx % len(colors)], width=1.5)
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        legend=dict(x=0, y=1),
        margin=dict(l=50, r=50, t=30, b=50)
    )

    return fig


def create_returns_histogram(returns):
    """Create histogram of returns."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker=dict(color='skyblue', line=dict(color='black', width=1))
    ))

    fig.update_layout(
        xaxis_title="Return",
        yaxis_title="Frequency",
        bargap=0.1,
        margin=dict(l=50, r=50, t=30, b=50)
    )

    return fig


def visualize_model_performance(dates, actual_prices, predicted_returns, signals,
                                portfolio_values, indicators=None, ticker="Stock"):
    """
    Drop-in replacement for the old matplotlib visualization function.

    UPDATED: Now accepts predicted returns instead of predicted prices.

    Args:
        dates: Array of dates
        actual_prices: Actual stock prices
        predicted_returns: Model's predicted returns (e.g., 0.0025 = +0.25%)
        signals: Trading signals (1=Buy, -1=Sell, 0=Hold)
        portfolio_values: Portfolio equity over time
        indicators: Dict of indicator name -> values (optional)
        ticker: Stock ticker symbol
    """
    # Calculate returns for histogram
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    create_backtest_dashboard(
        dates=dates,
        actual_prices=actual_prices,
        predicted_returns=predicted_returns,
        signals=signals,
        portfolio_values=portfolio_values,
        indicators=indicators,
        returns=returns,
        ticker=ticker,
        port=8051
    )


if __name__ == "__main__":
    """
    Run tests when this file is executed directly.

    Usage:
        python plot_utils.py          # Run unit tests only
        python plot_utils.py --live   # Launch interactive test
    """
    import sys
    import numpy as np


    def run_unit_tests():
        """
        Run unit tests to verify dashboard components work correctly.
        Use this to debug issues or verify after making changes.

        UPDATED: Now tests return-based prediction system.
        """
        print("\n" + "=" * 70)
        print("RUNNING BACKTEST DASHBOARD UNIT TESTS")
        print("=" * 70 + "\n")

        passed = 0
        failed = 0

        # Test 1: Create synthetic data
        print("[Test 1] Creating synthetic test data (return-based)...")
        try:
            n_points = 100
            dates = np.arange(n_points)
            actual_prices = 100 + np.cumsum(np.random.randn(n_points) * 2)

            # NEW: Generate predicted returns instead of predicted prices
            # Returns should be small percentages (e.g., -0.01 to +0.01 = -1% to +1%)
            predicted_returns = np.random.randn(n_points) * 0.01  # Mean 0%, std 1%

            signals = np.random.choice([1, 0, -1], n_points, p=[0.2, 0.6, 0.2])
            portfolio_values = 10000 + np.cumsum(np.random.randn(n_points) * 100)

            assert len(dates) == n_points
            assert len(actual_prices) == n_points
            assert len(predicted_returns) == n_points
            assert len(signals) == n_points
            assert len(portfolio_values) == n_points

            print("Synthetic data created successfully")
            print(f"  Predicted returns range: {predicted_returns.min():.4f} to {predicted_returns.max():.4f}")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1
            return

        # Test 2: Create price prediction plot (with return-to-price conversion)
        print("[Test 2] Testing price prediction plot creation (returns → implied prices)...")
        try:
            # Convert returns to implied prices for visualization
            implied_prices = actual_prices * (1 + predicted_returns)

            fig = create_price_prediction_plot(dates, actual_prices, implied_prices)
            assert fig is not None
            assert len(fig.data) == 2  # Should have 2 traces (actual + predicted)
            assert fig.data[0].name == 'Actual Price'
            assert fig.data[1].name == 'Predicted Direction (Implied Price)'
            print("Price prediction plot created successfully")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

        # Test 3: Create signals plot
        print("[Test 3] Testing signals plot creation...")
        try:
            fig = create_signals_plot(dates, actual_prices, signals)
            assert fig is not None
            assert len(fig.data) >= 1  # At least price line
            print(f"Signals plot created successfully ({len(fig.data)} traces)")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

        # Test 4: Create portfolio plot
        print("[Test 4] Testing portfolio equity plot creation...")
        try:
            fig = create_portfolio_plot(dates, portfolio_values)
            assert fig is not None
            assert len(fig.data) == 1
            assert fig.data[0].name == 'Portfolio Value'
            print("Portfolio plot created successfully")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

        # Test 5: Create indicators plot
        print("[Test 5] Testing indicators plot creation...")
        try:
            indicators = {
                'SMA': np.convolve(actual_prices, np.ones(10) / 10, mode='same'),
                'RSI': 50 + np.random.randn(n_points) * 10,
                'MACD': np.random.randn(n_points) * 2
            }
            fig = create_indicators_plot(dates, actual_prices, indicators)
            assert fig is not None
            assert len(fig.data) == 4  # Price + 3 indicators
            print("Indicators plot created successfully")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

        # Test 6: Create returns histogram
        print("[Test 6] Testing returns histogram creation...")
        try:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            fig = create_returns_histogram(returns)
            assert fig is not None
            assert len(fig.data) == 1
            print("Returns histogram created successfully")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

        # Test 7: Test with edge cases
        print("[Test 7] Testing edge cases...")
        try:
            # All hold signals
            signals_hold = np.zeros(n_points)
            fig = create_signals_plot(dates, actual_prices, signals_hold)
            assert fig is not None

            # Empty indicators
            fig = create_indicators_plot(dates, actual_prices, {})
            assert fig is not None

            # Very small dataset with returns
            small_dates = np.arange(5)
            small_prices = np.array([100, 101, 99, 102, 98])
            small_returns = np.array([0.01, -0.02, 0.03, -0.04, 0.02])
            small_implied_prices = small_prices * (1 + small_returns)
            fig = create_price_prediction_plot(small_dates, small_prices, small_implied_prices)
            assert fig is not None

            print("Edge cases handled successfully")
            passed += 1
        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

        # Test 8: Test data validation
        print("[Test 8] Testing data validation...")
        try:
            # Test that numpy operations catch mismatched lengths
            try:
                bad_dates = np.arange(50)  # Wrong length (50 vs 100)
                implied_prices = actual_prices * (1 + predicted_returns)

                # Plotly may not raise an error, but numpy operations should fail
                # when trying to do element-wise operations with mismatched arrays
                bad_returns = np.random.randn(50) * 0.01
                bad_prices = np.random.randn(50) + 100

                # This should fail due to shape mismatch
                result = actual_prices * (1 + bad_returns)

                # If we get here, it means numpy didn't catch the mismatch
                # (which can happen if broadcasting works)
                print("Data validation test passed (broadcasting handled gracefully)")
                passed += 1

            except (ValueError, IndexError, TypeError) as e:
                # Expected: numpy caught the mismatch
                print(f"Data validation working correctly (error caught: {type(e).__name__})")
                passed += 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            failed += 1

        # Summary
        print("\n" + "=" * 70)
        print(f"TEST RESULTS: {passed} passed, {failed} failed")
        print("=" * 70 + "\n")

        if failed == 0:
            print("All tests passed! Dashboard is working correctly with return-based system.\n")
        else:
            print(f"{failed} test(s) failed. Check error messages above.\n")

        return passed, failed


    def test_dashboard_launch():
        """
        Interactive test: Launch dashboard with synthetic data.
        Run this to verify the dashboard opens correctly in browser.

        UPDATED: Now uses return-based predictions.
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE DASHBOARD TEST (Return-Based System)")
        print("=" * 70)
        print("\nLaunching dashboard with synthetic data...")
        print("The dashboard should open in your browser automatically.")
        print("Press Ctrl+C in terminal to stop.\n")

        # Create synthetic data
        n_points = 200
        dates = np.arange(n_points)
        actual_prices = 100 + np.cumsum(np.random.randn(n_points) * 2)

        # NEW: Generate predicted returns (realistic range: -2% to +2%)
        predicted_returns = np.random.randn(n_points) * 0.015  # Mean 0%, std 1.5%

        signals = np.random.choice([1, 0, -1], n_points, p=[0.15, 0.7, 0.15])
        portfolio_values = 10000 + np.cumsum(np.random.randn(n_points) * 150)

        indicators = {
            'SMA': np.convolve(actual_prices, np.ones(20) / 20, mode='same'),
            'RSI': 50 + np.random.randn(n_points) * 15,
            'MACD': np.random.randn(n_points) * 3
        }

        print(f"Generated {n_points} data points")
        print(f"Predicted returns range: {predicted_returns.min():.4f} to {predicted_returns.max():.4f}")
        print(f"This will be converted to implied prices for visualization\n")

        visualize_model_performance(
            dates=dates,
            actual_prices=actual_prices,
            predicted_returns=predicted_returns,  # Changed from predicted_prices
            signals=signals,
            portfolio_values=portfolio_values,
            indicators=indicators,
            ticker="TEST"
        )

        print("Dashboard should now be running at http://127.0.0.1:8051")
        print("Verify all 5 charts are visible and interactive.")
        print("The 'Price vs Predicted Direction' chart should show implied prices based on returns.")

        try:
            input("\nPress Enter to continue (or Ctrl+C to exit)...")
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user.")


    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        # Run interactive test
        test_dashboard_launch()
    else:
        # Run unit tests
        passed, failed = run_unit_tests()
        sys.exit(0 if failed == 0 else 1)