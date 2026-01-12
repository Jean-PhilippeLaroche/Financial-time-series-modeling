import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_price_vs_prediction(dates, actual, predicted, title="Price vs Prediction"):
    """
    Plot actual stock prices vs model predictions.

    Args:
        dates (array-like): list/array of dates (x-axis)
        actual (array-like): actual stock prices
        predicted (array-like): predicted stock prices
        title (str): chart title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual", color="blue", linewidth=2)
    plt.plot(dates, predicted, label="Predicted", color="orange", linestyle="--", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    print("Plotted actual stock prices vs model predictions")


def plot_signals(dates, prices, signals, title="Trading Signals"):
    """
    Plot stock prices with buy/sell signals.

    Args:
        dates (array-like): list/array of dates
        prices (array-like): stock prices
        signals (array-like): 1 for Buy, -1 for Sell, 0 for Hold
        title (str): chart title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label="Price", color="blue", linewidth=2)

    # Buy signals
    buy_signals = np.where(signals == 1)[0]
    plt.scatter(
        np.array(dates)[buy_signals],
        np.array(prices)[buy_signals],
        label="Buy",
        marker="^",
        color="green",
        s=100
    )

    # Sell signals
    sell_signals = np.where(signals == -1)[0]
    plt.scatter(
        np.array(dates)[sell_signals],
        np.array(prices)[sell_signals],
        label="Sell",
        marker="v",
        color="red",
        s=100
    )

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    print("Plotted trading signals")


def plot_portfolio_equity(portfolio_values, dates=None):
    """
    Plot the portfolio equity curve over time.

    Args:
        portfolio_values (list or np.array): portfolio value at each timestep
        dates (list or np.array): optional dates for x-axis
    """
    plt.figure(figsize=(12, 6))
    if dates is None:
        dates = np.arange(len(portfolio_values))
    plt.plot(dates, portfolio_values, label="Portfolio Equity", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    print("Plotted portfolio equity curve")


def plot_indicator_overlay(prices, indicators_dict, dates=None):
    """
    Plot stock prices with multiple indicators overlaid.

    Args:
        prices (list or np.array): stock prices
        indicators_dict (dict): key=name of indicator, value=np.array of indicator values
        dates (list or np.array): optional dates for x-axis
    """
    plt.figure(figsize=(6, 4))
    if dates is None:
        dates = np.arange(len(prices))
    plt.plot(dates, prices, label="Price", color="black")
    for name, values in indicators_dict.items():
        plt.plot(dates, values, label=name)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Price with Indicator Overlays")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    print("Plotted indicator overlays")


def plot_return_distribution(returns, bins=50):
    """
    Plot histogram of daily returns.

    Args:
        returns (list or np.array): daily returns, can be log or simple returns
        bins (int): number of bins in histogram
    """
    plt.figure(figsize=(6, 4))
    plt.hist(returns, bins=bins, color="skyblue", edgecolor="black")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.title("Histogram of Daily Returns")
    plt.grid(True)
    plt.tight_layout()

    print("Plotted histogram of daily returns")


# ------------------
# CLI
# ------------------
if __name__ == "__main__":

    # Generate fake data for testing
    rng = pd.date_range("2023-01-01", periods=50, freq="D")
    actual_prices = np.linspace(100, 120, 50) + np.random.normal(0, 1, 50)
    predicted_prices = actual_prices + np.random.normal(0, 2, 50)

    # Create fake trading signals
    signals = np.zeros(50)
    signals[10] = 1   # Buy at day 10
    signals[25] = -1  # Sell at day 25
    signals[40] = 1   # Buy at day 40

    # Test first 2 functions
    plot_signals(rng, actual_prices, signals)
    plot_price_vs_prediction(rng, actual_prices, predicted_prices)

    # Portfolio equity curve
    portfolio_values = np.cumsum(np.random.randn(50)) + 1000
    plot_portfolio_equity(portfolio_values)

    # Indicator overlay
    sma = np.convolve(actual_prices, np.ones(5)/5, mode='valid')
    rsi = np.random.rand(50) * 100
    indicators = {
        "SMA": np.pad(sma, (len(actual_prices)-len(sma), 0), 'constant', constant_values=np.nan),
        "RSI": rsi
    }
    plot_indicator_overlay(actual_prices, indicators)

    # Return distribution
    returns = np.diff(actual_prices) / actual_prices[:-1]
    plot_return_distribution(returns)

    # Show everything at once
    plt.show()