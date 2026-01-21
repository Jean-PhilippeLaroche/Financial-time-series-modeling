"""
Prediction Visualization During Training

This module provides functions to visualize how the model's predictions evolve
during training, displaying them as percentage returns on an adaptive scale.

Features:
- Real-time Dash dashboard with auto-refresh every 5000ms
- Visualizes predictions vs actual values per epoch
- Adaptive y-axis scaling based on return magnitudes
- Shows prediction distribution evolution
- Interactive plots with zoom/pan capabilities
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging
import json
import threading
from pathlib import Path
import webbrowser
import time

# Dash imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px

# Global state for the dashboard
DASHBOARD_DATA = {
    'current_epoch': 0,
    'pred_returns': [],
    'actuals': [],
    'stats': {},
    'epoch_history': []
}
DASHBOARD_LOCK = threading.Lock()


def calculate_returns(predictions: np.ndarray, actuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction errors as percentage differences.

    For normalized data (0-1 scale), converts absolute differences to percentages.
    For price data, calculates percentage returns.

    Args:
        predictions: Model predictions (raw prices or normalized values)
        actuals: Actual target values (raw prices or normalized values)

    Returns:
        Tuple of (predicted_errors, actual_errors) as percentages
    """
    # Check if data is normalized (approximately between 0 and 1)
    is_normalized = (np.max(np.abs(actuals)) <= 2.0) and (np.min(actuals) >= -2.0)

    if is_normalized:
        # For normalized data: calculate absolute difference and scale to percentage
        # A difference of 0.01 in normalized space = 1% error
        pred_errors = (predictions - actuals) * 100
    else:
        # For price data: calculate percentage returns
        actuals_safe = np.where(actuals == 0, 1e-8, actuals)
        pred_errors = ((predictions - actuals) / actuals_safe) * 100

    actual_errors = np.zeros_like(actuals)  # Actual vs actual = 0% error

    return pred_errors, actual_errors


def get_adaptive_ylim(returns: np.ndarray, percentile: float = 95) -> Tuple[float, float]:
    """
    Calculate adaptive y-axis limits based on return distribution.

    Uses percentiles to avoid extreme outliers affecting the scale.

    Args:
        returns: Array of percentage returns
        percentile: Percentile for determining limits (default: 95)

    Returns:
        Tuple of (y_min, y_max) for plotting
    """
    if len(returns) == 0:
        return -5.0, 5.0

    lower = np.percentile(returns, 100 - percentile)
    upper = np.percentile(returns, percentile)

    # Add 10% margin for readability
    margin = (upper - lower) * 0.1
    y_min = lower - margin
    y_max = upper + margin

    # Ensure minimum range of at least 1%
    if y_max - y_min < 1.0:
        mid = (y_max + y_min) / 2
        y_min = mid - 0.5
        y_max = mid + 0.5

    return y_min, y_max


def update_dashboard_data(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        device: torch.device,
        scaler: Optional[object] = None,
        max_samples: int = 100
) -> dict:
    """
    Update the global dashboard data with current epoch predictions.

    This function is called from the training loop to update the dashboard
    in real-time without blocking training.

    Args:
        model: The trained model
        val_loader: Validation data loader
        epoch: Current epoch number
        device: Device to run predictions on
        scaler: Optional scaler for inverse transform
        max_samples: Maximum number of samples to plot (for clarity)

    Returns:
        Dictionary containing prediction statistics
    """
    model.eval()

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Handle models that return tuple (predictions, attention, activations)
            output = model(X_batch)
            if isinstance(output, tuple):
                predictions = output[0]  # Extract just the predictions
            else:
                predictions = output

            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions).flatten()
    actuals = np.concatenate(all_actuals).flatten()

    # Inverse transform if scaler provided
    if scaler is not None:
        try:
            # Try to inverse transform (works if scaler is for single feature/target)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        except (ValueError, AttributeError) as e:
            # If scaler doesn't match dimensions, skip inverse transform
            # This happens when scaler was fitted on all features but we only have targets
            logging.warning(f"Skipping inverse transform: {e}")
            logging.warning("Displaying predictions in normalized scale. "
                            "Pass a target-specific scaler if you want actual price scale.")
            pass

    # Calculate returns
    pred_returns, _ = calculate_returns(predictions, actuals)

    # Limit samples for clarity
    if len(pred_returns) > max_samples:
        indices = np.random.choice(len(pred_returns), max_samples, replace=False)
        pred_returns_display = pred_returns[indices]
    else:
        pred_returns_display = pred_returns

    # Calculate statistics
    stats = {
        'epoch': epoch,
        'mean_return': float(np.mean(pred_returns)),
        'std_return': float(np.std(pred_returns)),
        'median_return': float(np.median(pred_returns)),
        'min_return': float(np.min(pred_returns)),
        'max_return': float(np.max(pred_returns)),
        'mae': float(np.mean(np.abs(pred_returns)))
    }

    # Update global dashboard data (thread-safe)
    with DASHBOARD_LOCK:
        DASHBOARD_DATA['current_epoch'] = epoch
        DASHBOARD_DATA['pred_returns'] = pred_returns_display.tolist()
        DASHBOARD_DATA['actuals'] = actuals.tolist()
        DASHBOARD_DATA['stats'] = stats
        DASHBOARD_DATA['epoch_history'].append(stats)
        # NEW: Store raw predictions and actuals for predictions vs actuals plot
        DASHBOARD_DATA['predictions_raw'] = predictions.tolist()
        DASHBOARD_DATA['actuals_raw'] = actuals.tolist()

    logging.info(f"Epoch {epoch} Prediction Stats - Mean Return: {stats['mean_return']:.3f}%, "
                 f"Std: {stats['std_return']:.3f}%, MAE: {stats['mae']:.3f}%")

    return stats


def create_dash_app(port: int = 8050) -> dash.Dash:
    """
    Create and configure the Dash application for real-time visualization.

    Args:
        port: Port to run the dashboard on (default: 8050)

    Returns:
        Configured Dash application
    """
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Transformer Prediction Visualization",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),

        html.Div(id='epoch-display',
                 style={'textAlign': 'center', 'fontSize': 24, 'marginBottom': 20}),

        # Statistics cards
        html.Div([
            html.Div([
                html.H3('Mean Error', style={'color': '#3498db'}),
                html.H2(id='mean-return', style={'color': '#2c3e50'})
            ], style={'display': 'inline-block', 'width': '15%', 'textAlign': 'center', 'padding': 10}),

            html.Div([
                html.H3('Std Dev', style={'color': '#e74c3c'}),
                html.H2(id='std-return', style={'color': '#2c3e50'})
            ], style={'display': 'inline-block', 'width': '15%', 'textAlign': 'center', 'padding': 10}),

            html.Div([
                html.H3('MAE', style={'color': '#f39c12'}),
                html.H2(id='mae', style={'color': '#2c3e50'})
            ], style={'display': 'inline-block', 'width': '15%', 'textAlign': 'center', 'padding': 10}),

            html.Div([
                html.H3('Min Error', style={'color': '#e74c3c'}),
                html.H2(id='min-return', style={'color': '#2c3e50'})
            ], style={'display': 'inline-block', 'width': '15%', 'textAlign': 'center', 'padding': 10}),

            html.Div([
                html.H3('Max Error', style={'color': '#27ae60'}),
                html.H2(id='max-return', style={'color': '#2c3e50'})
            ], style={'display': 'inline-block', 'width': '15%', 'textAlign': 'center', 'padding': 10}),
        ], style={'marginBottom': 30}),

        # Main plots - Errors
        html.Div([
            dcc.Graph(id='scatter-plot', style={'width': '49%', 'display': 'inline-block'}),
            dcc.Graph(id='histogram-plot', style={'width': '49%', 'display': 'inline-block'})
        ]),

        # Predictions vs Actuals (for detecting collapse)
        html.Div([
            dcc.Graph(id='predictions-vs-actuals', style={'width': '100%'})
        ]),

        # Evolution plots
        html.Div([
            dcc.Graph(id='evolution-mean', style={'width': '49%', 'display': 'inline-block'}),
            dcc.Graph(id='evolution-mae', style={'width': '49%', 'display': 'inline-block'})
        ]),

        # Auto-refresh interval (5000ms = 5 seconds)
        dcc.Interval(
            id='interval-component',
            interval=5000,  # Update every 5 seconds
            n_intervals=0
        )
    ], style={'padding': 20})

    # Callbacks for updating all components
    @app.callback(
        [Output('epoch-display', 'children'),
         Output('mean-return', 'children'),
         Output('std-return', 'children'),
         Output('mae', 'children'),
         Output('min-return', 'children'),
         Output('max-return', 'children'),
         Output('scatter-plot', 'figure'),
         Output('histogram-plot', 'figure'),
         Output('predictions-vs-actuals', 'figure'),
         Output('evolution-mean', 'figure'),
         Output('evolution-mae', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_all_components(n):
        with DASHBOARD_LOCK:
            current_epoch = DASHBOARD_DATA['current_epoch']
            pred_returns = DASHBOARD_DATA['pred_returns']
            stats = DASHBOARD_DATA['stats']
            history = DASHBOARD_DATA['epoch_history']
            # NEW: Get actual prediction and actual values
            predictions_raw = DASHBOARD_DATA.get('predictions_raw', [])
            actuals_raw = DASHBOARD_DATA.get('actuals_raw', [])

        if current_epoch == 0 or not pred_returns:
            # Return empty state
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Waiting for training data...",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return (
                "Waiting for training to start...",
                "N/A", "N/A", "N/A", "N/A", "N/A",
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            )

        # Update text displays
        epoch_text = f"Current Epoch: {current_epoch}"
        mean_text = f"{stats['mean_return']:.3f}%"
        std_text = f"{stats['std_return']:.3f}%"
        mae_text = f"{stats['mae']:.3f}%"
        min_text = f"{stats['min_return']:.3f}%"
        max_text = f"{stats['max_return']:.3f}%"

        # Scatter plot with adaptive scaling
        pred_returns_array = np.array(pred_returns)
        y_min, y_max = get_adaptive_ylim(pred_returns_array)

        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            y=pred_returns,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='Prediction Errors'
        ))
        scatter_fig.add_hline(y=0, line_dash="dash", line_color="red",
                              annotation_text="Perfect Prediction")
        scatter_fig.update_layout(
            title=f"Epoch {current_epoch} - Prediction Errors (Adaptive Scale)",
            xaxis_title="Sample Index",
            yaxis_title="Prediction Error (%)",
            yaxis_range=[y_min, y_max],
            template="plotly_white",
            height=400
        )

        # Histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=pred_returns,
            nbinsx=30,
            marker_color='blue',
            opacity=0.7,
            name='Returns'
        ))
        hist_fig.add_vline(x=0, line_dash="dash", line_color="red",
                           annotation_text="Zero Error")
        hist_fig.add_vline(x=stats['mean_return'], line_dash="dash", line_color="green",
                           annotation_text=f"Mean: {stats['mean_return']:.3f}%")
        hist_fig.update_layout(
            title=f"Epoch {current_epoch} - Error Distribution",
            xaxis_title="Prediction Error (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400
        )

        # NEW: Predictions vs Actuals plot (for detecting collapse)
        pred_vs_actual_fig = go.Figure()

        if predictions_raw and actuals_raw:
            # Scatter plot: predictions vs actuals
            pred_vs_actual_fig.add_trace(go.Scatter(
                x=actuals_raw,
                y=predictions_raw,
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.5),
                name='Predictions'
            ))

            # Perfect prediction line (y=x)
            min_val = min(min(actuals_raw), min(predictions_raw))
            max_val = max(max(actuals_raw), max(predictions_raw))
            pred_vs_actual_fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction (y=x)'
            ))

            # Check for collapse (predictions all similar)
            pred_std = np.std(predictions_raw)
            pred_range = np.max(predictions_raw) - np.min(predictions_raw)
            pred_mean = np.mean(predictions_raw)

            # Also check actual values variance
            actual_std = np.std(actuals_raw)
            actual_range = np.max(actuals_raw) - np.min(actuals_raw)
            actual_mean = np.mean(actuals_raw)

            collapse_warning = ""
            diagnosis = ""

            if pred_std < 0.01 or pred_range < 0.02:
                collapse_warning = "PREDICTION COLLAPSE!"
                diagnosis = "Predictions have very low variance"

            pred_vs_actual_fig.update_layout(
                title=f"Epoch {current_epoch} - Predictions vs Actuals{collapse_warning}",
                xaxis_title="Actual Values (normalized)",
                yaxis_title="Predicted Values (normalized)",
                template="plotly_white",
                height=500,
                annotations=[
                    dict(
                        text=(
                            f"<b>Predictions:</b> Mean={pred_mean:.4f}, Std={pred_std:.4f}, Range={pred_range:.4f}<br>"
                            f"<b>Actuals:</b> Mean={actual_mean:.4f}, Std={actual_std:.4f}, Range={actual_range:.4f}<br>"
                            f"<b>{diagnosis}</b>"),
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="black",
                        borderwidth=1,
                        align="left"
                    )
                ]
            )
        else:
            pred_vs_actual_fig.update_layout(
                title="Predictions vs Actuals (waiting for data...)",
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                template="plotly_white",
                height=500
            )

        # Evolution plots
        if len(history) > 1:
            epochs = [h['epoch'] for h in history]
            means = [h['mean_return'] for h in history]
            stds = [h['std_return'] for h in history]
            maes = [h['mae'] for h in history]

            # Mean + Std evolution
            evolution_mean_fig = go.Figure()
            evolution_mean_fig.add_trace(go.Scatter(
                x=epochs, y=means,
                mode='lines+markers',
                name='Mean Return',
                line=dict(color='blue', width=2)
            ))
            # Add std band
            upper_bound = [m + s for m, s in zip(means, stds)]
            lower_bound = [m - s for m, s in zip(means, stds)]
            evolution_mean_fig.add_trace(go.Scatter(
                x=epochs + epochs[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev'
            ))
            evolution_mean_fig.add_hline(y=0, line_dash="dash", line_color="red")
            evolution_mean_fig.update_layout(
                title="Mean Prediction Error Evolution",
                xaxis_title="Epoch",
                yaxis_title="Error (%)",
                template="plotly_white",
                height=400
            )

            # MAE evolution
            evolution_mae_fig = go.Figure()
            evolution_mae_fig.add_trace(go.Scatter(
                x=epochs, y=maes,
                mode='lines+markers',
                name='MAE',
                line=dict(color='orange', width=2),
                marker=dict(symbol='square')
            ))
            evolution_mae_fig.update_layout(
                title="Mean Absolute Error Evolution",
                xaxis_title="Epoch",
                yaxis_title="MAE (%)",
                template="plotly_white",
                height=400
            )
        else:
            evolution_mean_fig = go.Figure()
            evolution_mean_fig.update_layout(title="Evolution (need 2+ epochs)")
            evolution_mae_fig = go.Figure()
            evolution_mae_fig.update_layout(title="Evolution (need 2+ epochs)")

        return (
            epoch_text,
            mean_text, std_text, mae_text, min_text, max_text,
            scatter_fig, hist_fig, pred_vs_actual_fig, evolution_mean_fig, evolution_mae_fig
        )

    return app


def start_dashboard(port: int = 8050, open_browser: bool = True):
    """
    Start the Dash dashboard in a background thread.

    Args:
        port: Port to run the dashboard on
        open_browser: Whether to automatically open browser

    Returns:
        Thread object running the dashboard
    """
    app = create_dash_app(port)

    def run_dashboard():
        app.run(debug=False, port=port, use_reloader=False)

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    # Give server time to start
    time.sleep(2)

    if open_browser:
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        logging.info(f"Dashboard opened at {url}")

    return dashboard_thread


# Main function for training loop integration
def visualize_predictions_per_epoch(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        device: torch.device,
        scaler: Optional[object] = None,
        max_samples: int = 100,
        writer: Optional[object] = None  # Kept for backward compatibility
) -> dict:
    """
    Update prediction visualization for the current epoch.

    This is the main function called from the training loop.
    It updates the dashboard data which automatically refreshes in the browser.

    Args:
        model: The trained model
        val_loader: Validation data loader
        epoch: Current epoch number
        device: Device to run predictions on
        scaler: Optional scaler for inverse transform
        max_samples: Maximum number of samples to plot
        writer: TensorBoard writer (kept for compatibility, not used)

    Returns:
        Dictionary containing prediction statistics
    """
    return update_dashboard_data(model, val_loader, epoch, device, scaler, max_samples)


def visualize_prediction_evolution(
        epoch_stats: list,
        writer: Optional[object] = None
) -> None:
    """
    Legacy function kept for backward compatibility.

    With Dash, evolution is automatically shown in the dashboard.
    This function does nothing but is kept so existing code doesn't break.

    Args:
        epoch_stats: List of stats dictionaries (not used with Dash)
        writer: TensorBoard writer (not used with Dash)
    """
    logging.info("Evolution visualization is automatically displayed in the Dash dashboard.")
    pass