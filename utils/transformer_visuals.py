import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import numpy as np
import webbrowser
from time import sleep
import logging

# Global state
_app = None
_server_thread = None
_current_data = None
_num_layers = None
_n_heads = None
_current_epoch = 0
_initialized = False

# Feature importance tracking
_feature_importance_history = []
_feature_names = None


def init_attention_window(num_layers, n_heads, seq_len, port=8050):
    """
    Creates a persistent Dash app showing attention heatmaps.
    One subplot per head per layer.
    Opens in browser automatically.
    """
    global _app, _server_thread, _current_data, _num_layers, _n_heads, _initialized

    if _initialized:
        return

    _num_layers = num_layers
    _n_heads = n_heads

    # Initialize with zeros
    _current_data = [np.zeros((n_heads, seq_len, seq_len)) for _ in range(num_layers)]

    # Create Dash app
    _app = dash.Dash(__name__)

    # Layout
    _app.layout = html.Div([
        html.H1("Attention Heatmaps", style={'textAlign': 'center'}),
        html.Div(id='epoch-display', style={'textAlign': 'center', 'fontSize': 20}),
        dcc.Graph(id='attention-heatmap', style={'height': '90vh'}),
        dcc.Interval(
            id='interval-component',
            interval=5000,  # Update every 5000ms -> 5s
            n_intervals=0
        )
    ])

    # Callback to update the graph
    @_app.callback(
        [Output('attention-heatmap', 'figure'),
         Output('epoch-display', 'children')],
        Input('interval-component', 'n_intervals')
    )
    def update_graph(n):
        global _current_data, _current_epoch, _num_layers, _n_heads

        # Create subplots
        fig = make_subplots(
            rows=_num_layers,
            cols=_n_heads,
            subplot_titles=[f"L{l + 1} H{h + 1}" for l in range(_num_layers) for h in range(_n_heads)],
            vertical_spacing=0.1 / _num_layers,
            horizontal_spacing=0.05 / _n_heads
        )

        # Add heatmaps
        for layer in range(_num_layers):
            for head in range(_n_heads):
                fig.add_trace(
                    go.Heatmap(
                        z=_current_data[layer][head],
                        colorscale='Viridis',
                        showscale=(head == _n_heads - 1),  # Only show colorbar on last column
                        zmin=None,
                        zmax=None
                    ),
                    row=layer + 1,
                    col=head + 1
                )

        # Update layout
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            height=max(300 * _num_layers, 600),
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=20)
        )

        epoch_text = f"Epoch: {_current_epoch}"

        return fig, epoch_text

    # Run server in background thread
    def run_server():
        # Silence werkzeug and dash loggers
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        log.disabled = True

        # Also silence the root logger for Flask
        logging.getLogger('dash.dash').setLevel(logging.ERROR)

        _app.run(debug=False, port=port, use_reloader=False)

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    # Open browser after short delay
    sleep(1.5)

    print(f"Attention visualization running on Dash at http://127.0.0.1:{port}")

    url = f"http://localhost:{port}"
    webbrowser.open(url)

    _initialized = True


def update_attention_window(mean_attn, epoch):
    """
    mean_attn: list of [heads, seq, seq] arrays (one per layer)
    Updates the live attention heatmap window.
    """
    global _current_data, _current_epoch

    if not _initialized:
        print("Warning: Attention window not initialized. Call init_attention_window() first.")
        return

    _current_data = mean_attn
    _current_epoch = epoch


def set_feature_names(feature_names):
    """
    Set the names of features for importance tracking.
    Should be called once before training.

    Args:
        feature_names: List of feature names (e.g., ['close_return', 'RSI', 'MACD', 'SMA'])
    """
    global _feature_names, _feature_importance_history
    _feature_names = feature_names
    _feature_importance_history = []

    print(f"Feature names set for importance tracking: {feature_names}")

def compute_feature_importance(batch_X, attn_weights):
    """
    Compute per-feature attention importance from attention weights.

    This works by:
    1. Taking attention weights across all heads and layers
    2. Computing how much attention each timestep receives (incoming attention)
    3. Weighting each feature by both attention and its magnitude
    4. Aggregating to get per-feature importance scores

    The algorithm now:
    - Uses both incoming and outgoing attention for better coverage
    - Weights by feature magnitude (absolute value) to capture influence
    - Normalizes properly to sum to 1.0

    Args:
        batch_X: Input batch tensor of shape (batch, seq_len, num_features)
        attn_weights: List of attention tensors from each layer
                     Each tensor has shape (batch, heads, seq_len, seq_len)

    Returns:
        feature_importance: numpy array of shape (num_features,) with importance scores
    """
    if len(attn_weights) == 0:
        return None

    batch_size, seq_len, num_features = batch_X.shape

    # Stack all attention weights: (num_layers, batch, heads, seq_len, seq_len)
    stacked_attn = torch.stack(attn_weights, dim=0)

    # Average across layers and heads: (batch, seq_len, seq_len)
    avg_attn = stacked_attn.mean(dim=(0, 2))

    # Compute both incoming and outgoing attention for each timestep
    # Incoming: how much attention this position receives (column sum)
    # Outgoing: how much attention this position gives (row sum)
    incoming_attn = avg_attn.sum(dim=1)  # (batch, seq_len)
    outgoing_attn = avg_attn.sum(dim=2)  # (batch, seq_len)

    # Combine both (average them)
    timestep_importance = (incoming_attn + outgoing_attn) / 2

    # Average across batch: (seq_len,)
    timestep_importance = timestep_importance.mean(dim=0)

    # Now attribute timestep importance to features
    # Use absolute values of input features weighted by their attention

    # Get absolute feature values averaged across batch: (seq_len, num_features)
    feature_magnitudes = batch_X.abs().mean(dim=0)

    # Also consider feature variance as a signal of importance
    # Features with higher variance are likely more informative
    feature_variance = batch_X.var(dim=0).mean(dim=0)  # (num_features,)

    # Normalize variance to [0, 1] range
    if feature_variance.max() > 0:
        feature_variance = feature_variance / feature_variance.max()

    # Weight each feature by timestep importance: (seq_len, num_features)
    weighted_features = feature_magnitudes * timestep_importance.unsqueeze(-1)

    # Sum across timesteps to get per-feature importance: (num_features,)
    feature_importance = weighted_features.sum(dim=0)

    # Boost by variance (features with higher variance get slight boost)
    # This helps identify features that are actively changing vs. static
    feature_importance = feature_importance * (1 + 0.5 * feature_variance)

    # Normalize to sum to 1
    feature_importance = feature_importance / (feature_importance.sum() + 1e-8)

    return feature_importance.detach().cpu().numpy()


def log_feature_importance_to_tensorboard(writer, batch_X, attn_weights, epoch):
    """
    Compute feature importance and log to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        batch_X: Input batch tensor of shape (batch, seq_len, num_features)
        attn_weights: List of attention tensors from each layer
        epoch: Current epoch number
    """
    global _feature_names, _feature_importance_history

    if writer is None or _feature_names is None:
        return

    # Compute feature importance
    feature_importance = compute_feature_importance(batch_X, attn_weights)

    if feature_importance is None:
        return

    # Validate that feature importance shape matches feature names
    if len(feature_importance) != len(_feature_names):
        print(f"WARNING: Feature importance shape ({len(feature_importance)}) "
              f"doesn't match feature names ({len(_feature_names)})")
        return

    # Store history
    _feature_importance_history.append(feature_importance)

    # Log individual feature importances
    for i, feature_name in enumerate(_feature_names):
        writer.add_scalar(f'FeatureImportance/{feature_name}',
                          feature_importance[i],
                          epoch)

    # Also log a bar chart for better visualization
    if epoch % 5 == 0:  # Every 5 epochs to avoid clutter
        # Create a simple text representation
        importance_str = "\n".join([
            f"{name}: {importance:.4f}"
            for name, importance in zip(_feature_names, feature_importance)
        ])
        writer.add_text('FeatureImportance/Summary', importance_str, epoch)


def get_feature_importance_history():
    """
    Get the history of feature importances across epochs.

    Returns:
        numpy array of shape (num_epochs, num_features)
    """
    global _feature_importance_history
    if len(_feature_importance_history) == 0:
        return None
    return np.array(_feature_importance_history)


def plot_feature_importance_evolution():
    """
    Plot how feature importance evolves over training.

    Returns:
        matplotlib figure showing importance trends
    """
    history = get_feature_importance_history()

    if history is None or _feature_names is None:
        print("No feature importance history available")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, feature_name in enumerate(_feature_names):
        ax.plot(history[:, i], label=feature_name, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Feature Importance', fontsize=12)
    ax.set_title('Feature Importance Evolution During Training', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def reset_feature_tracking():
    """
    Reset feature importance tracking.
    Call this when starting a new training run.
    """
    global _feature_importance_history
    _feature_importance_history = []


if __name__ == "__main__":
    pass