"""
train.py
---------
This script trains a stock prediction AI using indicators (RSI, MACD, SMA, etc.).
The training process:
1. Load and prepare data (from data_utils).
2. Define a neural network model.
3. Train the model on historical sequences.
4. Save the trained model.

TODO:
- Add support for multiple tickers (portfolio-level training).
- Support multiple walk-forward folds (not just one train/val split).
- Integrate wandb for richer experiment tracking
- Add multi-horizon forecasting (predict multiple days ahead).
- Add feature-wise attention / embedding layers for indicators.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import datetime
import subprocess
import webbrowser
import time
import platform
import uuid
import math
from utils.transformer_visuals import (
    update_attention_window,
    init_attention_window,
    set_feature_names,
    log_feature_importance_to_tensorboard
)
from utils.activation_health_check import (
    check_activation_health,
    log_activation_health_to_tensorboard
)


def log_activations_to_tensorboard(writer, activations, epoch):
    """
    Log activation histograms to TensorBoard.

    Args:
        writer: TensorBoard writer
        activations: Dict of activation tensors
        epoch: Current epoch
    """
    if writer is None:
        return

    for name, activation in activations.items():
        # Log histogram
        writer.add_histogram(f'Activations/{name}', activation, epoch)

        # Log statistics
        writer.add_scalar(f'Activations_Stats/{name}_mean',
                          activation.mean().item(), epoch)
        writer.add_scalar(f'Activations_Stats/{name}_std',
                          activation.std().item(), epoch)
        writer.add_scalar(f'Activations_Stats/{name}_max',
                          activation.max().item(), epoch)
        writer.add_scalar(f'Activations_Stats/{name}_min',
                          activation.min().item(), epoch)


def adaptive_grad_clip(model, percentile=95):
    """
    Adaptive gradient clipping based on gradient distribution.

    Args:
        model: PyTorch model
        percentile: Percentile to clip at (default: 95)

    Returns:
        float: Clip value used (for logging)
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).abs())

    if grads:
        all_grads = torch.cat(grads)
        clip_value = torch.quantile(all_grads, percentile / 100.0)
        torch.nn.utils.clip_grad_value_(model.parameters(),
                                        clip_value=clip_value.item())
        return clip_value.item()
    return None


def launch_tensorboard(logdir="runs", port=6006):
    """
    Launch TensorBoard as a subprocess and open it in the browser.

    Args:
        logdir (str): Directory containing TensorBoard logs.
        port (int): Port to run TensorBoard on (default: 6006).
    """
    try:
        # Kill any existing tensorboard processes (cross-platform)
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "tensorboard.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        else:  # Linux / macOS
            subprocess.run(["pkill", "-f", "tensorboard"], check=False)

        # Start TensorBoard
        tb_process = subprocess.Popen(
            ["tensorboard", f"--logdir={logdir}", f"--port={port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give TensorBoard a moment to start
        time.sleep(3)

        # Open browser automatically
        url = f"http://localhost:{port}"
        webbrowser.open(url)

        print(f"TensorBoard launched at {url}")
        return tb_process

    except FileNotFoundError:
        print("TensorBoard not found. Install it with `pip install tensorboard`.")
        return None


# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Device config
# -----------------------------
# Use GPU if available (important for training speed, on CPU it's basically impossible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colors for console outputs
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"

logging.info(f"{GREEN}Using device: {DEVICE}{RESET}")


# -----------------------------
# Experiment Logging
# -----------------------------
def get_tensorboard_writer(log_dir="runs"):
    """
    Create a TensorBoard writer for experiment logging.

    Args:
        log_dir (str): Base directory for logs.

    Returns:
        SummaryWriter object
    """
    # Add timestamped subfolder for each run
    timestamp = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_path = f"{log_dir}/run_{timestamp}"

    writer = SummaryWriter(log_dir=run_path)
    logging.info(f"TensorBoard logging started at {run_path}")

    return writer


# -----------------------------
# Transformer Model Definition
# -----------------------------

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings, need to inject position
    information because Transformers have no inherent sense of order
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use sine and cosine functions of different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformerPooled(nn.Module):
    """
    Variant that uses both mean and max pooling over the sequence
    instead of just taking the last timestep.
    More robust for capturing overall trends.
    """

    def __init__(
            self,
            input_size,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_len=5000

    ):
        super().__init__()

        self.return_attn = True  # allow extraction for transformer_visuals

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Output projection takes 2*d_model (mean + max pooling concatenated)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        activations = {}

        # Project and add positional encoding
        x = self.input_projection(x)
        activations['input_projection'] = x.detach()
        x = self.pos_encoder(x)
        activations['after_pos_encoding'] = x.detach()

        # to store attention values for viz
        all_attn = []

        # Manually forward through layers to access attention weights
        for layer_idx, layer in enumerate(self.transformer_encoder.layers):
            x_before = x
            x2, attn = layer.self_attn(
                x_before,
                x_before,
                x_before,
                need_weights=self.return_attn,
                average_attn_weights=False
            )
            all_attn.append(attn)  # shape: (batch, heads, seq, seq)

            # After attention
            x = layer.norm1(x_before + x2)
            activations[f'layer_{layer_idx}_after_attn'] = x.detach()

            # After feedforward
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + ff_out)
            activations[f'layer_{layer_idx}_after_ffn'] = x.detach()


        # Pool over sequence dimension
        # Mean pooling captures average pattern
        mean_pool = torch.mean(x, dim=1)  # (batch, d_model)
        # Max pooling captures strongest signals
        max_pool, _ = torch.max(x, dim=1)  # (batch, d_model)

        # Concatenate both pooling strategies
        x = torch.cat([mean_pool, max_pool], dim=1)  # (batch, 2*d_model)
        activations['concat_pool'] = x.detach()

        # Output projection
        output = self.output_projection(x)

        if self.return_attn:
            return output, all_attn, activations
        else:
            return output


# -----------------------------
# Training Loop
# -----------------------------

def train_model(X_train, y_train, X_val, y_val, input_size,
                epochs=20, batch_size=64, lr=1e-4, writer=None, scaler=None,
                early_stopping_patience=20, lr_scheduler_patience=5, lr_scheduler_factor=0.5,
                d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1,
                grad_clip_percentile=95, use_adaptive_clipping=True
                ):
    """
    Train the Transformer model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Number of input features
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        writer: TensorBoard writer
        scaler: Data scaler (for inverse transform)
        early_stopping_patience: Early stopping patience
        lr_scheduler_patience: LR scheduler patience
        lr_scheduler_factor: LR scheduler reduction factor
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        grad_clip_percentile: Percentile for adaptive clipping (default: 95)
        use_adaptive_clipping: use adaptive clipping or not (default: True)
    """

    set_feature_names(["close", "volume", "RSI", "MACD", "MACD_Signal", "SMA"])

    # Initialize Transformer model
    model = TimeSeriesTransformerPooled(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(DEVICE)

    if writer is not None:
        init_attention_window(num_layers, nhead, X_train.shape[1])

    logging.info(f"Initialized Transformer with {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.MSELoss()

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_scheduler_factor,
        patience=lr_scheduler_patience, min_lr=1e-6
    )

    # Dataset & DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")
    checkpoint_path = "best_model.pth"
    epochs_without_improvement = 0

    # Activation specific tracking variables
    consecutive_unhealthy_epochs = 0
    max_unhealthy_epochs = 5

    # Timer data storage
    epoch_times = []
    train_loop_times = []
    val_loop_times = []
    batch_load_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    adaptive_clipping_times = []
    other_times = []

    # Colors
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}TRAINING PROFILER{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --------------------------
        # TRAINING LOOP
        # --------------------------
        model.train()
        train_losses = []

        train_loop_start = time.time()

        # Per-epoch temporary accumulators
        ep_batch_load = 0.0
        ep_forward = 0.0
        ep_backward = 0.0
        ep_optimizer = 0.0
        ep_adaptive_clipping = 0.0
        ep_other = 0.0
        ep_train_loop = 0.0

        # Clipping tracking variables
        num_clips = 0
        total_clip_value = 0.0

        for batch_X, batch_y in train_loader:
            batch_load_start = time.time()
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            ep_batch_load += (time.time() - batch_load_start)

            # Forward pass
            fwd_start = time.time()

            outputs, attn, activations = model(batch_X)
            outputs = outputs.squeeze(-1)

            # Store the last batch_X and attn for epoch-level logging
            last_batch_X = batch_X
            last_attn = attn

            # Compute mean attention map across batch
            mean_attn = [a.mean(dim=0).detach().cpu().numpy() for a in attn]
            # mean_attn becomes a list: [ (heads, seq, seq), ... per layer ]

            loss = criterion(outputs, batch_y)

            ep_forward += time.time() - fwd_start

            # Backward pass
            bwd_start = time.time()
            loss.backward()

            ep_backward += time.time() - bwd_start

            # Adaptive gradient clipping
            if use_adaptive_clipping:
                adapt_start = time.time()
                clip_value = adaptive_grad_clip(model, percentile=grad_clip_percentile)
                if clip_value is not None:
                    num_clips += 1
                    total_clip_value += clip_value
                ep_adaptive_clipping += time.time() - adapt_start

            # Optimizer
            opt_start = time.time()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

            ep_optimizer += time.time() - opt_start

        # Timer values
        ep_train_loop += time.time() - train_loop_start
        ep_other = ep_train_loop - (ep_forward + ep_backward + ep_optimizer
                                    + ep_adaptive_clipping + ep_batch_load)

        train_loop_times.append(ep_train_loop)

        batch_load_times.append(ep_batch_load)
        forward_times.append(ep_forward)
        backward_times.append(ep_backward)
        optimizer_times.append(ep_optimizer)
        adaptive_clipping_times.append(ep_adaptive_clipping)
        other_times.append(ep_other)


        # --------------------------
        # VALIDATION LOOP
        # --------------------------
        model.eval()
        val_loop_start = time.time()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                outputs, _, _ = model(batch_X)
                outputs = outputs.squeeze(-1)

                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())

        val_loop_times.append(time.time() - val_loop_start)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        scheduler.step(avg_val_loss)

        epoch_times.append(time.time() - epoch_start)

        if writer is not None:
            update_attention_window(mean_attn, epoch)

        # First console outputs
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}EPOCH {epoch} / {epochs}{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")

        # --------------------------
        # TENSORBOARD LOGGING
        # --------------------------
        if writer is not None:
            log_feature_importance_to_tensorboard(writer, last_batch_X, last_attn, epoch)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            if epoch % 1 == 0:  # Every epoch
                log_activations_to_tensorboard(writer, activations, epoch)
                log_activation_health_to_tensorboard(writer, activations, epoch, num_layers)

            # Gradient clipping statistics
            if num_clips > 0:
                avg_clip_value = total_clip_value / num_clips
                clip_rate = num_clips / len(train_loader)
            else:
                avg_clip_value = 0.0
                clip_rate = 0.0

            writer.add_scalar('Gradients/avg_clip_value', avg_clip_value, epoch)
            writer.add_scalar('Gradients/clip_rate', clip_rate, epoch)

            # Colors
            YELLOW = "\033[93m"
            GREEN = "\033[92m"
            RED = "\033[91m"
            RESET = "\033[0m"

            # Warning if clipping too frequently
            if clip_rate > 0.8:
                print(f"\n{YELLOW}{'-' * 70}{RESET}")
                print(f"{YELLOW}ADAPTIVE CLIPPING WARNING (Epoch {epoch}){RESET}")
                print(f"{YELLOW}{'-' * 70}{RESET}")
                logging.warning(f"High clipping rate ({clip_rate:.1%})")
                print(f"{YELLOW}{'-' * 70}{RESET}\n")

        # --------------------------
        # ACTIVATION HEALTH CHECK
        # --------------------------
        # Run health check (even if writer is None)
        is_healthy, warnings = check_activation_health(activations, epoch, num_layers)

        # Print warnings to console if any issues detected
        if not is_healthy or len(warnings) > 0:
            if is_healthy:
                print(f"\n{YELLOW}{'-' * 70}{RESET}")
                print(f"{YELLOW}ACTIVATION WARNINGS (Epoch {epoch}){RESET}")
                print(f"{YELLOW}{'-' * 70}{RESET}")
            else:
                print(f"\n{RED}{'-' * 70}{RESET}")
                print(f"{RED}CRITICAL ACTIVATION ISSUES (Epoch {epoch}){RESET}")
                print(f"{RED}{'-' * 70}{RESET}")
            for warning in warnings:
                print(f"  {warning}")

            if is_healthy:
                print(f"{YELLOW}{'-' * 70}{RESET}\n")
            else:
                print(f"{RED}{'-' * 70}{RESET}\n")

        # Track consecutive unhealthy epochs
        if not is_healthy:
            consecutive_unhealthy_epochs += 1
            logging.warning(f"Unhealthy activations for {consecutive_unhealthy_epochs} consecutive epoch(s)")
        else:
            # Reset counter if activations are healthy
            if consecutive_unhealthy_epochs > 0:
                logging.info(f"Activations recovered after {consecutive_unhealthy_epochs} unhealthy epoch(s)")
            consecutive_unhealthy_epochs = 0

        # --------------------------
        # CONSOLE LOGGING
        # --------------------------
        logging.info(f"{BLUE}Epoch {epoch}/{epochs}{RESET}")
        total_train = train_loop_times[-1]

        print(f"  Train Loop: {total_train:.3f}s")
        print(f"    Batch loading:       {ep_batch_load:>6.2f}s ({ep_batch_load / total_train * 100:>5.1f}%)")
        print(f"    Forward pass :       {ep_forward:>6.2f}s ({ep_forward / total_train * 100:>5.1f}%)")
        print(f"    Backward pass:       {ep_backward:>6.2f}s ({ep_backward / total_train * 100:>5.1f}%)")

        if use_adaptive_clipping:
            print(f"    Adaptive clipping:   {ep_adaptive_clipping:>6.2f}s ({ep_adaptive_clipping / total_train * 100:>5.1f}%)")

        print(f"    Optimizer:           {ep_optimizer:>6.2f}s ({ep_optimizer / total_train * 100:>5.1f}%)")
        print(f"    Other:               {ep_other:>6.2f}s ({ep_other / total_train * 100:>5.1f}%)")
        print(f"  Validation Loop:   {val_loop_times[-1]:.3f}s")
        print(f"  Total:             {epoch_times[epoch-1]:.3f}s")

        if avg_val_loss < best_val_loss:
            print(f"{GREEN}  Loss:              {avg_train_loss:.6f} (train), {avg_val_loss:.6f} (val){RESET}")
        else:
            print(f"  Loss:              {avg_train_loss:.6f} (train), {avg_val_loss:.6f} (val){RESET}")
        print(f"  LR:                {optimizer.param_groups[0]['lr']:.2e}")

        # --------------------------
        # Early Stopping
        # --------------------------
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"{GREEN}New best model saved (val_loss: {avg_val_loss:.6f}){RESET}")
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info(f"{RED}Epochs without improvement: {epochs_without_improvement}{RESET}")
            if epochs_without_improvement >= early_stopping_patience:
                logging.info(
                    f"\n{RED}Early stopping triggered after {epochs_without_improvement} epochs without improvement{RESET}")
                break

    # --------------------------
    # LOAD BEST MODEL
    # --------------------------
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info(f"Loaded best model from {checkpoint_path}")

    # --------------------------
    # FINAL PROFILING SUMMARY
    # --------------------------
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}TRAINING PROFILER SUMMARY{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")
    print(f"Epoch avg time:        {np.mean(epoch_times):.3f}s")
    print(f"Train loop avg:        {np.mean(train_loop_times):.3f}s")
    print(f"Val loop avg:          {np.mean(val_loop_times):.3f}s")

    print("\n--- Batch timings ---")
    print(f"Avg batch load:        {np.mean(batch_load_times):.6f}s")
    print(f"Avg forward pass:      {np.mean(forward_times):.6f}s")
    print(f"Avg backward pass:     {np.mean(backward_times):.6f}s")
    print(f"Avg adaptive clipping: {np.mean(adaptive_clipping_times):.6f}s")
    print(f"Avg optimizer step:    {np.mean(optimizer_times):.6f}s")
    print(f"Avg other:             {np.mean(other_times):.6f}s")

    return model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Launch TensorBoard automatically
    tb_process = launch_tensorboard(logdir="runs", port=6006)

    # Create dummy train/val data
    logging.info("Creating dummy train/validation data...")

    # Simulate full dataset
    total_samples = 100
    window_size = 20
    num_features = 5
    train_size = 0.8

    # Generate dummy data
    X_full = np.random.rand(total_samples, window_size, num_features)
    y_full = np.random.rand(total_samples)

    # Split manually
    split_idx = int(total_samples * train_size)

    X_train = X_full[:split_idx]
    y_train = y_full[:split_idx]
    X_val = X_full[split_idx:]
    y_val = y_full[split_idx:]

    # Verify split
    assert len(X_train) == 80, f"Expected 80 train samples, got {len(X_train)}"
    assert len(y_train) == 80, f"Expected 80 train targets, got {len(y_train)}"
    assert len(X_val) == 20, f"Expected 20 validation samples, got {len(X_val)}"
    assert len(y_val) == 20, f"Expected 20 validation targets, got {len(y_val)}"

    logging.info("Data split verification passed.")

    # Test TensorBoard writer
    writer = get_tensorboard_writer()
    writer.add_scalar("Test/Loss", 0.123, 1)  # log dummy value
    writer.close()

    logging.info("TensorBoard writer test completed. Check 'runs/' folder for logs")

    # Train the Transformer model with the dummy data
    logging.info("Starting dummy training loop...")
    writer = get_tensorboard_writer()  # reopen writer for training logs
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],
        epochs=5,
        batch_size=32,
        lr=1e-4,
        writer=writer,
        scaler=None,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        early_stopping_patience=10,
        lr_scheduler_patience=3,
        lr_scheduler_factor=0.5
    )
    writer.close()
    logging.info("Dummy training completed.")

    # Keep TensorBoard alive until script stops
    try:
        logging.info("TensorBoard running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if tb_process:
            tb_process.terminate()
            logging.info("TensorBoard stopped.")