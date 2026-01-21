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

from utils.prediction_visualization import start_dashboard, visualize_predictions_per_epoch
from utils.transformer_visuals import (
    update_attention_window,
    init_attention_window,
    set_feature_names,
    log_feature_importance_to_tensorboard, reset_feature_tracking
)
from utils.activation_health_check import (
    check_activation_health,
    log_activation_health_to_tensorboard
)
from utils.embeddings_visualization import (
    log_embeddings_to_tensorboard,
    log_multiple_embedding_views
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


def simple_grad_clip(model, max_norm=1.0):
    """Simple and effective gradient norm clipping."""
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    clipped = total_norm > max_norm
    return {
        'clipped': clipped,
        'grad_norm': total_norm.item(),
        'clip_value': max_norm
    }


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


# Custom loss function
class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        # Penalty if predictions have no variance (all same)
        variance_penalty = 1.0 / (pred.std() + 1e-6)
        return mse_loss + 0.01 * variance_penalty


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
        pass

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
                max_norm=1.0, use_gradient_clipping=True
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
        grad_clip_percentile: Percentile for gradient clipping (default: 1.0)
        use_gradient_clipping: use gradient clipping or not (default: True)
    """

    reset_feature_tracking()

    dashboard_thread = start_dashboard(port=8052, open_browser=True)

    set_feature_names(["close_return","RSI", "MACD_Histogram", "SMA_Deviation", "ATR", "Volume_Ratio"])

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

    print(f"Initialized Transformer with {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = VarianceLoss()

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
    gradient_clipping_times = []
    other_times = []

    # Gradient norm clipping statistics
    num_clips = 0
    total_grad_norm = 0.0
    total_clip_value = 0.0
    num_batches_with_grads = 0

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
        ep_gradient_clipping = 0.0
        ep_other = 0.0
        ep_train_loop = 0.0


        # Clipping tracking variables
        num_clips = 0
        total_grad_norm = 0.0
        total_clip_value = 0.0
        num_batches_with_grads = 0

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

            mse_loss = criterion(outputs, batch_y)

            ep_forward += time.time() - fwd_start

            # Backward pass
            bwd_start = time.time()
            mse_loss.backward()

            ep_backward += time.time() - bwd_start

            # Gradient clipping
            if use_gradient_clipping:
                adapt_start = time.time()
                clip_info = simple_grad_clip(model, max_norm=1.0)

                # Track statistics
                if clip_info['grad_norm'] > 0:  # Only track if gradients exist
                    num_batches_with_grads += 1
                    total_grad_norm += clip_info['grad_norm']
                    total_clip_value += clip_info['clip_value']

                    if clip_info['clipped']:
                        num_clips += 1

                ep_gradient_clipping += time.time() - adapt_start

            # Optimizer
            opt_start = time.time()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(mse_loss.item())

            ep_optimizer += time.time() - opt_start

        # Calculate gradient clipping statistics for the epoch
        if use_gradient_clipping and num_batches_with_grads > 0:
            clip_rate = num_clips / num_batches_with_grads
            avg_grad_norm = total_grad_norm / num_batches_with_grads
            avg_clip_value = total_clip_value / num_batches_with_grads
        else:
            clip_rate = 0.0
            avg_grad_norm = 0.0
            avg_clip_value = 0.0

        # Timer values
        ep_train_loop += time.time() - train_loop_start
        ep_other = ep_train_loop - (ep_forward + ep_backward + ep_optimizer
                                    + ep_gradient_clipping + ep_batch_load)

        train_loop_times.append(ep_train_loop)

        batch_load_times.append(ep_batch_load)
        forward_times.append(ep_forward)
        backward_times.append(ep_backward)
        optimizer_times.append(ep_optimizer)
        gradient_clipping_times.append(ep_gradient_clipping)
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

                mse_loss = criterion(outputs, batch_y)
                val_losses.append(mse_loss.item())

        # --------------------------
        # UPDATE DASHBOARD
        # --------------------------
        # Update every epoch
        if epoch % 1 == 0:
            pred_stats = visualize_predictions_per_epoch(
                model=model,
                val_loader=val_loader,
                epoch=epoch,
                device=DEVICE,
                scaler=None,
                max_samples=100
            )

        val_loop_times.append(time.time() - val_loop_start)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # Calculate directional accuracy (for returns)
        with torch.no_grad():
            train_preds = []
            train_actuals = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs, _, _ = model(batch_X)
                train_preds.extend(outputs.squeeze(-1).cpu().numpy())
                train_actuals.extend(batch_y.cpu().numpy())

            train_preds = np.array(train_preds)
            train_actuals = np.array(train_actuals)

            # Directional accuracy: did we predict the sign correctly?
            train_direction_acc = np.mean(np.sign(train_preds) == np.sign(train_actuals))

            val_preds = []
            val_actuals = []
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs, _, _ = model(batch_X)
                val_preds.extend(outputs.squeeze(-1).cpu().numpy())
                val_actuals.extend(batch_y.cpu().numpy())

            val_preds = np.array(val_preds)
            val_actuals = np.array(val_actuals)
            val_direction_acc = np.mean(np.sign(val_preds) == np.sign(val_actuals))

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
            writer.add_scalar('Metrics/train_direction_accuracy', train_direction_acc, epoch)
            writer.add_scalar('Metrics/val_direction_accuracy', val_direction_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            if epoch % 1 == 0:  # Every epoch
                log_activations_to_tensorboard(writer, activations, epoch)
                log_activation_health_to_tensorboard(writer, activations, epoch, num_layers)

            # Gradient clipping statistics
            if use_gradient_clipping:
                writer.add_scalar('Gradients/clip_rate', clip_rate, epoch)
                writer.add_scalar('Gradients/avg_grad_norm', avg_grad_norm, epoch)
                writer.add_scalar('Gradients/avg_clip_value', avg_clip_value, epoch)

                # Additional useful metric: ratio of grad_norm to clip_value
                if avg_clip_value > 0:
                    norm_to_clip_ratio = avg_grad_norm / avg_clip_value
                    writer.add_scalar('Gradients/norm_to_clip_ratio', norm_to_clip_ratio, epoch)

            # Log embeddings every 5 epochs
            if epoch % 5 == 0:
                print(f"  Logging embeddings to TensorBoard...")
                log_embeddings_to_tensorboard(
                    writer=writer,
                    model=model,
                    data_loader=val_loader,
                    epoch=epoch,
                    max_samples=1000,
                    metadata_type='price_change',
                    device=DEVICE
                )

            with torch.no_grad():
                val_mse_losses = []
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    outputs, _, _ = model(batch_X)
                    outputs = outputs.squeeze(-1)
                    mse= criterion(outputs, batch_y)
                    val_mse_losses.append(mse.item())

                avg_val_mse = np.mean(val_mse_losses)

                writer.add_scalar('Loss/mse', avg_val_mse, epoch)

            writer.flush() # Ensures data is written immediately

            # Colors
            YELLOW = "\033[93m"
            GREEN = "\033[92m"
            RED = "\033[91m"
            RESET = "\033[0m"

            # Warning if clipping too frequently
            if use_gradient_clipping:
                if clip_rate > 0.3:  # More than 30% of batches clipping
                    print(f"\n{YELLOW}{'-' * 70}{RESET}")
                    print(f"{YELLOW}GRADIENT CLIPPING WARNING (Epoch {epoch}){RESET}")
                    print(f"{YELLOW}{'-' * 70}{RESET}")
                    print(f"  Clip Rate:       {clip_rate:.1%} (>{30}% threshold)")
                    print(f"  Avg Grad Norm:   {avg_grad_norm:.4f}")
                    print(f"  Avg Clip Value:  {avg_clip_value:.4f}")
                    print(f"  Ratio:           {avg_grad_norm / avg_clip_value:.2f}x")
                    print(f"{YELLOW}{'-' * 70}{RESET}\n")
                elif clip_rate > 0.0:
                    # Info message if some clipping occurred (but not too much)
                    logging.info(f"\n{GREEN}Gradient clipping: {clip_rate:.1%} of batches clipped (healthy){RESET}")

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

        if epoch % 5 == 0:
            pred_std = outputs.std().item()
            target_std = batch_y.std().item()

            if pred_std < 0.0001:
                print(f"\n{RED}WARNING: Prediction collapse detected!")
                print(f"  Pred std: {pred_std:.6f}")
                print(f"  Target std: {target_std:.6f}{RESET}\n")


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

        if use_gradient_clipping:
            print(f"    Gradient clipping:   {ep_gradient_clipping:>6.2f}s ({ep_gradient_clipping / total_train * 100:>5.1f}%)")

        print(f"    Optimizer:           {ep_optimizer:>6.2f}s ({ep_optimizer / total_train * 100:>5.1f}%)")
        print(f"    Other:               {ep_other:>6.2f}s ({ep_other / total_train * 100:>5.1f}%)")
        print(f"  Validation Loop:     {val_loop_times[-1]:.3f}s")
        print(f"  Total:               {epoch_times[epoch-1]:.3f}s | {(epoch_times[epoch-1]) / 60:.2f}m")

        if avg_val_loss < best_val_loss:
            print(f"{GREEN}  Loss (MSE):          {avg_train_loss:.8f} (train), {avg_val_loss:.8f} (val){RESET}")
        else:
            print(f"  Loss (MSE):          {avg_train_loss:.8f} (train), {avg_val_loss:.8f} (val){RESET}")

        # Add return magnitude context:
        print(f"  RMSE (return):       {np.sqrt(avg_train_loss):.4%} (train), {np.sqrt(avg_val_loss):.4%} (val)")

        print(f"  Direction Accuracy:  {train_direction_acc:.1%} (train), {val_direction_acc:.1%} (val)")
        if use_gradient_clipping:
            print(f"  Gradient Norm:       {avg_grad_norm:.4f} (clip rate: {clip_rate:.1%})")
        print(f"  LR:                  {optimizer.param_groups[0]['lr']:.2e}")

        # --------------------------
        # Early Stopping
        # --------------------------
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"{GREEN}New best model saved (val_loss: {avg_val_loss:.8f}, dir_acc: {val_direction_acc:.1%}){RESET}")
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

    print(f"{BLUE}\n--- Batch timings ---{RESET}")
    print(f"Avg batch load:        {np.mean(batch_load_times):.6f}s")
    print(f"Avg forward pass:      {np.mean(forward_times):.6f}s")
    print(f"Avg backward pass:     {np.mean(backward_times):.6f}s")
    print(f"Avg gradient clipping: {np.mean(gradient_clipping_times):.6f}s")
    print(f"Avg optimizer step:    {np.mean(optimizer_times):.6f}s")
    print(f"Avg other:             {np.mean(other_times):.6f}s")

    print(f"\n{BLUE}--- Gradient Clipping Summary ---{RESET}")
    print(f"Avg gradient clipping: {np.mean(gradient_clipping_times):.6f}s")
    print(f"Avg clip rate:         {clip_rate:.1%}")
    print(f"Avg grad norm:         {avg_grad_norm:.4f}")
    print(f"Avg clip value:        {avg_clip_value:.4f}")

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