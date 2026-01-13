import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def log_embeddings_to_tensorboard(writer, model, data_loader, epoch,
                                  max_samples=1000, metadata_type='time', device='cpu'):
    """
    Log embeddings to TensorBoard's projector for 3D visualization.

    Args:
        writer: TensorBoard SummaryWriter
        model: Your transformer model
        data_loader: DataLoader with validation/test data
        epoch: Current epoch number
        max_samples: Maximum number of samples to visualize (default: 1000)
        metadata_type: Type of metadata for coloring ('time', 'volatility', 'price_change')
        device: device the model was trained on
    """
    model.eval()
    device = device

    embeddings_list = []
    metadata_list = []
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
            if sample_count >= max_samples:
                break

            batch_X = batch_X.to(device)

            # Extract embeddings from the model
            embeddings = extract_embeddings(model, batch_X)

            # Move to CPU for storage
            embeddings_list.append(embeddings.cpu())

            # Create metadata for coloring (compute on same device, then move to CPU)
            if metadata_type == 'time':
                # Color by batch index (represents time progression)
                metadata = torch.full((embeddings.shape[0],), batch_idx, dtype=torch.long)
            elif metadata_type == 'volatility':
                # Calculate volatility from input sequence (std of close prices)
                # Close price is first feature
                volatility = batch_X[:, :, 0].std(dim=1)
                # Discretize into categories for better visualization
                metadata = discretize_values(volatility, num_bins=5).cpu()
            elif metadata_type == 'price_change':
                # Calculate price change direction
                price_start = batch_X[:, 0, 0]  # First close price
                price_end = batch_X[:, -1, 0]  # Last close price
                price_change_pct = ((price_end - price_start) / price_start) * 100
                metadata = discretize_values(price_change_pct, num_bins=5).cpu()
            else:
                metadata = torch.zeros(embeddings.shape[0], dtype=torch.long)

            metadata_list.append(metadata)
            sample_count += embeddings.shape[0]

    # Concatenate all embeddings and metadata (already on CPU)
    all_embeddings = torch.cat(embeddings_list, dim=0)[:max_samples]
    all_metadata = torch.cat(metadata_list, dim=0)[:max_samples]

    # Create metadata labels
    if metadata_type == 'time':
        label_names = [f'batch_{i}' for i in all_metadata.tolist()]
    elif metadata_type in ['volatility', 'price_change']:
        label_names = [f'bin_{i}' for i in all_metadata.tolist()]
    else:
        label_names = None

    # Log to TensorBoard

    # Convert metadata to list of strings for better compatibility
    metadata_list_str = [str(int(m.item())) for m in all_metadata]

    # Include epoch in tag name so each epoch creates a separate projector view
    tag_name = f'embeddings/{metadata_type}_epoch{epoch:04d}'

    print(f"  DEBUG: Embedding shape: {all_embeddings.shape}")
    print(f"  DEBUG: Metadata shape: {all_metadata.shape}")
    print(f"  DEBUG: Unique metadata values: {torch.unique(all_metadata).tolist()}")

    writer.add_embedding(
        all_embeddings,
        metadata=metadata_list_str,
        global_step=epoch,
        tag=tag_name
    )

    print(f"  Logged {all_embeddings.shape[0]} embeddings to TensorBoard (colored by {metadata_type}), tag name: {tag_name}")


def extract_embeddings(model, batch_X):
    """
    Extract embeddings from TimeSeriesTransformerPooled model.

    Returns the concatenated pooled representation (mean + max pooling)
    before the final output projection. This is the learned representation
    of each time series sample.

    Args:
        model: Your TimeSeriesTransformerPooled model
        batch_X: Input batch tensor

    Returns:
        embeddings: Tensor of shape (batch_size, 2*d_model)
    """
    # Project and add positional encoding
    x = model.input_projection(batch_X)  # (batch, seq, d_model)
    x = model.pos_encoder(x)

    # Pass through all transformer encoder layers
    # Manually iterate to avoid attention weight extraction overhead
    for layer in model.transformer_encoder.layers:
        # Self-attention
        x_before = x
        x2, _ = layer.self_attn(x_before, x_before, x_before, need_weights=False)
        x = layer.norm1(x_before + x2)

        # Feedforward
        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + ff_out)

    # Apply the same pooling strategy as the model
    mean_pool = torch.mean(x, dim=1)  # (batch, d_model)
    max_pool, _ = torch.max(x, dim=1)  # (batch, d_model)

    # Concatenate both pooling strategies - THIS IS THE EMBEDDING
    embeddings = torch.cat([mean_pool, max_pool], dim=1)  # (batch, 2*d_model)

    return embeddings


def discretize_values(values, num_bins=5):
    """
    Discretize continuous values into bins for categorical coloring.

    Args:
        values: Tensor of continuous values
        num_bins: Number of bins to create

    Returns:
        Tensor of bin indices (0 to num_bins-1)
    """
    # Use quantile-based binning for balanced categories
    # Create linspace on the same device as values
    quantile_levels = torch.linspace(0, 1, num_bins + 1, device=values.device)
    quantiles = torch.quantile(values, quantile_levels)
    bins = torch.searchsorted(quantiles[1:-1].contiguous(), values.contiguous())
    return bins


def log_multiple_embedding_views(writer, model, data_loader, epoch, max_samples=1000):
    """
    Log multiple views of embeddings with different coloring schemes.

    This creates three separate visualizations in TensorBoard:
    1. Colored by time (batch index)
    2. Colored by volatility
    3. Colored by price change direction
    """
    for metadata_type in ['time', 'volatility', 'price_change']:
        log_embeddings_to_tensorboard(
            writer, model, data_loader, epoch,
            max_samples=max_samples,
            metadata_type=metadata_type
        )