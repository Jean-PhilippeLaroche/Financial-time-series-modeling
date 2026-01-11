import torch
import logging
from typing import Dict, List, Tuple


def check_activation_health(activations: Dict[str, torch.Tensor],
                            epoch: int,
                            num_layers: int,
                            text=True) -> Tuple[bool, List[str]]:
    """
    Automatically check if activation values are healthy or problematic.

    This function validates activation distributions against expected ranges
    and detects common training issues like exploding/vanishing activations,
    dead neurons, and saturation.

    Args:
        activations: Dictionary of activation tensors from model forward pass
        epoch: Current epoch number
        num_layers: Total number of transformer layers in the model
        text: display or not the console text, default is True

    Returns:
        Tuple of (is_healthy: bool, warnings: List[str])
        - is_healthy: True if all checks pass, False if critical issues detected
        - warnings: List of warning/error messages describing any issues
    """

    warnings = []
    is_healthy = True
    is_warmup = epoch <= 3  # First 3 epochs are warmup period -> stricter after

    # =========================================================================
    # HEALTHY RANGES FOR EACH ACTIVATION TYPE
    # =========================================================================
    # These ranges are based on typical transformer behavior with normalized data
    # that I could find online -> TODO: adjust values based on future training

    HEALTHY_RANGES = {
        'input_projection': {
            'mean': (-0.5, 0.5),
            'std': (0.5, 2.0),
            'abs_max': 5.0
        },
        'after_pos_encoding': {
            'mean': (-0.5, 0.5),
            'std': (0.5, 2.5),
            'abs_max': 6.0
        },
        'concat_pool': {
            'mean': (-0.5, 0.5),
            'std': (1.0, 5.0),
            'abs_max': 15.0
        }
    }

    # Dynamic ranges for transformer layers (scale with layer depth)
    def get_layer_ranges(layer_idx: int, total_layers: int, is_ffn: bool = False):
        """
        Get expected ranges for a specific transformer layer.
        Later layers are allowed wider distributions.
        """
        # Base multiplier increases with layer depth
        layer_multiplier = 1 + (layer_idx / total_layers) * 0.5

        # FFN layers have wider distributions than attention layers
        ffn_multiplier = 1.5 if is_ffn else 1.0

        return {
            'mean': (-0.3 * layer_multiplier, 0.3 * layer_multiplier),
            'std': (0.5 * layer_multiplier * ffn_multiplier,
                    (2.0 + layer_idx) * ffn_multiplier),
            'abs_max': (8.0 + layer_idx * 2) * ffn_multiplier
        }


    def get_stats(tensor: torch.Tensor) -> Dict[str, float]:
        """Extract statistics from activation tensor."""
        flat = tensor.view(-1)
        return {
            'mean': flat.mean().item(),
            'std': flat.std().item(),
            'min': flat.min().item(),
            'max': flat.max().item(),
            'abs_max': flat.abs().max().item()
        }


    def check_range(value: float, expected_range: Tuple[float, float],
                    name: str, metric: str, is_critical: bool = False) -> None:
        """Check if a value is within expected range."""
        nonlocal is_healthy

        min_val, max_val = expected_range

        if value < min_val or value > max_val:
            severity = "CRITICAL" if is_critical else "WARNING"
            msg = f"{severity} - {name} {metric}: {value:.3f} (expected: {min_val:.2f} to {max_val:.2f})"
            warnings.append(msg)

            if is_critical:
                is_healthy = False
                logging.error(msg)

            # Removed warning logs
            #else:
                #logging.warning(msg)


    def detect_dead_neurons(tensor: torch.Tensor, name: str, threshold: float = 0.01) -> None:
        """Detect if too many neurons are inactive (dead)."""
        nonlocal is_healthy

        flat = tensor.view(-1).abs()
        dead_ratio = (flat < threshold).float().mean().item()

        # Critical if >70% dead, warning if >50%
        if dead_ratio > 0.7:
            msg = f"CRITICAL - {name}: {dead_ratio * 100:.1f}% dead neurons (threshold: {threshold})"
            warnings.append(msg)
            is_healthy = False
            logging.error(msg)
        elif dead_ratio > 0.5:
            msg = f"WARNING - {name}: {dead_ratio * 100:.1f}% dead neurons (threshold: {threshold})"
            warnings.append(msg)
            # Removed warning logs
            #logging.warning(msg)


    def detect_saturation(tensor: torch.Tensor, name: str, threshold: float = 10.0) -> None:
        """Detect if activations are saturating at extreme values."""
        flat = tensor.view(-1).abs()
        saturated_ratio = (flat > threshold).float().mean().item()

        if saturated_ratio > 0.1:  # >10% saturated is problematic
            msg = f"WARNING - {name}: {saturated_ratio * 100:.1f}% saturated (>{threshold})"
            warnings.append(msg)

            # Removed warning logs
            #logging.warning(msg)

    def check_nan_inf(tensor: torch.Tensor, name: str) -> bool:
        """Check for NaN or Inf values (critical error)."""
        nonlocal is_healthy

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            msg = f"CRITICAL - {name}: Contains {'NaN' if has_nan else 'Inf'} values"
            warnings.append(msg)
            is_healthy = False
            logging.error(msg)
            return True
        return False

    def check_layer_progression(layer_stats: List[Dict], layer_type: str) -> None:
        """Check if activation std grows reasonably across layers."""
        nonlocal is_healthy

        if len(layer_stats) < 2:
            return

        for i in range(1, len(layer_stats)):
            prev_std = layer_stats[i - 1]['std']
            curr_std = layer_stats[i]['std']

            # Check for exponential explosion (>3x growth)
            if curr_std > prev_std * 3.0 and prev_std > 0.1:
                msg = (f"CRITICAL - {layer_type} explosion: "
                       f"Layer {i - 1} std={prev_std:.2f} → Layer {i} std={curr_std:.2f} "
                       f"({curr_std / prev_std:.1f}x increase)")
                warnings.append(msg)
                is_healthy = False
                logging.error(msg)

            # Check for vanishing (>10x decrease)
            elif curr_std < prev_std / 10.0 and curr_std < 0.1:
                msg = (f"WARNING - {layer_type} vanishing: "
                       f"Layer {i - 1} std={prev_std:.2f} → Layer {i} std={curr_std:.2f}")
                warnings.append(msg)

                # Removed warning logs
                #logging.warning(msg)

    # =========================================================================
    # RUN CHECKS
    # =========================================================================

    layer_attn_stats = []
    layer_ffn_stats = []

    for name, activation in activations.items():
        # Skip NaN/Inf check first (critical)
        if check_nan_inf(activation, name):
            continue  # Skip other checks if NaN/Inf detected

        stats = get_stats(activation)

        # =====================================================================
        # 1. CHECK INPUT PROJECTION
        # =====================================================================
        if name == 'input_projection':
            ranges = HEALTHY_RANGES['input_projection']

            # Mean should be near zero (critical after warmup)
            check_range(stats['mean'], ranges['mean'], name, 'mean',
                        is_critical=not is_warmup)

            # Std should be reasonable
            check_range(stats['std'], ranges['std'], name, 'std',
                        is_critical=False)

            # Check for extreme values
            if stats['abs_max'] > ranges['abs_max']:
                msg = f"WARNING - {name}: max absolute value {stats['abs_max']:.2f} > {ranges['abs_max']}"
                warnings.append(msg)

                #logging.warning(msg)

            # Check for dead neurons (less strict for input)
            detect_dead_neurons(activation, name, threshold=0.001)

        # =====================================================================
        # 2. CHECK POSITIONAL ENCODING
        # =====================================================================
        elif name == 'after_pos_encoding':
            ranges = HEALTHY_RANGES['after_pos_encoding']

            check_range(stats['mean'], ranges['mean'], name, 'mean',
                        is_critical=False)
            check_range(stats['std'], ranges['std'], name, 'std',
                        is_critical=False)

            if stats['abs_max'] > ranges['abs_max']:
                msg = f"WARNING - {name}: max absolute value {stats['abs_max']:.2f} > {ranges['abs_max']}"
                warnings.append(msg)
                #logging.warning(msg)

        # =====================================================================
        # 3. CHECK TRANSFORMER LAYERS
        # =====================================================================
        elif 'layer_' in name:
            # Parse layer index
            try:
                layer_idx = int(name.split('_')[1])
            except (IndexError, ValueError):
                continue

            is_ffn = 'ffn' in name
            ranges = get_layer_ranges(layer_idx, num_layers, is_ffn=is_ffn)

            # Store stats for progression check
            if is_ffn:
                layer_ffn_stats.append(stats)
            else:
                layer_attn_stats.append(stats)

            # Check mean (should stay near zero)
            check_range(stats['mean'], ranges['mean'], name, 'mean',
                        is_critical=False)

            # Check std (critical if too extreme)
            min_std, max_std = ranges['std']
            if stats['std'] < min_std / 2:  # Severely vanishing
                msg = f"CRITICAL - {name}: std={stats['std']:.3f} too low (vanishing gradients likely)"
                warnings.append(msg)
                is_healthy = False
                logging.error(msg)
            elif stats['std'] > max_std * 2:  # Severely exploding
                msg = f"CRITICAL - {name}: std={stats['std']:.3f} too high (exploding activations)"
                warnings.append(msg)
                is_healthy = False
                logging.error(msg)
            else:
                check_range(stats['std'], ranges['std'], name, 'std',
                            is_critical=False)

            # Check for extreme values
            if stats['abs_max'] > ranges['abs_max'] * 1.5:
                msg = f"WARNING - {name}: max absolute value {stats['abs_max']:.2f} > {ranges['abs_max']}"
                warnings.append(msg)
                #logging.warning(msg)

            # Dead neuron detection (stricter for deeper layers)
            threshold = 0.01 if layer_idx == 0 else 0.05
            detect_dead_neurons(activation, name, threshold=threshold)

            # Saturation detection
            detect_saturation(activation, name, threshold=10.0 + layer_idx * 2)

        # =====================================================================
        # 4. CHECK POOLED REPRESENTATION
        # =====================================================================
        elif name == 'concat_pool':
            ranges = HEALTHY_RANGES['concat_pool']

            check_range(stats['mean'], ranges['mean'], name, 'mean',
                        is_critical=False)
            check_range(stats['std'], ranges['std'], name, 'std',
                        is_critical=False)

            # Final representation before output
            if stats['abs_max'] > ranges['abs_max'] * 2:
                msg = f"WARNING - {name}: max absolute value {stats['abs_max']:.2f} very high"
                warnings.append(msg)
                #logging.warning(msg)

            # Check if pooling is working (std too low = not learning)
            if stats['std'] < 0.5 and not is_warmup:
                msg = f"WARNING - {name}: std={stats['std']:.3f} too low (model may not be learning)"
                warnings.append(msg)
                #logging.warning(msg)

    # =========================================================================
    # 5. CHECK LAYER-TO-LAYER PROGRESSION
    # =========================================================================
    if len(layer_attn_stats) > 1:
        check_layer_progression(layer_attn_stats, "Attention")

    if len(layer_ffn_stats) > 1:
        check_layer_progression(layer_ffn_stats, "FFN")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    # Colors
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    if text:
        if is_healthy and len(warnings) == 0:
            logging.info(f"{GREEN}Epoch {epoch}: All activation checks passed, model is healthy{RESET}")
        elif is_healthy and len(warnings) > 0:
            logging.info(f"{GREEN}Epoch {epoch}: Model is healthy with {len(warnings)} minor warnings{RESET}")
        else:
            logging.error(f"{RED}Epoch {epoch}: Critical activation issues detected, ({len(warnings)} total issues){RESET}")

    return is_healthy, warnings


def log_activation_health_to_tensorboard(writer, activations: Dict[str, torch.Tensor],
                                         epoch: int, num_layers: int):
    """
    Run health check and log results to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        activations: Dictionary of activation tensors
        epoch: Current epoch number
        num_layers: Total number of transformer layers
    """
    if writer is None:
        return

    is_healthy, warnings = check_activation_health(activations, epoch, num_layers, False)

    # Log health status as scalar (1.0 = healthy, 0.0 = unhealthy)
    writer.add_scalar('ActivationHealth/is_healthy',
                      1.0 if is_healthy else 0.0, epoch)

    # Log number of warnings
    writer.add_scalar('ActivationHealth/num_warnings',
                      len(warnings), epoch)

    # Log warnings as text
    if len(warnings) > 0:
        warning_text = "\n".join(warnings)
        writer.add_text('ActivationHealth/warnings', warning_text, epoch)