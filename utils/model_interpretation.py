import torch
from scripts.train import TimeSeriesTransformerPooled
from utils.data_utils import prepare_data_for_ai, add_indicators, clean_data
import logging
from utils.data_utils import load_stock_csv
import os
import numpy as np
import json


def find_file(filename, start_dir=None):
    """
    Recursively search for a file in the project directory.
    """
    if start_dir is None:
        # Assume this file lives somewhere inside the project
        start_dir = os.path.dirname(os.path.abspath(__file__))

        # Go to project root (one level above utils / module)
        start_dir = os.path.dirname(start_dir)

    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(f"{filename} not found in project starting at {start_dir}")


def model_interpretation(ticker="AAPL", train_size=0.8, window_size=20,
                         dim_feedforward=256, d_model=64, nhead=4,
                         num_layers=2, input_size=None):

    if input_size is None:

        # Using SQLite database if ticker is AAPL or MSFT
        using_sqlite = ticker in ("AAPL", "MSFT")

        df_raw = load_stock_csv(ticker)
        if df_raw is None:
            logging.error("Could not load raw CSV for ticker; exiting.")

        df_tmp = add_indicators(df_raw)
        df_tmp = clean_data(df_tmp)

        n_total = len(df_tmp)
        split_idx = int(n_total * train_size)

        X_train, y_train, scaler = prepare_data_for_ai(
                ticker=ticker,
                data_dir=None,
                feature_columns=None,
                target_column="close",
                window_size=window_size,
                SQLite=using_sqlite,
                start_idx=0,
                end_idx=split_idx
            )
        input_size = X_train.shape[2]

    else:
        input_size=input_size

    d_model = d_model
    nhead = nhead
    num_layers = num_layers
    dim_feedforward = dim_feedforward

    file_path = find_file("best_model.pth")

    model = TimeSeriesTransformerPooled(input_size=input_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
    model.load_state_dict(torch.load(file_path, map_location="cpu"))
    model.eval()

    results_file = f'model_weights.json'
    params_json = {name: p.cpu().numpy().tolist() for name, p in model.state_dict().items()}
    with open(results_file, "w") as f:
        json.dump(params_json, f, indent=2)

    print(f"Model weights saved to {results_file}")


def extract_model_parameters(num_layers=2, json_path='model_weights.json'):
    """
    Extract and organize model parameters from the JSON file.

    Args:
        json_path: Path to the model weights JSON file
        num_layers: Number of layers of the trained model
    Returns:
        Dictionary with organized model parameters
    """

    file_path = find_file("model_weights.json")

    with open(file_path, 'r') as f:
        params = json.load(f)

    # Convert lists back to numpy arrays for easier manipulation
    params_np = {k: np.array(v) for k, v in params.items()}

    # Organize parameters by component
    model_params = {
        'input_embedding': {},
        'positional_encoding': {},
        'transformer_layers': [],
        'output_layer': {}
    }

    # Extract input embedding layer
    if 'input_projection.weight' in params_np:
        model_params['input_embedding']['W_emb'] = params_np['input_projection.weight']
    if 'input_projection.bias' in params_np:
        model_params['input_embedding']['b_emb'] = params_np['input_projection.bias']

    # Extract positional encoding (if stored)
    if 'pos_encoder.pe' in params_np:
        model_params['positional_encoding']['PE'] = params_np['pos_encoder.pe']

    for layer_idx in range(num_layers):
        layer_params = {
            'layer_num': layer_idx,
            'self_attention': {},
            'feedforward': {},
            'layer_norm_1': {},
            'layer_norm_2': {}
        }

        # Self-attention weights
        # Query, Key, Value projections (in_proj contains Q, K, V concatenated)
        prefix = f'transformer_encoder.layers.{layer_idx}.self_attn'

        if f'{prefix}.in_proj_weight' in params_np:
            # Split into Q, K, V
            in_proj = params_np[f'{prefix}.in_proj_weight']
            d_model = in_proj.shape[1]
            layer_params['self_attention']['W_Q'] = in_proj[:d_model, :]
            layer_params['self_attention']['W_K'] = in_proj[d_model:2 * d_model, :]
            layer_params['self_attention']['W_V'] = in_proj[2 * d_model:, :]

        if f'{prefix}.in_proj_bias' in params_np:
            in_proj_bias = params_np[f'{prefix}.in_proj_bias']
            d_model = len(in_proj_bias) // 3
            layer_params['self_attention']['b_Q'] = in_proj_bias[:d_model]
            layer_params['self_attention']['b_K'] = in_proj_bias[d_model:2 * d_model]
            layer_params['self_attention']['b_V'] = in_proj_bias[2 * d_model:]

        if f'{prefix}.out_proj.weight' in params_np:
            layer_params['self_attention']['W_O'] = params_np[f'{prefix}.out_proj.weight']
        if f'{prefix}.out_proj.bias' in params_np:
            layer_params['self_attention']['b_O'] = params_np[f'{prefix}.out_proj.bias']

        # Feed-forward network
        prefix = f'transformer_encoder.layers.{layer_idx}'

        if f'{prefix}.linear1.weight' in params_np:
            layer_params['feedforward']['W_1'] = params_np[f'{prefix}.linear1.weight']
        if f'{prefix}.linear1.bias' in params_np:
            layer_params['feedforward']['b_1'] = params_np[f'{prefix}.linear1.bias']

        if f'{prefix}.linear2.weight' in params_np:
            layer_params['feedforward']['W_2'] = params_np[f'{prefix}.linear2.weight']
        if f'{prefix}.linear2.bias' in params_np:
            layer_params['feedforward']['b_2'] = params_np[f'{prefix}.linear2.bias']

        # Layer normalization parameters
        if f'{prefix}.norm1.weight' in params_np:
            layer_params['layer_norm_1']['gamma'] = params_np[f'{prefix}.norm1.weight']
        if f'{prefix}.norm1.bias' in params_np:
            layer_params['layer_norm_1']['beta'] = params_np[f'{prefix}.norm1.bias']

        if f'{prefix}.norm2.weight' in params_np:
            layer_params['layer_norm_2']['gamma'] = params_np[f'{prefix}.norm2.weight']
        if f'{prefix}.norm2.bias' in params_np:
            layer_params['layer_norm_2']['beta'] = params_np[f'{prefix}.norm2.bias']

        model_params['transformer_layers'].append(layer_params)

    # Extract output layer
    if 'fc_out.weight' in params_np:
        model_params['output_layer']['W_out'] = params_np['fc_out.weight']
    if 'fc_out.bias' in params_np:
        model_params['output_layer']['b_out'] = params_np['fc_out.bias']

    return model_params, params_np


def print_parameter_shapes(model_params):
    """
    Print the shapes of all extracted parameters for verification.
    """
    print("=" * 70)
    print("MODEL PARAMETER SHAPES")
    print("=" * 70)

    print("\n1. INPUT EMBEDDING:")
    for name, param in model_params['input_embedding'].items():
        print(f"   {name}: {param.shape}")

    if model_params['positional_encoding']:
        print("\n2. POSITIONAL ENCODING:")
        for name, param in model_params['positional_encoding'].items():
            print(f"   {name}: {param.shape}")

    print("\n3. TRANSFORMER LAYERS:")
    for i, layer in enumerate(model_params['transformer_layers']):
        print(f"\n   Layer {i}:")
        print(f"   - Self-Attention:")
        for name, param in layer['self_attention'].items():
            print(f"      {name}: {param.shape}")
        print(f"   - Feed-Forward:")
        for name, param in layer['feedforward'].items():
            print(f"      {name}: {param.shape}")
        print(f"   - Layer Norm 1:")
        for name, param in layer['layer_norm_1'].items():
            print(f"      {name}: {param.shape}")
        print(f"   - Layer Norm 2:")
        for name, param in layer['layer_norm_2'].items():
            print(f"      {name}: {param.shape}")

    print("\n4. OUTPUT LAYER:")
    for name, param in model_params['output_layer'].items():
        print(f"   {name}: {param.shape}")
    print("=" * 70)


def export_parameters_to_latex(model_params, output_file='model_params.tex'):
    """
    Export parameter dimensions to LaTeX format.
    """
    with open(output_file, 'w') as f:
        f.write("% Model Parameters\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lll}\n")
        f.write("\\hline\n")
        f.write("Component & Parameter & Dimension \\\\\n")
        f.write("\\hline\n")

        # Input embedding
        for name, param in model_params['input_embedding'].items():
            shape_str = ' \\times '.join(map(str, param.shape))
            f.write(f"Input Embedding & ${name}$ & $\\mathbb{{R}}^{{{shape_str}}}$ \\\\\n")

        # Transformer layers
        for i, layer in enumerate(model_params['transformer_layers']):
            f.write(f"\\hline\n")
            f.write(f"\\multicolumn{{3}}{{c}}{{Transformer Layer {i}}} \\\\\n")
            f.write(f"\\hline\n")

            for name, param in layer['self_attention'].items():
                shape_str = ' \\times '.join(map(str, param.shape))
                f.write(f"Self-Attention & ${name}^{{({i})}}$ & $\\mathbb{{R}}^{{{shape_str}}}$ \\\\\n")

            for name, param in layer['feedforward'].items():
                shape_str = ' \\times '.join(map(str, param.shape))
                f.write(f"Feed-Forward & ${name}^{{({i})}}$ & $\\mathbb{{R}}^{{{shape_str}}}$ \\\\\n")

        # Output layer
        f.write("\\hline\n")
        for name, param in model_params['output_layer'].items():
            shape_str = ' \\times '.join(map(str, param.shape))
            f.write(f"Output Layer & ${name}$ & $\\mathbb{{R}}^{{{shape_str}}}$ \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Model Parameter Dimensions}\n")
        f.write("\\label{tab:model_params}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table exported to {output_file}")


def export_parameters_to_csv(model_params, output_file='model_params.csv'):
    """
    Export parameter dimensions to CSV format.
    """
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Component', 'Parameter', 'Dimension'])

        # Input embedding
        for name, param in model_params['input_embedding'].items():
            shape_str = '×'.join(map(str, param.shape))
            writer.writerow(['Input Embedding', name, f'ℝ^{shape_str}'])

        # Transformer layers
        for i, layer in enumerate(model_params['transformer_layers']):
            writer.writerow(['', '', ''])  # Separator
            writer.writerow([f'Transformer Layer {i}', '', ''])

            for name, param in layer['self_attention'].items():
                shape_str = '×'.join(map(str, param.shape))
                writer.writerow(['Self-Attention', f'{name}^({i})', f'ℝ^{shape_str}'])

            for name, param in layer['feedforward'].items():
                shape_str = '×'.join(map(str, param.shape))
                writer.writerow(['Feed-Forward', f'{name}^({i})', f'ℝ^{shape_str}'])

        # Output layer
        writer.writerow(['', '', ''])  # Separator
        for name, param in model_params['output_layer'].items():
            shape_str = '×'.join(map(str, param.shape))
            writer.writerow(['Output Layer', name, f'ℝ^{shape_str}'])

    print(f"\nCSV table exported to {output_file}")


def export_parameters_to_markdown(model_params, output_file='model_params.md'):
    """
    Export parameter dimensions to Markdown format.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Model Parameter Dimensions\n\n")
        f.write("| Component | Parameter | Dimension |\n")
        f.write("|-----------|-----------|----------|\n")

        # Input embedding
        for name, param in model_params['input_embedding'].items():
            shape_str = '×'.join(map(str, param.shape))
            f.write(f"| Input Embedding | {name} | ℝ^{shape_str} |\n")

        # Transformer layers
        for i, layer in enumerate(model_params['transformer_layers']):
            f.write(f"| **Transformer Layer {i}** | | |\n")

            for name, param in layer['self_attention'].items():
                shape_str = '×'.join(map(str, param.shape))
                f.write(f"| Self-Attention | {name}^({i}) | ℝ^{shape_str} |\n")

            for name, param in layer['feedforward'].items():
                shape_str = '×'.join(map(str, param.shape))
                f.write(f"| Feed-Forward | {name}^({i}) | ℝ^{shape_str} |\n")

        # Output layer
        for name, param in model_params['output_layer'].items():
            shape_str = '×'.join(map(str, param.shape))
            f.write(f"| Output Layer | {name} | ℝ^{shape_str} |\n")

    print(f"\nMarkdown table exported to {output_file}")


def main_interpretation(ticker="AAPL", train_size=0.8, window_size=20,
                        dim_feedforward=256, d_model=64, nhead=4, num_layers=2,
                        file="csv", input_size=None):
    """
    Convenience function to run full model interpretation.
    """

    # Creating model_weights.json
    model_interpretation(ticker, train_size, window_size,
                         dim_feedforward, d_model, nhead, num_layers,
                         input_size)

    # Extract parameters
    model_params, raw_params = extract_model_parameters(num_layers, 'model_weights.json')

    # Print parameter shapes
    print_parameter_shapes(model_params)

    # Export to desired file
    if file == "csv":
        export_parameters_to_csv(model_params=model_params)
    elif file == "markdown":
        export_parameters_to_markdown(model_params=model_params)
    elif file == "LaTeX":
        export_parameters_to_latex(model_params=model_params)
    else:
        print(f"{file} incorrect or unsupported")


if __name__ == "__main__":

    main_interpretation(file="csv")