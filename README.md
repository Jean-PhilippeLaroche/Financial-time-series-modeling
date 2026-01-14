# Transformer-Based Intraday Trading Model
This project explores the use of Transformer architectures for modeling high-frequency financial time series and evaluating intraday trading strategies.
The model is trained on five years of historical one-minute market data, enriched with technical indicators such as RSI, MACD, and moving averages, and structured into sliding-window sequences to preserve temporal dynamics.
The pipeline leverages a pooling-based Transformer to capture both global trends and strong local signals within each time window, while enabling efficient parallel training compared to recurrent models. 
Beyond predictive accuracy, the project emphasizes interpretability through attention map visualizations and realism through a backtesting framework that incorporates transaction costs and portfolio evolution.
The full end-to-end pipeline from data ingestion and feature engineering to model training, visualization, and trading evaluation is documented in an Excalidraw diagram.
- https://excalidraw.com/#json=4mQ57WtX6dZF32uV4L1DO,EDTs6uWyswa4nHoszKC7bw


# Path
- No hardcoded absolute paths for exporting(like C:/Users/...).
os.path.join(BASE_DIR, ...) so project works on any machine
- Example:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")

**Train.py**
- if __name__ == "__main__" tests run on dummy data.
- Useful for visualisation of tensorboard metrics change or training changes


# Environment management
**Export dependencies:**
- pip freeze > requirements.txt

**On other machines, install:**
- pip install -r requirements.txt

**PyTorch version works with cpu, not cuda so need to install it manually:**
- pip3 install torch torchvision torchaudio --index url https://download.pytorch.org/whl/cu126

***Project using Python version 3.13.3. No other version compatible.***


# Logging & outputs
- All logs, models, and outputs are inside the project folder (e.g., logs/ or models/).
- Model data tracked using Tensorboard during training, running on localhost.
- Attention heatmaps tracked using Dash, running on localhost


# Version control
- Using Git to keep the code synced.
- Raw data tracked as well as the SQLite database containing AAPL and MSFT.


# How top open Tensorboard log files under runs folder:
1. tensorboard --logdir=runs indide project .venv terminal
2. go to browser and type http://localhost:6006
Note: added automatic opening, use manual way if needed shouldn't be the case


# Run Commands
**Larger model for better performance:**
- python main.py --ticker MSFT --window 120 --epochs 80 --batch 256 --lr 3e-4 --d_model 256 --nhead 8 --num_layers 6 --dim_feedforward 1024 --dropout 0.1 --threshold 0.02 --transaction_cost 0.0015 --max_norm 1.0 --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --patience 20

**Smaller model for faster training:**
- python main.py --ticker MSFT --window 20 --epochs 20 --batch 128 --lr 1e-4 --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --dropout 0.1 --threshold 0.02 --transaction_cost 0.0015 --max_norm 1.0 --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --patience 20 --no_gradient_clipping

**Visualization commands:**
1. --no_viz: no plotting at the end
2. --model_interpretation: calls main_interpretation, export weights to a .json file
and the parameters to a .csv file from best_model.pth

**Command for multi ticker history price download for raw data:**
- python scripts/download_data_polygon.py --tickers AAPL MSFT AMZN JPM BAC XOM CAT WMT KO TSLA --start 2020-11-01 --end 2025-11-01 --interval 1m --output_dir data/raw


# Ticker choice
- If AAPL or MSFT is selected as the ticker, the data will be automatically loaded from the
SQLite database. A
- MZN, BAC, CAT, JPM, KO, TSLA, WMT and XOM will be loaded from the .csv files


# Hypertuning parameters

- python hyperparameter_tuning.py --ticker MSFT --n_trials 50

**CAUTION:**
- dim_feedforward must be d_model * 4
- d_model must be divisible by nhead

**List of parameters:**
-  --ticker MSFT
-  --window 120
-  --epochs 80
-  --batch 256
-  --lr 3e-4
-  --d_model 256
-  --nhead 8
-  --num_layers 6
-  --dim_feedforward 1024
-  --dropout 0.1
-  --threshold 0.02
-  --transaction_cost 0.0015
-  --max_norm 1.0
-  --lr_scheduler_patience 6
-  --lr_scheduler_factor 0.5
-  --patience 20

# Embedding Space Visualization guide (TensorBoard projector)
The Projector tab shows how the model internally represents time series data in high-dimensional space. Each training run logs embeddings every 5 epochs.

**What you're seeing:**
- **Each dot** = one 60-day stock price window
- **Position in 3D** = learned representation (256-dimensional, reduced to 3D via t-SNE/PCA)
- **Colors** = three views available in dropdown:
  - `price_change`: Bins 0-4 represent strong bearish → strong bullish patterns
  - `volatility`: Bins 0-4 represent low → high market volatility
  - `time`: Sequential batches through the validation set

**Good signs (model is learning):**
- Clear cluster separation between bins (e.g., bullish/bearish patterns form distinct groups)
- Similar market conditions cluster together
- High/low volatility periods occupy different regions
- Clusters tighten and separate more clearly as epochs progress

**Warning signs:**
- Random scatter with no structure (model hasn't learned meaningful patterns)
- All colors mixed together (model can't distinguish different market regimes)
- Clusters don't improve from epoch 5 → 10 → 15 (learning has plateaued)

**How to use:**
1. Select an epoch from the dropdown (e.g., `embeddings/price_change_epoch0050`)
2. Switch projection method: t-SNE (best for clusters) or PCA (faster, preserves variance)
3. Compare different epochs in separate browser tabs to watch learning progress
4. Rotate the 3D view to explore cluster structure from different angles


# Activation histograms in Tensorboard guide
**Input Projection**
- Healthy Range:
Mean: -0.5 to +0.5
Std: 0.5 to 2.0
Min/Max: -5 to +5

**After Positional Encoding**
- Healthy Range:
Mean: -0.5 to +0.5
Std: 0.5 to 2.5
Min/Max: -6 to +6

**After Attention Layers (layer_N_after_attn)**
- Healthy Range for Layer 0:
Mean: -0.3 to +0.3
Std: 0.5 to 2.0
Min/Max: -8 to +8

- Healthy Range for Layer 1:
Mean: -0.5 to +0.5
Std: 0.7 to 3.0
Min/Max: -10 to +10

- Healthy Range for Layer 2:
Mean: -0.7 to +0.7
Std: 1.0 to 4.0
Min/Max: -12 to +12

**After Feed-Forward Networks (layer_N_after_ffn)**
- Healthy Range for Layer 0:
Mean: -0.3 to +0.3
Std: 0.8 to 3.0
Min/Max: -10 to +10

- Healthy Range for Layer 1:
Mean: -0.5 to +0.5
Std: 1.0 to 4.0
Min/Max: -12 to +12

- Healthy Range for Layer 2:
Mean: -0.7 to +0.7
Std: 1.5 to 5.0
Min/Max: -15 to +15

**Concatenated Pooling (concat_pool)**
- Healthy Range:
Mean: -0.5 to +0.5
Std: 1.0 to 5.0
Min/Max: -15 to +15

**What to look for**
1. Gradual distribution widening: Each layer has slightly wider distribution than previous (but not exponentially)
2. Centered distributions: Mean stays close to zero across all layers
3. Stable over epochs: Distribution parameters don't change drastically between epochs
4. Bell-shaped histograms: Most values concentrated near the mean with smooth tails
5. Consistent std growth: Standard deviation grows predictably (~20-50% per layer)

**What to avoid**
1. Mean drift: Mean moving away from zero over epochs
2. Bimodal distributions: Two distinct peaks in histogram
3. High sparsity: >50% of activations very close to zero (meaning dead neurons)
4. Saturation: Large portion of activations at extreme values (>10 or <-10)
5. Erratic changes: Distribution shifts dramatically between epochs

**When to stop training**
1. Exploding activations: Std > 20 or Min/Max beyond ±50
2. Vanishing activations: Std < 0.01 in any layer after first epoch
3. NaN or Inf values: Any non-finite values in histograms
4. Collapse: All activations converge to single value
5. Exponential growth: Each layer's std is 2x or more than previous layer

**Console outputs**
- Automatically logging warnings or critical values.
- Warnings = model is healthy (1), critical = model is not healthy (0)


# TO DO:
1. Add forward testing with the best_model.pth
2. Keep on working on the project, add more advanced functionnalities