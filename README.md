# Long Term Time Series Forecasting for Strong Periodic Sequence

---

## 1. Introduction

Time series forecasting for **strongly periodic data** (e.g., weather patterns, seasonal sales) remains challenging for transformer-based models. While recent architectures like iTransformer and TimeXer excel at capturing long-term dependencies, they often struggle to model the **intra-period and inter-period variations** inherent in periodic sequences.

This project proposes **hybrid forecasting models** that combine:
- **TimeXer** (NeurIPS 2024): A transformer architecture designed for time series with exogenous variables
- **Specialized Feature Extractors**: WaveNet (dilated convolutions) or TimesNet (2D temporal modeling)

**Key Contributions:**
1. **WaXer**: TimeXer enhanced with WaveNet's dilated causal convolutions for capturing long-range dependencies
2. **TaXer**: TimeXer enhanced with TimesNet's FFT-based 2D temporal modeling for periodic pattern detection

We evaluate these models on:
- **Weather Data**: Korean (5 cities) and Global (3 cities) weather time series with hourly measurements
- **Sticker Sales**: Kaggle competition data with 7 years of sales across 6 countries, 3 brands, and 5 product categories

---

## 2. Related Work

### 2.1. Long-Term Forecasting SOTA Models

Recent state-of-the-art models for time series forecasting were developed by researchers from **Tsinghua University**.

- **[iTransformer](https://arxiv.org/abs/2310.06625)** — *Inverted Transformers Are Effective for Time Series Forecasting* (ICLR 2024 Spotlight)
  [[GitHub](https://github.com/thuml/iTransformer)]

- **[TimeXer](https://arxiv.org/abs/2402.19072)** — *Empowering Transformers for Time Series Forecasting with Exogenous Variables* (NeurIPS 2024)
  [[GitHub](https://github.com/thuml/TimeXer)]

These models excel in handling long-term dependencies but still struggle with strongly periodic or structured time series.

---

### 2.2. WaveNet

- **Paper:** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- **Publisher:** DeepMind, 2016

Key components:
- **Dilated Causal Convolution (DCC):**
  Uses exponentially increasing dilation factors in 1D convolutions to capture long-term dependencies efficiently.
- **Residual & Skip Connections:**
  Facilitates faster convergence and enables deep architectures to learn hierarchical temporal features.

---

### 2.3. TimesNet

- **Paper:** [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186)
- **Conference:** ICLR 2023

TimesNet converts time series into a 2D representation to better model intra-period and inter-period relations:
1. Applies Fourier Transform (FT) to find the top-K frequency components.
2. Splits the time series into several periods based on these frequencies.
3. Stacks the periods into a 2D tensor.
4. Uses **2D convolutions** to capture temporal dependencies within and across periods.

---

## 3. Method

We propose hybrid models that combine **TimeXer's transformer architecture** with specialized **feature extractors** (WaveNet or TimesNet) to better capture periodic patterns in time series data.

### 3.1. WaXer (TimeXer + WaveNet)

**WaXer** integrates WaveNet's dilated causal convolutions as a feature extractor before the TimeXer encoder.

**Architecture:**
```
Input Sequence
      │
      ▼
┌─────────────────────┐
│  WaveNet Feature    │  ← Dilated causal convolutions
│    Extractor        │    (dilation: 1, 2, 4, 8, ...)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  TimeXer Encoder    │  ← Transformer with exogenous variables
└─────────────────────┘
      │
      ▼
   Prediction
```

**Key Components:**
- **WaveNet Feature Extractor:** Uses stacked dilated convolutions with exponentially increasing dilation rates to capture long-range temporal dependencies efficiently.
- **Residual Connections:** Skip connections aggregate features from all layers for richer representations.
- **TimeXer Integration:** Extracted features are fed into TimeXer's inverted transformer for final prediction.

---

### 3.2. TaXer (TimeXer + TimesNet)

**TaXer** uses TimesNet's 2D temporal modeling as a feature extractor to capture periodic patterns through frequency-domain analysis.

**Architecture:**
```
Input Sequence
      │
      ▼
┌─────────────────────┐
│  FFT Period         │  ← Find top-K frequency components
│    Detection        │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  2D Temporal        │  ← Reshape to 2D and apply
│    Convolutions     │    Inception blocks
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  TimeXer Encoder    │  ← Transformer with exogenous variables
└─────────────────────┘
      │
      ▼
   Prediction
```

**Key Components:**
- **FFT-based Period Detection:** Automatically discovers dominant periodicities in the data.
- **2D Temporal Modeling:** Converts 1D sequences to 2D tensors based on detected periods, enabling 2D convolutions to capture intra-period and inter-period variations.
- **Inception Blocks:** Multi-scale 2D convolutions extract features at different temporal resolutions.

---

### 3.3. Model Comparison

| Model | Feature Extractor | Strength |
|-------|------------------|----------|
| iTransformer | None | Baseline inverted transformer |
| TimeXer | None | Handles exogenous variables |
| WaveNet | Dilated Conv | Long-range dependencies |
| TimesNet | 2D Conv + FFT | Periodic pattern detection |
| **WaXer** | WaveNet → TimeXer | Long-range + transformer |
| **TaXer** | TimesNet → TimeXer | Periodic + transformer |

---

## 4. How to Run

### 4.1. Installation

```bash
# Create conda environment
conda create -n taxer python=3.10
conda activate taxer

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas matplotlib scikit-learn
```

---

### 4.2. Training

#### Using Training Scripts (Recommended)

**Weather Dataset:**
```bash
# Usage: ./scripts/train_weather.sh <model> <num_gpus> <dataset_type>
# dataset_type: korean or global

./scripts/train_weather.sh waxer 8 korean    # Train WaXer on Korean weather
./scripts/train_weather.sh taxer 8 global    # Train TaXer on Global weather
./scripts/train_weather.sh itransformer 4 korean
```

**Sticker Dataset:**
```bash
# Usage: ./scripts/train_sticker.sh <model> <num_gpus>

./scripts/train_sticker.sh waxer 8           # Train WaXer on sticker data
./scripts/train_sticker.sh taxer 8           # Train TaXer on sticker data
```

#### Direct Training (Single GPU)

```bash
python main.py \
    --model waxer \
    --dataset weather \
    --city korea \
    --mode train \
    --epochs 35 \
    --seq_len 512 \
    --pred_len 16 \
    --batch_size 32 \
    --lr 1e-4
```

#### Multi-GPU Training with torchrun

```bash
torchrun --nproc_per_node=8 main.py \
    --model waxer \
    --dataset weather \
    --city korea \
    --mode train \
    --epochs 35 \
    --batch_size 32
```

---

### 4.3. Evaluation

Evaluate all model checkpoints and generate comparison metrics:

```bash
# Usage: ./scripts/eval_models.sh <dataset> <gpu_id>

./scripts/eval_models.sh korean 0    # Evaluate all models on Korean weather
./scripts/eval_models.sh global 0    # Evaluate all models on Global weather
./scripts/eval_models.sh sticker 0   # Evaluate all models on Sticker data
./scripts/eval_models.sh all 0       # Evaluate all datasets
```

Results are saved to `results/eval_results/{dataset}/`:
- `{model}_eval.csv` - Per-checkpoint metrics
- `summary.json` - Best checkpoint per model

---

### 4.4. Testing & Forecasting

**Testing (evaluate on test set):**
```bash
python main.py \
    --model waxer \
    --dataset weather \
    --city korea \
    --mode test \
    --ckpt_name waxer_e31_s512_p16
```

**Forecasting (autoregressive prediction):**
```bash
python main.py \
    --model waxer \
    --dataset weather \
    --city korea \
    --mode forecast \
    --forecast_len 96 \
    --bootstrapping_step 1 \
    --ckpt_name waxer_e31_s512_p16
```

---

### 4.5. Available Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `itransformer` | Inverted Transformer (ICLR 2024) | `d_model`, `n_heads`, `e_layers` |
| `timexer` | TimeXer with exogenous variables (NeurIPS 2024) | `d_model`, `patch_len` |
| `wavenet` | WaveNet dilated convolutions | `d_model`, `e_layers`, `wave_kernel_size` |
| `timesnet` | TimesNet 2D temporal modeling | `d_model`, `top_k`, `time_inception` |
| `waxer` | TimeXer + WaveNet features | `wavenet_d_model`, `wavenet_layers` |
| `taxer` | TimeXer + TimesNet features | `times_d_model`, `times_top_k`, `times_layers` |

---

### 4.6. Datasets

| Dataset | City Parameter | Input Dim | Description |
|---------|---------------|-----------|-------------|
| `sticker` | N/A | 15 | Kaggle sticker sales (6 countries, 3 stores, 5 categories) |
| `weather` | `korea` | 30 | Korean weather (Seoul, Busan, Daegu, Gangneung, Gwangju) |
| `weather` | `global` | 18 | Global weather (Berlin, LA, New York) |

---

### 4.7. Checkpoint Structure

Checkpoints are saved in `checkpoints/{dataset}/{model}_v{version}/`:

```
checkpoints/korean/waxer_v1/
├── args.json                    # Training arguments
├── train.log                    # Training logs
├── waxer_e15_s512_p16.pth      # Checkpoint at epoch 16
├── waxer_e31_s512_p16.pth      # Checkpoint at epoch 32
└── waxer_e34_s512_p16.pth      # Final checkpoint
```

Checkpoint naming format: `{model}_e{epoch}_s{seq_len}_p{pred_len}.pth`

---

### 4.8. Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--seq_len` | Input sequence length | 512 |
| `--pred_len` | Prediction length | 16 |
| `--batch_size` | Batch size per GPU | 32 |
| `--epochs` | Number of training epochs | 35 |
| `--lr` | Learning rate | 1e-4 |
| `--d_model` | Model dimension | 32 |
| `--e_layers` | Number of encoder layers | 1 |
| `--dropout` | Dropout rate | 0.2 |

---
