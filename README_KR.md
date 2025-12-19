# 강한 주기성을 가진 시계열의 장기 예측 (Long Term Time Series Forecasting)

---

## 1. 소개

**강한 주기성을 가진 data** (예: 날씨 패턴, 계절별 판매량)의 시계열 예측은 Transformer 기반 모델에게 여전히 어려운 과제입니다. iTransformer나 TimeXer와 같은 최신 architecture는 long-term dependency를 잘 포착하지만, 주기적 sequence에 내재된 **intra-period (주기 내)와 inter-period (주기 간) 변동**을 모델링하는 데는 어려움을 겪습니다.

본 프로젝트는 다음을 결합한 **hybrid 예측 모델**을 제안합니다:
- **TimeXer** (NeurIPS 2024): Exogenous variable을 위해 설계된 Transformer architecture
- **특화된 Feature Extractor**: WaveNet (dilated convolution) 또는 TimesNet (2D temporal modeling)

**주요 기여:**
1. **WaXer**: Long-range dependency 포착을 위해 WaveNet의 dilated causal convolution으로 강화된 TimeXer
2. **TaXer**: 주기적 pattern 탐지를 위해 TimesNet의 FFT 기반 2D temporal modeling으로 강화된 TimeXer

다음 dataset으로 평가합니다:
- **날씨 Data**: 한국 (5개 도시) 및 글로벌 (3개 도시) 시간별 기상 시계열
- **스티커 판매**: 7년간 6개국, 3개 브랜드, 5개 제품 카테고리의 Kaggle 대회 data

---

## 2. 관련 연구

### 2.1. 장기 예측 SOTA 모델

시계열 예측을 위한 최근 state-of-the-art 모델들은 **칭화대학교** 연구자들이 개발했습니다.

- **[iTransformer](https://arxiv.org/abs/2310.06625)** — *Inverted Transformers Are Effective for Time Series Forecasting* (ICLR 2024 Spotlight)
  [[GitHub](https://github.com/thuml/iTransformer)]

- **[TimeXer](https://arxiv.org/abs/2402.19072)** — *Empowering Transformers for Time Series Forecasting with Exogenous Variables* (NeurIPS 2024)
  [[GitHub](https://github.com/thuml/TimeXer)]

이 모델들은 long-term dependency 처리에 뛰어나지만, 강한 주기성이나 구조화된 시계열에서는 여전히 어려움을 겪습니다.

---

### 2.2. WaveNet

- **논문:** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- **출판:** DeepMind, 2016

주요 구성요소:
- **Dilated Causal Convolution (DCC):**
  1D convolution에서 지수적으로 증가하는 dilation factor를 사용하여 long-term dependency를 효율적으로 포착합니다.
- **Residual & Skip Connection:**
  빠른 수렴을 촉진하고 deep architecture가 계층적 temporal feature를 학습할 수 있게 합니다.

---

### 2.3. TimesNet

- **논문:** [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186)
- **학회:** ICLR 2023

TimesNet은 시계열을 2D 표현으로 변환하여 intra-period와 inter-period 관계를 더 잘 모델링합니다:
1. Fourier Transform (FT)을 적용하여 top-K frequency component를 찾습니다.
2. 이 frequency들을 기반으로 시계열을 여러 period로 분할합니다.
3. Period들을 2D tensor로 stack합니다.
4. **2D convolution**을 사용하여 period 내부와 period 간의 temporal dependency를 포착합니다.

---

## 3. 방법론

**TimeXer의 Transformer architecture**와 특화된 **feature extractor** (WaveNet 또는 TimesNet)를 결합하여 시계열 data의 주기적 pattern을 더 잘 포착하는 hybrid 모델을 제안합니다.

### 3.1. WaXer (TimeXer + WaveNet)

**WaXer**는 TimeXer encoder 이전에 WaveNet의 dilated causal convolution을 feature extractor로 통합합니다.

**Architecture:**
```
Input Sequence
      │
      ▼
┌─────────────────────┐
│  WaveNet Feature    │  ← Dilated causal convolution
│    Extractor        │    (dilation: 1, 2, 4, 8, ...)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  TimeXer Encoder    │  ← Exogenous variable을 가진 Transformer
└─────────────────────┘
      │
      ▼
   Prediction
```

**주요 구성요소:**
- **WaveNet Feature Extractor:** 지수적으로 증가하는 dilation rate를 가진 stacked dilated convolution을 사용하여 long-range temporal dependency를 효율적으로 포착합니다.
- **Residual Connection:** 모든 layer의 feature를 Skip connection으로 집계하여 풍부한 representation을 만듭니다.
- **TimeXer 통합:** 추출된 feature는 최종 prediction을 위해 TimeXer의 inverted transformer에 입력됩니다.

---

### 3.2. TaXer (TimeXer + TimesNet)

**TaXer**는 frequency-domain 분석을 통해 주기적 pattern을 포착하기 위해 TimesNet의 2D temporal modeling을 feature extractor로 사용합니다.

**Architecture:**
```
Input Sequence
      │
      ▼
┌─────────────────────┐
│  FFT Period         │  ← Top-K frequency component 찾기
│    Detection        │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  2D Temporal        │  ← 2D로 reshape 후 Inception block 적용
│    Convolution      │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  TimeXer Encoder    │  ← Exogenous variable을 가진 Transformer
└─────────────────────┘
      │
      ▼
   Prediction
```

**주요 구성요소:**
- **FFT 기반 Period 탐지:** Data에서 주요 주기성을 자동으로 발견합니다.
- **2D Temporal Modeling:** 탐지된 period를 기반으로 1D sequence를 2D tensor로 변환하여, 2D convolution이 intra-period와 inter-period 변동을 포착할 수 있게 합니다.
- **Inception Block:** Multi-scale 2D convolution으로 다양한 temporal resolution에서 feature를 추출합니다.

---

### 3.3. 모델 비교

| 모델 | Feature Extractor | 강점 |
|------|------------------|------|
| iTransformer | 없음 | Baseline inverted transformer |
| TimeXer | 없음 | Exogenous variable 처리 |
| WaveNet | Dilated Conv | Long-range dependency |
| TimesNet | 2D Conv + FFT | 주기적 pattern 탐지 |
| **WaXer** | WaveNet → TimeXer | Long-range + Transformer |
| **TaXer** | TimesNet → TimeXer | Periodic + Transformer |

---

## 4. 실행 방법

### 4.1. 설치

```bash
# Conda 환경 생성
conda create -n taxer python=3.10
conda activate taxer

# CUDA 포함 PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Dependency 설치
pip install numpy pandas matplotlib scikit-learn
```

---

### 4.2. 학습

#### Training Script 사용 (권장)

**날씨 Dataset:**
```bash
# 사용법: ./scripts/train_weather.sh <model> <num_gpus> <dataset_type>
# dataset_type: korean 또는 global

./scripts/train_weather.sh waxer 8 korean    # 한국 날씨로 WaXer 학습
./scripts/train_weather.sh taxer 8 global    # 글로벌 날씨로 TaXer 학습
./scripts/train_weather.sh itransformer 4 korean
```

**스티커 Dataset:**
```bash
# 사용법: ./scripts/train_sticker.sh <model> <num_gpus>

./scripts/train_sticker.sh waxer 8           # 스티커 data로 WaXer 학습
./scripts/train_sticker.sh taxer 8           # 스티커 data로 TaXer 학습
```

#### 직접 학습 (Single GPU)

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

#### torchrun을 이용한 Multi-GPU 학습

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

### 4.3. 평가

모든 모델 checkpoint를 평가하고 비교 metric을 생성합니다:

```bash
# 사용법: ./scripts/eval_models.sh <dataset> <gpu_id>

./scripts/eval_models.sh korean 0    # 한국 날씨에서 모든 모델 평가
./scripts/eval_models.sh global 0    # 글로벌 날씨에서 모든 모델 평가
./scripts/eval_models.sh sticker 0   # 스티커 data에서 모든 모델 평가
./scripts/eval_models.sh all 0       # 모든 dataset 평가
```

결과는 `results/eval_results/{dataset}/`에 저장됩니다:
- `{model}_eval.csv` - Checkpoint별 metric
- `summary.json` - 모델별 최고 checkpoint

---

### 4.4. 테스트 및 예측

**테스트 (test set에서 평가):**
```bash
python main.py \
    --model waxer \
    --dataset weather \
    --city korea \
    --mode test \
    --ckpt_name waxer_e31_s512_p16
```

**예측 (autoregressive prediction):**
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

### 4.5. 사용 가능한 모델

| 모델 | 설명 | 주요 Parameter |
|------|------|----------------|
| `itransformer` | Inverted Transformer (ICLR 2024) | `d_model`, `n_heads`, `e_layers` |
| `timexer` | Exogenous variable을 가진 TimeXer (NeurIPS 2024) | `d_model`, `patch_len` |
| `wavenet` | WaveNet dilated convolution | `d_model`, `e_layers`, `wave_kernel_size` |
| `timesnet` | TimesNet 2D temporal modeling | `d_model`, `top_k`, `time_inception` |
| `waxer` | TimeXer + WaveNet feature | `wavenet_d_model`, `wavenet_layers` |
| `taxer` | TimeXer + TimesNet feature | `times_d_model`, `times_top_k`, `times_layers` |

---

### 4.6. Dataset

| Dataset | City Parameter | Input Dim | 설명 |
|---------|---------------|-----------|------|
| `sticker` | N/A | 15 | Kaggle 스티커 판매 (6개국, 3개 매장, 5개 카테고리) |
| `weather` | `korea` | 30 | 한국 날씨 (서울, 부산, 대구, 강릉, 광주) |
| `weather` | `global` | 18 | 글로벌 날씨 (베를린, LA, 뉴욕) |

---

### 4.7. Checkpoint 구조

Checkpoint는 `checkpoints/{dataset}/{model}_v{version}/`에 저장됩니다:

```
checkpoints/korean/waxer_v1/
├── args.json                    # 학습 argument
├── train.log                    # 학습 log
├── waxer_e15_s512_p16.pth      # Epoch 16의 checkpoint
├── waxer_e31_s512_p16.pth      # Epoch 32의 checkpoint
└── waxer_e34_s512_p16.pth      # 최종 checkpoint
```

Checkpoint 명명 형식: `{model}_e{epoch}_s{seq_len}_p{pred_len}.pth`

---

### 4.8. 주요 Hyperparameter

| Parameter | 설명 | Default |
|-----------|------|---------|
| `--seq_len` | Input sequence 길이 | 512 |
| `--pred_len` | Prediction 길이 | 16 |
| `--batch_size` | GPU당 batch size | 32 |
| `--epochs` | 학습 epoch 수 | 35 |
| `--lr` | Learning rate | 1e-4 |
| `--d_model` | Model dimension | 32 |
| `--e_layers` | Encoder layer 수 | 1 |
| `--dropout` | Dropout rate | 0.2 |

---
