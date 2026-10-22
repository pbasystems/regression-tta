# Test-Time Adaptation for Remaining Useful Life (RUL) Prediction by Subspace Alignment

This repository adapts and extends **[Test-time Adaptation for Regression by Subspace Alignment (SSA, ICLR 2025)](https://openreview.net/forum?id=SXtl7NRyE5)** from static image/tabular regression to **Remaining Useful Life (RUL) prediction** on time-series predictive maintenance benchmarks (**NASA C-MAPSS** turbofan engine degradation and **IEEE PHM 2012** PRONOSTIA bearings).

[[Original Paper (OpenReview)](https://openreview.net/forum?id=SXtl7NRyE5)] [[arXiv](https://arxiv.org/abs/2410.03263)] [[Poster](https://kzkadc.github.io/assets/publications/ICLR2025_poster.pdf)]

<details><summary><strong>Original Paper Abstract (ICLR 2025)</strong></summary>
This paper investigates <em>test-time adaptation (TTA) for regression</em>, where a regression model pre-trained in a source domain is adapted to an unknown target distribution with unlabeled target data. Although regression is one of the fundamental tasks in machine learning, most of the existing TTA methods have classification-specific designs, which assume that models output class-categorical predictions, whereas regression models typically output only single scalar values. To enable TTA for regression, we adopt a feature alignment approach, which aligns the feature distributions between the source and target domains to mitigate the domain gap. However, we found that naive feature alignment employed in existing TTA methods for classification is ineffective or even worse for regression because the features are distributed in a small subspace and many of the raw feature dimensions have little significance to the output. For an effective feature alignment in TTA for regression, we propose <em>Significant-subspace Alignment (SSA)</em>. SSA consists of two components: subspace detection and dimension weighting. Subspace detection finds the feature subspace that is representative and significant to the output. Then, the feature alignment is performed in the subspace during TTA. Meanwhile, dimension weighting raises the importance of the dimensions of the feature subspace that have greater significance to the output. We experimentally show that SSA outperforms various baselines on real-world datasets.
</details>

![overview](overview.png)

---

## Key Modifications & Extensions for RUL Prediction

1. **Sequential & Raw-Waveform RUL Architectures (`model.py`)**:
   - `lstm_tta` (`LSTMTTARegressor`): Hybrid architecture combining the `LSTM-DANN` temporal feature extractor (`LSTM -> Dropout -> Linear(64) -> ReLU`) with `BatchNorm1d(64)` and a single linear regression head (`Linear(64, 1)`) required for closed-form **Significant-subspace Alignment (SSA)** dimension weighting.
   - `lstm_dann` (`LSTMDANNRegressor`): Exact feature extractor and regressor head from the *LSTM-DANN* domain adaptation baseline.
   - `lstm` (`LSTMRegressor`): Multi-layer LSTM with `BatchNorm1d` and linear RUL head.
   - `cnn1d` (`CNN1DRegressor`): 1D Convolutional feature extractor + `BatchNorm1d` + linear head for high-frequency raw vibration waveforms `(2560, 2)` in **IEEE PHM 2012**.
   - CuDNN RNN test-time backward pass support in `tta/tta_engine.py` when adapting recurrent backbones in evaluation mode (`train_mode: False`).

2. **Predictive Maintenance Datasets (`dataset/`)**:
   - **NASA C-MAPSS (`dataset/cmapss.py`)**: Sliding-window loader (`window_size=30`, piecewise-linear RUL cap `max_rul=125.0`, engine-level train/val splitting, per-domain min-max normalization) supporting all 12 cross-domain transfer pairs across `FD001`, `FD002`, `FD003`, and `FD004`.
   - **IEEE PHM 2012 Bearing Dataset (`dataset/phm2012.py`)**: Raw vibration waveform `(2560, 2)` loader with automatic `.npy` binary caching across Operating Conditions 1, 2, and 3.

3. **Benchmarking & Evaluation Pipelines**:
   - **Realistic Deployment Benchmark (`run_realistic_cmapss_benchmark.py`)**: Evaluates a single fixed architecture (`lstm_tta`) across all 12 C-MAPSS transfer tasks using a single dataset-general source training recipe, out-of-the-box Default TTA, and a single dataset-general calibrated TTA configuration (both Online streaming and Offline 1-pass).
   - **Full Per-Task Optuna Benchmark (`run_all_cmapss_benchmarks.py`)**: Exhaustive 12-task benchmark with Optuna hyperparameter search and Weights & Biases (`wandb`) logging.
   - **Plotting & Feature Space Analysis (`plot_realistic_benchmark_results.py`, `plot_benchmark_results.py`, `compare_dann_vs_tta_features.py`)**: Generates per-method and combined RMSE/MAE bar charts against the published *LSTM-DANN* baseline, plus side-by-side t-SNE/PCA feature subspace comparisons.

---

## 0. Environment & Dataset Setup (with `uv`)

### Install Dependencies with `uv`
This project uses **[uv](https://docs.astral.sh/uv/)** for fast, reproducible Python environment management (instead of Docker):

```bash
# Install dependencies and create the virtual environment
uv sync
```

### Configure Weights & Biases (`wandb`) via `.env`
Experiment tracking and Optuna trial logging use **Weights & Biases (`wandb`)**, which automatically loads credentials from a `.env` file via `python-dotenv`.
1. Copy the provided `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your `WANDB_API_KEY` inside `.env`:
   ```env
   WANDB_API_KEY=your_wandb_api_key_here
   ```
*(If you want to run benchmarks without W&B logging, pass `--no-wandb` to the benchmark scripts.)*

### Download the NASA C-MAPSS Dataset
1. Download the official NASA C-MAPSS dataset archive from:
   - **https://data.nasa.gov/docs/legacy/CMAPSSData.zip**
2. Extract `CMAPSSData.zip` (which contains `train_FD001.txt`–`train_FD004.txt`, `test_FD001.txt`–`test_FD004.txt`, and `RUL_FD001.txt`–`RUL_FD004.txt`).
3. Update `CMAPSS_PATH` (and any other dataset paths) in `dataset/dataset_config.py`:

```python
# dataset/dataset_config.py
CMAPSS_PATH = r"/path/to/CMAPSSData"
PHM2012_PATH = r"/path/to/ieee-phm-2012-data-challenge-dataset-master"
```

---

## 1. Realistic C-MAPSS RUL Benchmark (Recommended)

In real-world predictive maintenance deployment, hyperparameter tuning cannot be performed on each target test fleet at inference time. `run_realistic_cmapss_benchmark.py` runs the complete 12-task benchmark (`FD001` $\leftrightarrow$ `FD004`) under a realistic protocol:
1. **Source-Only & Target-Only (Oracle)** baselines.
2. **General Source Pre-Training**: Tunes one general set of training hyperparameters on source validation splits (`FD001`–`FD004`) and trains one source model per domain.
3. **Default TTA (Online & Offline)**: Evaluates the original paper's default SSA settings out-of-the-box with zero TTA tuning.
4. **General Dataset-Wide TTA (Online & Offline)**: Tunes TTA hyperparameters once on a single calibration pair (`FD001 -> train_FD003`) and applies them frozen across all 12 tasks.

```bash
# Run the full 12-task realistic C-MAPSS benchmark
uv run run_realistic_cmapss_benchmark.py --model-type lstm_tta -o result/benchmark_realistic

# Generate individual and combined RMSE & MAE comparison plots
uv run plot_realistic_benchmark_results.py \
  --csv result/benchmark_realistic/summary_all_metrics.csv \
  --paper-csv result/result/benchmark_all/lstm_dann_paper_results.csv \
  -o result/benchmark_realistic/plots
```

### Compare Learned Feature Spaces (`LSTM-DANN` vs. `SSA`)
To train/load both the original `LSTM-DANN` model and the adapted `SSA` model on a transfer task (e.g., `FD001 -> FD002`), extract their 64-D test feature vectors, and plot side-by-side t-SNE and PCA domain alignments:

```bash
uv run compare_dann_vs_tta_features.py \
  --source-fd FD001 \
  --target-fd FD002 \
  --realistic-root result/benchmark_realistic \
  -o result/feature_comparison_fd001_to_fd002
```

---

## 2. Step-by-Step Single-Task RUL Workflow

You can also run the 3-stage pipeline (Source Training $\rightarrow$ Feature Statistics $\rightarrow$ Test-Time Adaptation) for an individual C-MAPSS task using the YAML configs in `configs/`:

### Step 1: Train the Source RUL Model
```bash
uv run train_source.py -c configs/train_source/cmapss.yaml -o result/source/cmapss
```

### Step 2: Compute Source Feature Subspace Statistics
```bash
uv run feature_stats.py -c configs/feature_stats/cmapss.yaml -o result/source/cmapss
```

### Step 3: Run Test-Time Adaptation (SSA) on Target Domain
```bash
uv run adaptation.py -c configs/tta/cmapss.yaml -o result/tta/cmapss
```

*(For original paper replication on `SVHN -> MNIST`, replace `cmapss.yaml` with `svhn.yaml`.)*

---

## 3. Full Per-Task Optuna Benchmark

To run per-task Optuna hyperparameter exploration across all 12 C-MAPSS tasks:

```bash
uv run run_all_cmapss_benchmarks.py --model-type lstm_dann -o result/benchmark_all

# Plot all per-task benchmark results
uv run plot_benchmark_results.py \
  --csv result/result/benchmark_all/summary_all_metrics.csv \
  --paper-csv result/result/benchmark_all/lstm_dann_paper_results.csv \
  -o result/result/benchmark_all/plots
```

---

## Citation

If you use this code or build upon **Significant-subspace Alignment (SSA)**, please cite the original ICLR 2025 paper:

```bibtex
@inproceedings{adachi2025testtime,
  title={Test-time Adaptation for Regression by Subspace Alignment},
  author={Kazuki Adachi and Shin'ya Yamaguchi and Atsutoshi Kumagai and Tomoki Hamagami},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=SXtl7NRyE5}
}
```
