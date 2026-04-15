# MBTI Personality Type Classifier — Multi-Model Comparison

Trains and evaluates eight models on the **MBTI 500 dataset** to predict Myers–Briggs personality types from user posts.

---

## Models Compared

| Category | Models |
|---|---|
| Transformer | BERT, DistilBERT, RoBERTa |
| Sequence | Bi-LSTM (PyTorch) |
| Classical ML | Logistic Regression, Linear SVM, Random Forest, Naïve Bayes |

---

## Dataset

**MBTI 500** — available on Kaggle:
[https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)

- 106,067 rows of forum posts labelled with one of 16 MBTI personality types
- Two columns: `posts` (pipe-separated entries) and `type`

---

## Project Structure

```
.
├── bert_mbti.py   # Main script
├── MBTI 500.csv              # Dataset (download separately)
├── cache/                    # Auto-created; stores preprocessed data and tensors
├── saved_models/             # Auto-created; stores best model checkpoints (.pth)
├── mbti_distribution.png     # EDA plot (generated on run)
├── comparison_accuracy.png   # Results plot (generated on run)
├── comparison_metrics.png    # Results plot (generated on run)
├── comparison_dimensions.png # Results plot (generated on run)
└── comparison_radar.png      # Results plot (generated on run)
```

---

## Installation

**1. Install PyTorch** (with CUDA if available — recommended):
```bash
pip3 install torch torchvision torchaudio
```

**2. Install remaining dependencies:**
```bash
pip3 install transformers scikit-learn matplotlib seaborn
pip3 install numpy==1.24.4
```

**3. Download NLTK resources:**
```bash
python3 -m nltk.downloader punkt stopwords wordnet
python3 -m nltk.downloader -d ~/nltk_data all
```

---

## Usage

1. Download `MBTI 500.csv` from Kaggle and place it in the same directory as `bert_mbti.py` (or edit the `DATA_PATH` variable at the top of the script).

2. Run the script:
```bash
python3 bert_mbti.py
```

The script will automatically:
- Load and preprocess the dataset
- Upsample minority classes to balance the training data
- Train all eight models with early stopping
- Save the best checkpoint for each model
- Generate and save four comparison plots
- Print a final summary table

---

## Caching

To avoid re-running expensive preprocessing and tokenisation steps, the script caches intermediate results:

| Cache File | Contents |
|---|---|
| `cache/preprocessed_data.pkl` | Lemmatised, filtered DataFrame |
| `cache/upsampled_texts.pkl` | Upsampled texts and encoded labels |
| `cache/tensors_<ModelName>_<hash>_maxlen<N>.pt` | Tokenised tensors per transformer |
| `cache/tensors_BiLSTM_<split>_<hash>_maxlen<N>.pt` | Bi-LSTM token ID tensors per split |
| `cache/bilstm_vocab.pkl` | Bi-LSTM vocabulary dictionary |

> **To force a full re-run**, delete the `cache/` folder:
> ```bash
> rm -rf cache/
> ```

---

## Key Configuration

These values can be edited at the top of `bert_mbti.py`:

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | `"MBTI 500.csv"` | Path to the dataset CSV |
| `MAX_LEN` | `128` | Maximum token length for transformers |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `10` | Maximum training epochs |
| `LEARNING_RATE` | `1e-5` | Learning rate for transformer fine-tuning |
| `EARLY_STOPPING_PATIENCE` | `3` | Epochs without improvement before stopping |
| `RANDOM_STATE` | `2018` | Seed for reproducibility |
| `MIN_WORD_ROW_FREQUENCY` | `69` | Minimum document frequency for vocabulary filter |

---

## Output

After a full run, the following files are generated:

- **`mbti_distribution.png`** — Bar chart of MBTI type distribution in the raw dataset
- **`comparison_accuracy.png`** — Test accuracy comparison across all models
- **`comparison_metrics.png`** — Grouped bar chart of Accuracy, Precision, Recall, and F1
- **`comparison_dimensions.png`** — Heatmap of per-dimension accuracy (EI, NS, FT, JP)
- **`comparison_radar.png`** — Radar chart summarising all four metrics per model

---

## Results Summary

| Model | Type | Accuracy | F1 |
|---|---|---|---|
| Linear SVM | Classical ML | 0.9813 | 0.9812 |
| Random Forest | Classical ML | 0.9794 | 0.9794 |
| BERT | Transformer | 0.9544 | 0.9541 |
| DistilBERT | Transformer | 0.9505 | 0.9501 |
| RoBERTa | Transformer | 0.9490 | 0.9480 |
| Naïve Bayes | Classical ML | 0.8917 | 0.8932 |
| Bi-LSTM | Sequence | 0.8663 | 0.8605 |
| Logistic Regression | Classical ML | 0.8342 | 0.8340 |

---

## Group

**CS354N — Computational Intelligence Laboratory**
Project: *Predicting MBTI Personality Types Based on Posts* | Group 13

| Name | Roll Number |
|---|---|
| Akella Akhila | 230001005 |
| Kommireddy Jayanthi | 230001041 |
| Nelluri Pavithra | 230001057 |
| Parimi Sunitha | 230001061 |
| Soha Sultana | 230001071 |
