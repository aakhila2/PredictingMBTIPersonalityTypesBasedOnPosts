"""
MBTI Personality Type Classifier — Multi-Model Comparison
==========================================================
Trains and compares the following models on the MBTI 500 dataset:

  Transformer models  : BERT, DistilBERT, RoBERTa
  Sequence model      : Bi-LSTM (PyTorch)
  Classical ML models : Logistic Regression, Linear SVM, Random Forest, Naive Bayes

Dataset: MBTI 500 dataset from Kaggle
  - https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset

Usage:
  1. Place 'MBTI 500.csv' in the same directory as this script (or edit DATA_PATH).
  2. pip3 install torch torchvision torchaudio
     pip3 install transformers scikit-learn matplotlib seaborn
     pip3 install numpy==1.24.4
     python3 -m nltk.downloader punkt stopwords wordnet
     python3 -m nltk.downloader -d ~/nltk_data all
  3. python bert_mbti_comparison.py

Caching:
  - Preprocessed DataFrame   → CACHE_DIR/preprocessed_data.pkl
  - Upsampled texts/labels   → CACHE_DIR/upsampled_texts.pkl
  - Per-model tokenized tensors → CACHE_DIR/tensors_<ModelName>.pt
  Delete the cache/ folder (or individual files) to force a full re-run.
"""

import os
import re
import pickle
import hashlib
import nltk 
nltk.data.path.append("/home/asangawar/nltk_data")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_PATH               = "MBTI 500.csv"
MODEL_SAVE_DIR          = "saved_models"
CACHE_DIR               = "cache"          # ← all cached artefacts go here
MAX_LEN                 = 128
BATCH_SIZE              = 32
EPOCHS                  = 10
LEARNING_RATE           = 1e-5
EARLY_STOPPING_PATIENCE = 3
RANDOM_STATE            = 2018
MIN_WORD_ROW_FREQUENCY  = 69

# Transformer model registry
TRANSFORMER_CONFIGS = {
    "BERT": (
        BertForSequenceClassification,
        BertTokenizer,
        "bert-base-uncased",
    ),
    "DistilBERT": (
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        "distilbert-base-uncased",
    ),
    "RoBERTa": (
        RobertaForSequenceClassification,
        RobertaTokenizer,
        "roberta-base",
    ),
}

TYPE_COLORS = {
    'Transformer': '#4C72B0',
    'Sequence':    '#DD8452',
    'Classical ML':'#55A868',
}

os.makedirs(CACHE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────
def _pkl_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def _pt_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.pt")


def save_pickle(obj, name: str):
    with open(_pkl_path(name), "wb") as f:
        pickle.dump(obj, f)
    print(f"[CACHE] Saved → {_pkl_path(name)}")


def load_pickle(name: str):
    with open(_pkl_path(name), "rb") as f:
        return pickle.load(f)


def cache_exists_pkl(name: str) -> bool:
    return os.path.exists(_pkl_path(name))


def save_tensors(input_ids: torch.Tensor, attention_masks: torch.Tensor, name: str):
    """Save a pair of tensors (input_ids, attention_masks) to a single .pt file."""
    torch.save({"input_ids": input_ids, "attention_masks": attention_masks}, _pt_path(name))
    print(f"[CACHE] Saved tensors → {_pt_path(name)}")


def load_tensors(name: str):
    """Load and return (input_ids, attention_masks) from a .pt cache file."""
    data = torch.load(_pt_path(name), map_location="cpu")
    print(f"[CACHE] Loaded tensors ← {_pt_path(name)}")
    return data["input_ids"], data["attention_masks"]


def cache_exists_pt(name: str) -> bool:
    return os.path.exists(_pt_path(name))


def _texts_hash(texts: list) -> str:
    """Quick MD5 fingerprint of the text list so we can detect data changes."""
    digest = hashlib.md5("".join(texts[:200]).encode()).hexdigest()
    return digest


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: '{path}'\n"
            "  Download 'MBTI 500.csv' from:\n"
            "  https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset\n"
        )
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 2. EDA — MBTI Type Distribution
# ─────────────────────────────────────────────
def plot_type_distribution(data: pd.DataFrame):
    type_counts = data['type'].value_counts(normalize=True).sort_index() * 100
    palette = sns.color_palette("hsv", len(type_counts))
    plt.figure(figsize=(12, 9))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette=palette)
    plt.ylabel('Percentage (%)')
    plt.xlabel('MBTI Type')
    plt.title('MBTI Type Distribution')
    plt.tight_layout()
    plt.savefig("mbti_distribution.png", dpi=150)
    plt.show()
    print("[INFO] Distribution plot saved → mbti_distribution.png")


# ─────────────────────────────────────────────
# 3. Text Preprocessing  (with pickle cache)
# ─────────────────────────────────────────────
def download_nltk_resources():
    nltk.download('wordnet',   quiet=True)
    nltk.download('stopwords', quiet=True)
    wordnet_zip = '/usr/share/nltk_data/corpora/wordnet.zip'
    wordnet_dir = '/usr/share/nltk_data/corpora/wordnet'
    if os.path.exists(wordnet_zip) and not os.path.exists(wordnet_dir):
        os.system(f"unzip -q {wordnet_zip} -d /usr/share/nltk_data/corpora/")


def tokenize(text: str):
    text = re.sub(r'http\S+|[^a-zA-Z0-9\s]', '', text)
    return text.lower().split()


def build_filtered_word_set(data: pd.DataFrame, min_freq: int) -> set:
    word_count = defaultdict(int)
    for post in data['posts']:
        for word in set(tokenize(post)):
            word_count[word] += 1
    filtered = {w for w, c in word_count.items() if c >= min_freq}
    print(f"[INFO] Vocabulary after frequency filter: {len(filtered)} unique words")
    return filtered


def filter_text_preserve_delimiter(text: str, filtered_words: set) -> str:
    posts = text.split('|||')
    filtered = [
        ' '.join(w for w in tokenize(p) if w in filtered_words)
        for p in posts
    ]
    return '|||'.join(filtered)


def clean_and_preprocess(text: str, lemmatizer, stop_words: set) -> str:
    text = re.sub(r'http\S+',  '', text)
    text = re.sub(r'[^\w\s]',  '', text)
    words = text.split()
    cleaned = [
        lemmatizer.lemmatize(w.lower())
        for w in words
        if w.lower() not in stop_words
    ]
    return ' '.join(cleaned)


def preprocess_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the preprocessed DataFrame.
    Result is cached to CACHE_DIR/preprocessed_data.pkl so the heavy
    NLP work is skipped on every re-run.
    """
    cache_name = "preprocessed_data"
    if cache_exists_pkl(cache_name):
        print("[CACHE] Loading preprocessed data from cache...")
        return load_pickle(cache_name)

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    print("[INFO] Building vocabulary filter...")
    filtered_words = build_filtered_word_set(data, MIN_WORD_ROW_FREQUENCY)

    print("[INFO] Applying vocabulary filter...")
    data['filtered_posts'] = data['posts'].apply(
        lambda x: filter_text_preserve_delimiter(x, filtered_words)
    )
    data['filtered_posts'] = data['filtered_posts'].str.replace('|||', '', regex=False)

    print("[INFO] Lemmatizing and removing stop words...")
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))
    data['filtered_posts'] = data['filtered_posts'].apply(
        lambda x: clean_and_preprocess(x, lemmatizer, stop_words)
    )

    before = len(data)
    data = data[data['filtered_posts'].str.strip() != ''].reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(data)} empty rows. Remaining: {len(data)}")

    save_pickle(data, cache_name)
    return data


# ─────────────────────────────────────────────
# 4. Label Encoding & Upsampling  (with pickle cache)
# ─────────────────────────────────────────────
def encode_and_upsample(data: pd.DataFrame):
    """
    Returns (data_upsampled, label_encoder).
    Cached to CACHE_DIR/upsampled_texts.pkl.
    """
    cache_name = "upsampled_texts"
    if cache_exists_pkl(cache_name):
        print("[CACHE] Loading upsampled texts/labels from cache...")
        return load_pickle(cache_name)

    label_encoder = LabelEncoder()
    data['encoded_labels'] = label_encoder.fit_transform(data['type'])

    class_counts = data['encoded_labels'].value_counts()
    max_count    = class_counts.max()
    print(f"[INFO] Upsampling {len(class_counts)} classes to {max_count} each...")

    parts = []
    for class_idx in class_counts.index:
        subset    = data[data['encoded_labels'] == class_idx]
        upsampled = resample(subset, replace=True, n_samples=max_count, random_state=123)
        parts.append(upsampled)

    data_upsampled = pd.concat(parts).sample(frac=1).reset_index(drop=True)
    print(f"[INFO] Upsampled dataset size: {len(data_upsampled)}")

    save_pickle((data_upsampled, label_encoder), cache_name)
    return data_upsampled, label_encoder


# ─────────────────────────────────────────────
# 5. Transformer Tokenization  (with .pt tensor cache)
# ─────────────────────────────────────────────
def _encode_text_worker(args):
    """Top-level so ProcessPoolExecutor can pickle it."""
    sent, max_len, pretrained_name = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pretrained_name)
    enc = tok.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
    )
    return enc['input_ids'], enc['attention_mask']


def transformer_encode(texts, max_len: int, pretrained_name: str, model_name: str):
    """
    Tokenise *texts* for *pretrained_name*.

    On the first call the tensors are saved to
      CACHE_DIR/tensors_<model_name>.pt
    Subsequent calls load them directly, skipping tokenisation entirely.

    A lightweight hash of the first 200 texts is embedded in the cache
    filename so that if the underlying data changes the old cache is
    automatically bypassed (new file is written alongside).
    """
    data_hash  = _texts_hash(texts)
    cache_name = f"tensors_{model_name}_{data_hash}_maxlen{max_len}"

    if cache_exists_pt(cache_name):
        print(f"[CACHE] Tensor cache hit for {model_name} — skipping tokenisation.")
        return load_tensors(cache_name)

    args = [(sent, max_len, pretrained_name) for sent in texts]
    input_ids, attention_masks = [], []
    from transformers import AutoTokenizer

    print(f"[INFO] Tokenizing {len(texts)} texts with '{pretrained_name}' (FAST)...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]

    input_ids_t = input_ids
    attention_masks_t = attention_masks
    save_tensors(input_ids_t, attention_masks_t, cache_name)
    return input_ids_t, attention_masks_t


# ─────────────────────────────────────────────
# 6. DataLoaders
# ─────────────────────────────────────────────
def build_dataloaders(input_ids, attention_masks, labels):
    tr_inp, tmp_inp, tr_lbl, tmp_lbl = train_test_split(
        input_ids, labels, random_state=RANDOM_STATE, test_size=0.2)
    va_inp, te_inp, va_lbl, te_lbl = train_test_split(
        tmp_inp, tmp_lbl, random_state=RANDOM_STATE, test_size=0.5)
    tr_msk, tmp_msk, _, _ = train_test_split(
        attention_masks, labels, random_state=RANDOM_STATE, test_size=0.2)
    va_msk, te_msk, _, _ = train_test_split(
        tmp_msk, tmp_lbl, random_state=RANDOM_STATE, test_size=0.5)

    def make_dl(inp, msk, lbl, shuffle):
        ds = TensorDataset(inp, msk, lbl)
        s  = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        return DataLoader(ds, sampler=s, batch_size=BATCH_SIZE)

    train_dl = make_dl(tr_inp, tr_msk, tr_lbl, shuffle=True)
    val_dl   = make_dl(va_inp, va_msk, va_lbl, shuffle=False)
    test_dl  = make_dl(te_inp, te_msk, te_lbl, shuffle=False)
    print(f"[INFO] Batches — Train: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")
    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────
# 7. Shared helpers
# ─────────────────────────────────────────────
def extract_dimension_labels(encoded_labels, label_encoder):
    decoded = label_encoder.inverse_transform(list(encoded_labels))
    return {
        'EI': [1 if lbl[0] == 'E' else 0 for lbl in decoded],
        'NS': [1 if lbl[1] == 'N' else 0 for lbl in decoded],
        'FT': [1 if lbl[2] == 'F' else 0 for lbl in decoded],
        'JP': [1 if lbl[3] == 'J' else 0 for lbl in decoded],
    }


# ─────────────────────────────────────────────
# 8. Transformer Training & Evaluation
# ─────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="  Train", leave=False):
        b_ids, b_msk, b_lbl = (t.to(device) for t in batch)
        model.zero_grad()
        out  = model(b_ids, attention_mask=b_msk, labels=b_lbl)
        out.loss.backward()
        total_loss += out.loss.item()
        optimizer.step()
    return total_loss / len(dataloader)


def evaluate_transformer(model, dataloader, label_encoder, device):
    model.eval()
    predictions, true_labels = [], []
    for batch in tqdm(dataloader, desc="  Eval ", leave=False):
        b_ids, b_msk, b_lbl = (t.to(device) for t in batch)
        with torch.no_grad():
            out = model(b_ids, attention_mask=b_msk)
        predictions.extend(np.argmax(out.logits.cpu().numpy(), axis=1).tolist())
        true_labels.extend(b_lbl.cpu().numpy().tolist())

    acc  = accuracy_score(true_labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted')
    true_dims = extract_dimension_labels(true_labels, label_encoder)
    pred_dims = extract_dimension_labels(predictions, label_encoder)
    dim_acc   = {d: accuracy_score(true_dims[d], pred_dims[d]) for d in true_dims}
    return acc, prec, rec, f1, dim_acc


def train_transformer_model(model_name, model_cls, tok_cls, pretrained,
                             texts, labels_tensor, label_encoder, device):
    print(f"\n{'='*55}\n  Training: {model_name}\n{'='*55}")

    # ← model_name passed so each model gets its own tensor cache file
    input_ids, attention_masks = transformer_encode(
        texts, MAX_LEN, pretrained, model_name)
    train_dl, val_dl, test_dl = build_dataloaders(input_ids, attention_masks, labels_tensor)

    num_labels = len(label_encoder.classes_)
    model = model_cls.from_pretrained(
        pretrained, num_labels=num_labels,
        output_attentions=False, output_hidden_states=False)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.01)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_path  = os.path.join(MODEL_SAVE_DIR, f'best_{model_name}.pth')
    best_acc   = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n  Epoch {epoch+1}/{EPOCHS}")
        loss = train_epoch(model, train_dl, optimizer, device)
        print(f"  Train Loss: {loss:.4f}")

        acc, prec, rec, f1, dims = evaluate_transformer(model, val_dl, label_encoder, device)
        print(f"  Val  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        for d, da in dims.items():
            print(f"    {d}: {da:.4f}")

        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print("  ✓ Saved best model.")
        else:
            no_improve += 1
            print(f"  No improve ({no_improve}/{EARLY_STOPPING_PATIENCE})")
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print("  Early stopping.")
                break

    model.load_state_dict(torch.load(best_path))
    test_acc, test_prec, test_rec, test_f1, test_dims = evaluate_transformer(
        model, test_dl, label_encoder, device)
    print(f"\n  [{model_name}] TEST → Acc={test_acc:.4f} "
          f"P={test_prec:.4f} R={test_rec:.4f} F1={test_f1:.4f}")
    for d, da in test_dims.items():
        print(f"    {d}: {da:.4f}")

    return {
        'model': model_name, 'type': 'Transformer',
        'accuracy': test_acc, 'precision': test_prec,
        'recall': test_rec,   'f1': test_f1,
        'dim_accuracies': test_dims,
    }, model


# ─────────────────────────────────────────────
# 9. Bi-LSTM Model
# ─────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hn, _) = self.lstm(emb)
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(self.dropout(hidden))


def build_vocab(texts):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def text_to_ids(text, vocab, max_len):
    ids = [vocab.get(w, vocab['<UNK>']) for w in text.split()][:max_len]
    ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids


def train_bilstm(texts, labels_np, label_encoder, device):
    """
    Bi-LSTM tensors (token ID sequences) are also cached to
    CACHE_DIR/tensors_BiLSTM_<hash>_maxlen<N>.pt so vocabulary
    building and padding are skipped on re-runs.
    """
    print(f"\n{'='*55}\n  Training: Bi-LSTM\n{'='*55}")

    tr_txt, tmp_txt, tr_lbl, tmp_lbl = train_test_split(
        texts, labels_np, test_size=0.2, random_state=RANDOM_STATE)
    va_txt, te_txt, va_lbl, te_lbl = train_test_split(
        tmp_txt, tmp_lbl, test_size=0.5, random_state=RANDOM_STATE)

    # ── Vocab cache ───────────────────────────────────────────────
    vocab_cache = "bilstm_vocab"
    if cache_exists_pkl(vocab_cache):
        print("[CACHE] Loading Bi-LSTM vocab from cache...")
        vocab = load_pickle(vocab_cache)
    else:
        vocab = build_vocab(tr_txt)
        save_pickle(vocab, vocab_cache)

    # ── Tensor cache (per split) ──────────────────────────────────
    data_hash = _texts_hash(texts)

    def _make_or_load_split(split_name, txts, lbls):
        cache_name = f"tensors_BiLSTM_{split_name}_{data_hash}_maxlen{MAX_LEN}"
        if cache_exists_pt(cache_name):
            print(f"[CACHE] Tensor cache hit for Bi-LSTM/{split_name}.")
            ids_t, lbl_t = load_tensors(cache_name)
            # load_tensors returns (input_ids, attention_masks); reuse slots
            return ids_t, lbl_t
        ids_t = torch.tensor([text_to_ids(t, vocab, MAX_LEN) for t in txts])
        lbl_t = torch.tensor(lbls)
        # We store labels in the "attention_masks" slot of the generic helper
        save_tensors(ids_t, lbl_t, cache_name)
        return ids_t, lbl_t

    tr_ids, tr_lbl_t = _make_or_load_split("train", tr_txt, tr_lbl)
    va_ids, va_lbl_t = _make_or_load_split("val",   va_txt, va_lbl)
    te_ids, te_lbl_t = _make_or_load_split("test",  te_txt, te_lbl)

    def make_dl(ids, lbls, shuffle):
        ds = TensorDataset(ids, lbls)
        s  = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        return DataLoader(ds, sampler=s, batch_size=BATCH_SIZE)

    train_dl = make_dl(tr_ids, tr_lbl_t, True)
    val_dl   = make_dl(va_ids, va_lbl_t, False)
    test_dl  = make_dl(te_ids, te_lbl_t, False)

    num_classes = len(label_encoder.classes_)
    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=128, hidden_dim=256,
        num_layers=2, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    no_improve = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for ids_b, lbl_b in tqdm(train_dl, desc=f"  LSTM Ep {epoch+1}", leave=False):
            ids_b, lbl_b = ids_b.to(device), lbl_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids_b), lbl_b)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids_b, lbl_b in val_dl:
                preds.extend(torch.argmax(model(ids_b.to(device)), dim=1).cpu().numpy())
                trues.extend(lbl_b.numpy())
        acc = accuracy_score(trues, preds)
        print(f"  Epoch {epoch+1}  Loss={total_loss/len(train_dl):.4f}  Val Acc={acc:.4f}")

        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print("  ✓ Saved best LSTM.")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ids_b, lbl_b in test_dl:
            preds.extend(torch.argmax(model(ids_b.to(device)), dim=1).cpu().numpy())
            trues.extend(lbl_b.numpy())

    acc  = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average='weighted')
    true_dims = extract_dimension_labels(trues, label_encoder)
    pred_dims = extract_dimension_labels(preds,  label_encoder)
    dim_acc   = {d: accuracy_score(true_dims[d], pred_dims[d]) for d in true_dims}

    print(f"\n  [Bi-LSTM] TEST → Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    for d, da in dim_acc.items():
        print(f"    {d}: {da:.4f}")

    return {
        'model': 'Bi-LSTM', 'type': 'Sequence',
        'accuracy': acc, 'precision': prec,
        'recall': rec,   'f1': f1,
        'dim_accuracies': dim_acc,
    }


# ─────────────────────────────────────────────
# 10. Classical ML Models
# ─────────────────────────────────────────────
CLASSICAL_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial'),
    "Linear SVM": LinearSVC(max_iter=2000, C=1.0),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
    "Naive Bayes": MultinomialNB(alpha=0.1),
}


def train_classical_models(texts, labels_np, label_encoder):
    results = []
    tr_txt, tmp_txt, tr_lbl, tmp_lbl = train_test_split(
        texts, labels_np, test_size=0.2, random_state=RANDOM_STATE)
    _, te_txt, _, te_lbl = train_test_split(
        tmp_txt, tmp_lbl, test_size=0.5, random_state=RANDOM_STATE)

    vectorizer = TfidfVectorizer(max_features=30000, sublinear_tf=True, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(tr_txt)
    X_test  = vectorizer.transform(te_txt)

    for name, clf in CLASSICAL_MODELS.items():
        print(f"\n{'='*55}\n  Training: {name}\n{'='*55}")
        clf.fit(X_train, tr_lbl)
        preds = clf.predict(X_test)

        acc  = accuracy_score(te_lbl, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            te_lbl, preds, average='weighted')
        true_dims = extract_dimension_labels(list(te_lbl), label_encoder)
        pred_dims = extract_dimension_labels(list(preds),  label_encoder)
        dim_acc   = {d: accuracy_score(true_dims[d], pred_dims[d]) for d in true_dims}

        print(f"  TEST → Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
        for d, da in dim_acc.items():
            print(f"    {d}: {da:.4f}")

        results.append({
            'model': name, 'type': 'Classical ML',
            'accuracy': acc, 'precision': prec,
            'recall': rec,   'f1': f1,
            'dim_accuracies': dim_acc,
        })
    return results


# ─────────────────────────────────────────────
# 11. Comparison Plots  (4 charts)
# ─────────────────────────────────────────────
def plot_comparison(all_results: list):
    df = pd.DataFrame([{
        'Model':     r['model'],
        'Type':      r['type'],
        'Accuracy':  r['accuracy'],
        'Precision': r['precision'],
        'Recall':    r['recall'],
        'F1 Score':  r['f1'],
    } for r in all_results]).sort_values('Accuracy', ascending=False).reset_index(drop=True)

    colors = [TYPE_COLORS[t] for t in df['Type']]

    # ── Chart 1: Accuracy bar ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(df['Model'], df['Accuracy'], color=colors,
                  edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.3f}", ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_ylim(0, min(df['Accuracy'].max() + 0.12, 1.0))
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('MBTI Classification — Test Accuracy by Model',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    legend_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in TYPE_COLORS.items()]
    ax.legend(handles=legend_patches, title='Model Type', loc='upper right')
    plt.tight_layout()
    plt.savefig("comparison_accuracy.png", dpi=150)
    plt.show()
    print("[INFO] Saved → comparison_accuracy.png")

    # ── Chart 2: Multi-metric grouped bars ───────────────────────
    metrics       = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    x             = np.arange(len(df))
    bar_width     = 0.2
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, (metric, mc) in enumerate(zip(metrics, metric_colors)):
        ax.bar(x + i * bar_width, df[metric], bar_width,
               label=metric, color=mc, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(df['Model'], rotation=20, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('MBTI Classification — Multi-Metric Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("comparison_metrics.png", dpi=150)
    plt.show()
    print("[INFO] Saved → comparison_metrics.png")

    # ── Chart 3: Dimension accuracy heatmap ──────────────────────
    dims = ['EI', 'NS', 'FT', 'JP']
    heat_data = pd.DataFrame(
        [[r['dim_accuracies'][d] for d in dims] for r in all_results],
        index=[r['model'] for r in all_results],
        columns=dims,
    )
    heat_data = heat_data.loc[
        heat_data.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, len(heat_data) * 0.75 + 1.5))
    sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='YlGnBu',
                vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Accuracy'})
    ax.set_title('MBTI Dimension Accuracy per Model',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('MBTI Dimension')
    ax.set_ylabel('Model')
    plt.tight_layout()
    plt.savefig("comparison_dimensions.png", dpi=150)
    plt.show()
    print("[INFO] Saved → comparison_dimensions.png")

    # ── Chart 4: Radar / Spider chart ────────────────────────────
    labels_radar = ['Accuracy', 'Precision', 'Recall', 'F1']
    N      = len(labels_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", len(df))
    for i, row in df.iterrows():
        vals = [row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score']]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row['Model'], color=palette[i])
        ax.fill(angles, vals, alpha=0.07, color=palette[i])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Radar Chart — All Models',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig("comparison_radar.png", dpi=150)
    plt.show()
    print("[INFO] Saved → comparison_radar.png")

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("="*70)
    print(df[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*70)


# ─────────────────────────────────────────────
# 12. Single-text Inference
# ─────────────────────────────────────────────
def predict_mbti(text: str, model, tokenizer, label_encoder, device) -> str:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    lem = WordNetLemmatizer()
    sw  = set(stopwords.words('english'))
    cleaned = clean_and_preprocess(text, lem, sw)
    enc = tokenizer.encode_plus(
        cleaned, add_special_tokens=True, max_length=MAX_LEN,
        padding='max_length', return_attention_mask=True,
        truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        out = model(enc['input_ids'].to(device),
                    attention_mask=enc['attention_mask'].to(device))
    idx = torch.argmax(out.logits, dim=1).cpu().numpy()[0]
    return label_encoder.inverse_transform([idx])[0]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"[INFO] Cache directory: {os.path.abspath(CACHE_DIR)}")
    print("[INFO] Delete the cache/ folder to force a full re-run.\n")

    # 1. Load
    data = load_data(DATA_PATH)

    # 2. EDA
    plot_type_distribution(data)

    # 3. NLTK
    #download_nltk_resources()

    # 4. Preprocess  (cached after first run)
    #data = preprocess_dataframe(data)

    # 5. Encode & upsample  (cached after first run)
    data_up, label_encoder = encode_and_upsample(data)
    texts = data_up['posts'].tolist()
    labels_np = data_up['encoded_labels'].values
    #texts = texts[:50000]
    #labels_np = labels_np[:50000]
    labels_t  = torch.tensor(labels_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    all_results         = []
    best_trans_result   = None
    best_trans_model    = None
    best_trans_tok_cls  = None
    best_trans_pretrain = None

    # ── A. Transformer models  (tensor cache per model) ───────────
    for model_name, (model_cls, tok_cls, pretrained) in TRANSFORMER_CONFIGS.items():
        result, trained_model = train_transformer_model(
            model_name, model_cls, tok_cls, pretrained,
            texts, labels_t, label_encoder, device)
        all_results.append(result)

        if best_trans_result is None or result['accuracy'] > best_trans_result['accuracy']:
            best_trans_result   = result
            best_trans_model    = trained_model
            best_trans_tok_cls  = tok_cls
            best_trans_pretrain = pretrained

    # ── B. Bi-LSTM  (tensor cache per split) ─────────────────────
    all_results.append(train_bilstm(texts, labels_np, label_encoder, device))

    # ── C. Classical ML ───────────────────────────────────────────
    all_results.extend(train_classical_models(texts, labels_np, label_encoder))

    # ── D. Plots ──────────────────────────────────────────────────
    plot_comparison(all_results)

    # ── E. Inference demo (best transformer) ──────────────────────
    best_name = best_trans_result['model']
    tok = best_trans_tok_cls.from_pretrained(best_trans_pretrain)
    sample = (
        "I love exploring new ideas and connecting with people from all walks of life. "
        "I often find myself thinking about the future and what possibilities lie ahead. "
        "Socializing energizes me, though I also value deep one-on-one conversations."
    )
    pred = predict_mbti(sample, best_trans_model, tok, label_encoder, device)
    print(f"\n[DEMO — {best_name}]\nText : {sample}\nPredicted MBTI: {pred}")


if __name__ == '__main__':
    main()
