import numpy as np
from scipy.special import expit

from .preprocessing import clean_text


SIDE_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_len",
    "exclamation_count",
    "question_count",
    "digit_ratio",
    "upper_ratio",
    "clean_word_count",
]


def build_side_features(texts):
    rows = []
    for text in texts:
        raw = text if isinstance(text, str) else ""
        words = raw.split()
        alpha_chars = [ch for ch in raw if ch.isalpha()]
        clean_words = clean_text(raw).split()
        rows.append(
            [
                len(raw),
                len(words),
                float(np.mean([len(w) for w in words])) if words else 0.0,
                raw.count("!"),
                raw.count("?"),
                sum(ch.isdigit() for ch in raw) / max(len(raw), 1),
                sum(ch.isupper() for ch in alpha_chars) / max(len(alpha_chars), 1),
                len(clean_words),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


class TorchLinearFold:
    expects_raw_text = False

    def __init__(self, vectorizer, weight, bias):
        self.vectorizer = vectorizer
        self.weight = np.asarray(weight, dtype=np.float32).ravel()
        self.bias = float(bias)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        logits = X.dot(self.weight) + self.bias
        return expit(np.asarray(logits).ravel())


class SklearnTextFold:
    expects_raw_text = False

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)[:, 1]


class CuMLTextFold:
    expects_raw_text = False

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts):
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        X = cpsp.csr_matrix(self.vectorizer.transform(texts))
        return cp.asnumpy(self.classifier.predict_proba(X))[:, 1]


def tokenize_words(text):
    return clean_text(text).split()


class BowMLPModule:
    pass


def build_neural_module(model_kind, vocab_size, emb_dim, hidden_dim, dropout):
    import torch
    from torch import nn

    class BowMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.net = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            mask = (x != 0).unsqueeze(-1)
            emb = self.embedding(x) * mask
            pooled = emb.sum(1) / mask.sum(1).clamp_min(1)
            return self.net(pooled).squeeze(1)

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            channels = hidden_dim
            self.convs = nn.ModuleList(
                [nn.Conv1d(emb_dim, channels, kernel_size=k, padding=0) for k in (3, 4, 5)]
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(channels * len(self.convs), 1)

        def forward(self, x):
            emb = self.embedding(x).transpose(1, 2)
            pooled = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
            return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)

    if model_kind == "bow_mlp":
        return BowMLP()
    if model_kind == "text_cnn":
        return TextCNN()
    raise ValueError(f"unknown neural model kind: {model_kind}")


class NeuralTextFold:
    expects_raw_text = False

    def __init__(self, vocab, state_dict, model_kind, max_len, emb_dim, hidden_dim, dropout):
        self.vocab = vocab
        self.state_dict = state_dict
        self.model_kind = model_kind
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def _encode(self, texts):
        X = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = [self.vocab.get(tok, 1) for tok in tokenize_words(text)[: self.max_len]]
            if ids:
                X[i, : len(ids)] = ids
        return X

    def predict_proba(self, texts, batch_size=1024):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_neural_module(
            self.model_kind,
            len(self.vocab) + 2,
            self.emb_dim,
            self.hidden_dim,
            self.dropout,
        )
        model.load_state_dict(self.state_dict)
        model.to(device)
        model.eval()
        X = self._encode(texts)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).to(device)
                preds.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(preds)


class CuMLSideFold:
    expects_raw_text = True

    def __init__(self, scaler, classifier):
        self.scaler = scaler
        self.classifier = classifier

    def predict_proba(self, texts):
        import cupy as cp

        X = self.scaler.transform(build_side_features(texts))
        probs = self.classifier.predict_proba(cp.asarray(X, dtype=cp.float32))
        return cp.asnumpy(probs)[:, 1]


def predict_positive_proba(model, X):
    module = model.__class__.__module__
    if module.startswith("cuml"):
        import cupy as cp

        return cp.asnumpy(model.predict_proba(cp.asarray(X, dtype=cp.float32)))[:, 1]
    return model.predict_proba(X)[:, 1]


class OOFEnsembleSentimentModel:
    def __init__(self, model_specs, meta_model, side_scaler, stage2_models=None, final_model=None):
        self.model_specs = model_specs
        self.meta_model = meta_model
        self.side_scaler = side_scaler
        self.stage2_models = stage2_models or []
        self.final_model = final_model

    def _base_matrix(self, texts):
        cleaned = [clean_text(t) for t in texts]
        columns = []
        for spec in self.model_specs:
            fold_preds = [
                fold.predict_proba(texts if getattr(fold, "expects_raw_text", False) else cleaned)
                for fold in spec["folds"]
            ]
            columns.append(np.mean(fold_preds, axis=0))
        return np.column_stack(columns)

    def predict_proba(self, texts):
        texts = list(texts)
        base = self._base_matrix(texts)
        side = self.side_scaler.transform(build_side_features(texts))
        X_level1 = np.hstack([base, side])
        if not self.stage2_models:
            return predict_positive_proba(self.meta_model, X_level1)

        level2 = np.column_stack([predict_positive_proba(model, X_level1) for model in self.stage2_models])
        final_input = np.hstack([X_level1, level2])
        return predict_positive_proba(self.final_model, final_input)

    def predict(self, texts, threshold=0.5):
        return (self.predict_proba(texts) >= threshold).astype(np.int64)
