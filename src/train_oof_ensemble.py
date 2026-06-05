import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .ensemble import (
    OOFEnsembleSentimentModel,
    SIDE_FEATURE_NAMES,
    CuMLSideFold,
    CuMLTextFold,
    SklearnTextFold,
    TorchLinearFold,
    build_side_features,
    predict_positive_proba,
)
from .preprocessing import clean_series


MODEL_SPECS = [
    {
        "name": "word_1_2_50k",
        "backend": "torch",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 50_000,
        "min_df": 2,
        "max_df": 0.95,
        "epochs": 5,
        "lr": 2e-3,
        "weight_decay": 2e-4,
    },
    {
        "name": "word_1_3_70k",
        "backend": "torch",
        "analyzer": "word",
        "ngram_range": (1, 3),
        "max_features": 70_000,
        "min_df": 2,
        "max_df": 0.95,
        "epochs": 5,
        "lr": 2e-3,
        "weight_decay": 3e-4,
    },
    {
        "name": "char_wb_3_5_80k",
        "backend": "torch",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 80_000,
        "min_df": 2,
        "max_df": 0.95,
        "epochs": 4,
        "lr": 1e-3,
        "weight_decay": 2e-4,
    },
    {
        "name": "xgb_word_1_2_30k",
        "backend": "xgboost_gpu",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 30_000,
        "min_df": 2,
        "max_df": 0.95,
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.035,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
    },
    {
        "name": "cuml_word_1_2_60k_c2",
        "backend": "cuml_tfidf_gpu",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 60_000,
        "min_df": 2,
        "max_df": 0.95,
        "C": 2.0,
        "max_iter": 300,
    },
    {
        "name": "cuml_char_wb_3_5_90k_c4",
        "backend": "cuml_tfidf_gpu",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 90_000,
        "min_df": 2,
        "max_df": 0.95,
        "C": 4.0,
        "max_iter": 300,
    },
    {
        "name": "xgb_char_wb_3_5_40k",
        "backend": "xgboost_gpu",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 40_000,
        "min_df": 2,
        "max_df": 0.95,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
    },
    {
        "name": "catboost_word_1_2_25k",
        "backend": "catboost_gpu",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 25_000,
        "min_df": 2,
        "max_df": 0.95,
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.045,
    },
    {
        "name": "catboost_char_wb_3_5_35k",
        "backend": "catboost_gpu",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 35_000,
        "min_df": 2,
        "max_df": 0.95,
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
    },
    {
        "name": "cuml_side_card_lr",
        "backend": "cuml_side_gpu",
        "max_iter": 200,
    },
]


class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.tocsr()
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().ravel().astype(np.float32)
        return torch.from_numpy(row), torch.tensor(self.y[idx], dtype=torch.float32)


def load_split(path):
    df = pd.read_csv(path)
    df["raw_text"] = df["text"].fillna("").astype(str)
    df["text"] = clean_series(df["text"])
    return df


def make_vectorizer(spec):
    return TfidfVectorizer(
        analyzer=spec["analyzer"],
        ngram_range=spec["ngram_range"],
        max_features=spec["max_features"],
        min_df=spec["min_df"],
        max_df=spec["max_df"],
        sublinear_tf=True,
        strip_accents="unicode",
    )


def train_torch_linear(X_train, y_train, X_val, spec, device, batch_size):
    model = nn.Linear(X_train.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        SparseDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model.train()
    for epoch in range(1, spec["epochs"] + 1):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"    epoch {epoch}/{spec['epochs']} loss={np.mean(losses):.4f}")

    val_pred = predict_torch_linear(model, X_val, device, batch_size)
    weight = model.weight.detach().cpu().numpy().ravel()
    bias = float(model.bias.detach().cpu().numpy()[0])
    return val_pred, weight, bias


def predict_torch_linear(model, X, device, batch_size):
    model.eval()
    X = X.tocsr()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = X[start : start + batch_size].toarray().astype(np.float32)
            xb = torch.from_numpy(batch).to(device)
            logits = model(xb).squeeze(1)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def metric_row(y_true, probs):
    preds = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary")
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def train_xgboost_gpu(X_train, y_train, X_val, X_test, spec):
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        n_estimators=spec["n_estimators"],
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
        subsample=spec["subsample"],
        colsample_bytree=spec["colsample_bytree"],
        random_state=config.RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(X_train, y_train, verbose=False)
    val_pred = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]
    return val_pred, test_pred, clf


def train_cuml_tfidf_gpu(X_train, y_train, X_val, X_test, spec):
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

    clf = CuMLLogisticRegression(max_iter=spec["max_iter"], C=spec["C"])
    clf.fit(cpsp.csr_matrix(X_train), cp.asarray(y_train, dtype=cp.int32))
    val_pred = cp.asnumpy(clf.predict_proba(cpsp.csr_matrix(X_val)))[:, 1]
    test_pred = cp.asnumpy(clf.predict_proba(cpsp.csr_matrix(X_test)))[:, 1]
    return val_pred, test_pred, clf


def train_catboost_gpu(X_train, y_train, X_val, X_test, spec):
    from catboost import CatBoostClassifier

    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        iterations=spec["iterations"],
        depth=spec["depth"],
        learning_rate=spec["learning_rate"],
        random_seed=config.RANDOM_SEED,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    val_pred = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]
    return val_pred, test_pred, clf


def train_cuml_side_gpu(raw_train, y_train, raw_val, raw_test, spec):
    import cupy as cp
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

    scaler = StandardScaler()
    X_train = scaler.fit_transform(build_side_features(raw_train))
    X_val = scaler.transform(build_side_features(raw_val))
    X_test = scaler.transform(build_side_features(raw_test))
    clf = CuMLLogisticRegression(max_iter=spec["max_iter"])
    clf.fit(cp.asarray(X_train, dtype=cp.float32), cp.asarray(y_train, dtype=cp.int32))
    val_pred = cp.asnumpy(clf.predict_proba(cp.asarray(X_val, dtype=cp.float32)))[:, 1]
    test_pred = cp.asnumpy(clf.predict_proba(cp.asarray(X_test, dtype=cp.float32)))[:, 1]
    return val_pred, test_pred, CuMLSideFold(scaler, clf)


def train_meta_gpu(X_meta, y, X_meta_test):
    import xgboost as xgb

    meta = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.95,
        colsample_bytree=0.95,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
    )
    meta.fit(X_meta, y, verbose=False)
    return meta, meta.predict_proba(X_meta)[:, 1], meta.predict_proba(X_meta_test)[:, 1]


def make_stage2_model(name):
    if name == "xgb_l2":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_estimators=220,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.95,
            colsample_bytree=0.95,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
        )
    if name == "catboost_l2":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            iterations=220,
            depth=3,
            learning_rate=0.04,
            random_seed=config.RANDOM_SEED,
            verbose=False,
        )
    if name == "cuml_lr_l2":
        from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

        return CuMLLogisticRegression(max_iter=300)
    raise ValueError(f"unknown stage2 model: {name}")


def fit_stage2_model(model, X, y):
    module = model.__class__.__module__
    if module.startswith("cuml"):
        import cupy as cp

        model.fit(cp.asarray(X, dtype=cp.float32), cp.asarray(y, dtype=cp.int32))
    else:
        model.fit(X, y)
    return model


def train_multistage_stack(X_level1, y, X_level1_test, folds):
    stage2_names = ["xgb_l2", "catboost_l2", "cuml_lr_l2"]
    stage2_oof = np.zeros((X_level1.shape[0], len(stage2_names)), dtype=np.float32)
    stage2_test = np.zeros((X_level1_test.shape[0], len(stage2_names)), dtype=np.float32)
    final_stage2_models = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED + 17)

    for model_idx, name in enumerate(stage2_names):
        print(f"\nstage2 {model_idx + 1}/{len(stage2_names)}: {name}")
        test_fold_preds = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_level1, y), start=1):
            print(f"  fold {fold}/{folds}")
            model = fit_stage2_model(make_stage2_model(name), X_level1[tr_idx], y[tr_idx])
            stage2_oof[val_idx, model_idx] = predict_positive_proba(model, X_level1[val_idx])
            test_fold_preds.append(predict_positive_proba(model, X_level1_test))
        stage2_test[:, model_idx] = np.mean(test_fold_preds, axis=0)
        final_model = fit_stage2_model(make_stage2_model(name), X_level1, y)
        final_stage2_models.append(final_model)

    X_final = np.hstack([X_level1, stage2_oof])
    X_final_test = np.hstack([X_level1_test, stage2_test])
    final_oof = np.zeros(X_final.shape[0], dtype=np.float32)
    final_test_folds = []
    print("\nstage3 final blender: xgb_l2")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_final, y), start=1):
        print(f"  fold {fold}/{folds}")
        fold_model = make_stage2_model("xgb_l2")
        fold_model.set_params(n_estimators=300, max_depth=2, learning_rate=0.025)
        fold_model.fit(X_final[tr_idx], y[tr_idx], verbose=False)
        final_oof[val_idx] = predict_positive_proba(fold_model, X_final[val_idx])
        final_test_folds.append(predict_positive_proba(fold_model, X_final_test))

    final_model = make_stage2_model("xgb_l2")
    final_model.set_params(n_estimators=300, max_depth=2, learning_rate=0.025)
    final_model.fit(X_final, y, verbose=False)
    oof_probs = final_oof
    test_probs = np.mean(final_test_folds, axis=0)
    return final_stage2_models, final_model, stage2_oof, stage2_test, oof_probs, test_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA GPU not available. Re-run with --allow-cpu only if CPU fallback is intended.")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train_df = load_split(config.TRAIN_CSV)
    dev_df = load_split(config.DEV_CSV)
    test_df = load_split(config.TEST_CSV)
    full_df = pd.concat([train_df, dev_df], ignore_index=True)

    texts = full_df["text"].to_numpy()
    raw_texts = full_df["raw_text"].to_numpy()
    y = full_df["label"].to_numpy()
    test_texts = test_df["text"].to_numpy()
    test_raw_texts = test_df["raw_text"].to_numpy()
    y_test = test_df["label"].to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.RANDOM_SEED)
    oof = np.zeros((len(full_df), len(MODEL_SPECS)), dtype=np.float32)
    test_base = np.zeros((len(test_df), len(MODEL_SPECS)), dtype=np.float32)
    saved_specs = []
    base_metrics = {}
    t0 = time.time()

    for model_idx, spec in enumerate(MODEL_SPECS):
        print(f"\nmodel {model_idx + 1}/{len(MODEL_SPECS)}: {spec['name']}")
        fold_entries = []
        test_fold_preds = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, y), start=1):
            print(f"  fold {fold}/{args.folds}")
            if spec["backend"] == "cuml_side_gpu":
                val_pred, test_pred, fold_model = train_cuml_side_gpu(
                    raw_texts[tr_idx],
                    y[tr_idx],
                    raw_texts[val_idx],
                    test_raw_texts,
                    spec,
                )
                fold_entries.append(fold_model)
                test_fold_preds.append(test_pred)
                oof[val_idx, model_idx] = val_pred
                continue

            vectorizer = make_vectorizer(spec)
            X_tr = vectorizer.fit_transform(texts[tr_idx])
            X_val = vectorizer.transform(texts[val_idx])
            X_test = vectorizer.transform(test_texts)
            if spec["backend"] == "torch":
                if not sparse.isspmatrix_csr(X_tr):
                    X_tr = X_tr.tocsr()
                val_pred, weight, bias = train_torch_linear(
                    X_tr, y[tr_idx], X_val, spec, device, args.batch_size
                )
                fold_entries.append(TorchLinearFold(vectorizer, weight, bias))

                fold_model = nn.Linear(X_test.shape[1], 1)
                with torch.no_grad():
                    fold_model.weight.copy_(torch.from_numpy(weight.reshape(1, -1)))
                    fold_model.bias.copy_(torch.tensor([bias], dtype=torch.float32))
                fold_model.to(device)
                test_fold_preds.append(predict_torch_linear(fold_model, X_test, device, args.batch_size))
                del fold_model
            elif spec["backend"] == "xgboost_gpu":
                val_pred, test_pred, clf = train_xgboost_gpu(X_tr, y[tr_idx], X_val, X_test, spec)
                fold_entries.append(SklearnTextFold(vectorizer, clf))
                test_fold_preds.append(test_pred)
            elif spec["backend"] == "cuml_tfidf_gpu":
                val_pred, test_pred, clf = train_cuml_tfidf_gpu(X_tr, y[tr_idx], X_val, X_test, spec)
                fold_entries.append(CuMLTextFold(vectorizer, clf))
                test_fold_preds.append(test_pred)
            elif spec["backend"] == "catboost_gpu":
                val_pred, test_pred, clf = train_catboost_gpu(X_tr, y[tr_idx], X_val, X_test, spec)
                fold_entries.append(SklearnTextFold(vectorizer, clf))
                test_fold_preds.append(test_pred)
            else:
                raise ValueError(f"unknown backend: {spec['backend']}")
            oof[val_idx, model_idx] = val_pred

            del X_tr, X_val, X_test
            if device.type == "cuda":
                torch.cuda.empty_cache()

        test_base[:, model_idx] = np.mean(test_fold_preds, axis=0)
        base_metrics[spec["name"]] = {
            "oof": metric_row(y, oof[:, model_idx]),
            "test": metric_row(y_test, test_base[:, model_idx]),
        }
        saved_specs.append({"name": spec["name"], "folds": fold_entries})
        print(f"  oof f1={base_metrics[spec['name']]['oof']['f1']:.4f}")

    side_scaler = StandardScaler()
    side_train = side_scaler.fit_transform(build_side_features(raw_texts))
    side_test = side_scaler.transform(build_side_features(test_raw_texts))

    X_meta = np.hstack([oof, side_train])
    X_meta_test = np.hstack([test_base, side_test])
    stage2_models, final_model, stage2_oof, stage2_test, oof_probs, test_probs = train_multistage_stack(
        X_meta, y, X_meta_test, args.folds
    )
    train_end = len(train_df)
    metrics = {
        "train": metric_row(y[:train_end], oof_probs[:train_end]),
        "dev": metric_row(y[train_end:], oof_probs[train_end:]),
        "test": metric_row(y_test, test_probs),
        "base_models": base_metrics,
        "stage2_models": {
            "xgb_l2": metric_row(y, stage2_oof[:, 0]),
            "catboost_l2": metric_row(y, stage2_oof[:, 1]),
            "cuml_lr_l2": metric_row(y, stage2_oof[:, 2]),
        },
        "side_features": SIDE_FEATURE_NAMES,
        "meta_features": [spec["name"] for spec in MODEL_SPECS] + SIDE_FEATURE_NAMES,
        "final_features": [spec["name"] for spec in MODEL_SPECS]
        + SIDE_FEATURE_NAMES
        + ["xgb_l2", "catboost_l2", "cuml_lr_l2"],
        "config": {
            "folds": args.folds,
            "batch_size": args.batch_size,
            "device": str(device),
            "model_specs": MODEL_SPECS,
            "train_rows_for_oof": int(len(full_df)),
            "test_rows": int(len(test_df)),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
    }

    bundle = OOFEnsembleSentimentModel(saved_specs, None, side_scaler, stage2_models, final_model)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nensemble")
    print(f"  oof  acc={metrics['train']['accuracy']:.4f} f1={metrics['train']['f1']:.4f}")
    print(f"  test acc={metrics['test']['accuracy']:.4f} f1={metrics['test']['f1']:.4f}")
    print(f"saved: {config.MODEL_PATH}")
    print(f"saved: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
