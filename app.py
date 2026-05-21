"""Streamlit demo for the sentiment classifier.

Deployable to share.streamlit.io — entrypoint must sit at repo root.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.model import SentimentModel, LogisticRegressionScratch  # noqa: F401 (needed for unpickling)


# ----------------------------- page setup ----------------------------- #
st.set_page_config(
    page_title="Sentiment Analysis — Amazon Reviews",
    page_icon=":speech_balloon:",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not config.MODEL_PATH.exists():
        return None
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def load_metrics():
    if not config.METRICS_PATH.exists():
        return None
    with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------- UI --------------------------------- #
st.title("Sentiment Analysis")
st.caption("Amazon product reviews — positive (1) vs. negative (0)")

bundle = load_model()
metrics = load_metrics()

if bundle is None:
    st.error(
        "Trained model not found.\n\n"
        "Run locally first:\n"
        "```\n"
        "pip install -r requirements.txt\n"
        "python -m src.download_data\n"
        "python -m src.train\n"
        "```\n"
        f"This produces `{config.MODEL_PATH.relative_to(config.ROOT)}`, "
        "which the app loads on startup."
    )
    st.stop()

tab_predict, tab_batch, tab_metrics = st.tabs(["Single review", "Batch (CSV)", "Model metrics"])

# ---------- Single review ---------- #
with tab_predict:
    st.subheader("Try a review")
    examples = {
        "(custom)": "",
        "Positive example": "Absolutely love this product. Fast shipping, great quality, exactly as described.",
        "Negative example": "Total waste of money. Broke after two days and customer service ignored me.",
        "Mixed example": "Decent product for the price but the packaging was damaged and it's a bit smaller than I expected.",
    }
    pick = st.selectbox("Example", list(examples.keys()))
    default_text = examples[pick]
    text = st.text_area("Review text", value=default_text, height=160,
                        placeholder="Paste a product review here...")

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                          help="Probability cutoff for predicting Positive (1).")

    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter a review.")
        else:
            prob = float(bundle.predict_proba([text])[0])
            label = "Positive (1)" if prob >= threshold else "Negative (0)"
            color = "#16a34a" if prob >= threshold else "#dc2626"

            st.markdown(
                f"<h3 style='color:{color};margin-top:1rem'>{label}</h3>",
                unsafe_allow_html=True,
            )
            st.progress(prob)
            c1, c2 = st.columns(2)
            c1.metric("P(positive)", f"{prob:.3f}")
            c2.metric("P(negative)", f"{1 - prob:.3f}")

            with st.expander("How was this decided?"):
                st.write(
                    "The text is cleaned (lowercase, punctuation stripped), "
                    "converted into a TF-IDF feature vector, then scored by a "
                    "hand-built logistic regression model "
                    "(numpy gradient descent, L2 regularization)."
                )

# ---------- Batch CSV ---------- #
with tab_batch:
    st.subheader("Batch prediction")
    st.write("Upload a CSV with a `text` column. We'll add a `prediction` and `prob_positive` column.")

    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
        else:
            if "text" not in df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                with st.spinner(f"Scoring {len(df):,} rows ..."):
                    probs = bundle.predict_proba(df["text"].astype(str).tolist())
                    df["prob_positive"] = np.round(probs, 4)
                    df["prediction"] = (probs >= 0.5).astype(int)
                st.success(f"Done. Positive rate: {df['prediction'].mean():.1%}")
                st.dataframe(df.head(50), use_container_width=True)
                st.download_button(
                    "Download predictions",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# ---------- Metrics ---------- #
with tab_metrics:
    st.subheader("Model performance")
    if metrics is None:
        st.info("No metrics file found. Re-run `python -m src.train` to generate it.")
    else:
        cols = st.columns(3)
        for c, split in zip(cols, ["train", "dev", "test"]):
            m = metrics.get(split, {})
            c.metric(f"{split.upper()} accuracy", f"{m.get('accuracy', 0):.4f}")
            c.write(f"precision: {m.get('precision', 0):.4f}")
            c.write(f"recall:    {m.get('recall', 0):.4f}")
            c.write(f"f1:        {m.get('f1', 0):.4f}")

        if metrics.get("history"):
            st.write("**Training history**")
            hist_df = pd.DataFrame(metrics["history"])
            st.line_chart(hist_df.set_index("iter")[
                [c for c in ["train_cost", "dev_cost"] if c in hist_df.columns]
            ])

        with st.expander("Hyperparameters"):
            st.json(metrics.get("config", {}))

st.markdown("---")
st.caption(
    "Built with TF-IDF + hand-rolled logistic regression "
    "(based on Andrew Ng's ML course). Source: github.com/<you>/sentiment-analysis"
)
