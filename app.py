import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.model import LogisticRegressionScratch, SentimentModel  # noqa: F401  (unpickling)


st.set_page_config(page_title="Sentiment Analysis", layout="centered")


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


st.title("Sentiment Analysis")
st.caption("Amazon product reviews — positive (1) vs. negative (0)")

bundle = load_model()
metrics = load_metrics()

if bundle is None:
    st.error("Trained model not found. Run `python -m src.download_data` then `python -m src.train`.")
    st.stop()

tab_single, tab_batch, tab_metrics = st.tabs(["Single", "Batch (CSV)", "Metrics"])

with tab_single:
    examples = {
        "(custom)": "",
        "Positive": "Absolutely love this product. Fast shipping, great quality, exactly as described.",
        "Negative": "Total waste of money. Broke after two days and customer service ignored me.",
        "Mixed": "Decent for the price but packaging was damaged and it's smaller than I expected.",
    }
    pick = st.selectbox("Example", list(examples.keys()))
    text = st.text_area("Review", value=examples[pick], height=160)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter a review.")
        else:
            prob = float(bundle.predict_proba([text])[0])
            label = "Positive" if prob >= threshold else "Negative"
            color = "#16a34a" if prob >= threshold else "#dc2626"
            st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
            st.progress(prob)
            c1, c2 = st.columns(2)
            c1.metric("P(positive)", f"{prob:.3f}")
            c2.metric("P(negative)", f"{1 - prob:.3f}")

with tab_batch:
    st.write("Upload a CSV with a `text` column.")
    up = st.file_uploader("CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
        else:
            if "text" not in df.columns:
                st.error("CSV must contain a `text` column.")
            else:
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

with tab_metrics:
    if metrics is None:
        st.info("No metrics file. Run `python -m src.train`.")
    else:
        cols = st.columns(3)
        for c, split in zip(cols, ["train", "dev", "test"]):
            m = metrics.get(split, {})
            c.metric(f"{split} acc", f"{m.get('accuracy', 0):.4f}")
            c.write(f"f1: {m.get('f1', 0):.4f}")

        if metrics.get("history"):
            hist = pd.DataFrame(metrics["history"])
            cols_to_plot = [c for c in ["train_cost", "dev_cost"] if c in hist.columns]
            if cols_to_plot:
                st.line_chart(hist.set_index("iter")[cols_to_plot])

        with st.expander("config"):
            st.json(metrics.get("config", {}))
