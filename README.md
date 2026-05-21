# Sentiment Analysis

A binary classifier that predicts whether an Amazon product review is **positive (1)** or **negative (0)**.
I built this after taking Andrew Ng's Machine Learning course — I wanted to implement logistic regression
from scratch in numpy instead of just calling scikit-learn. The trained model is deployed on
[Streamlit](https://share.streamlit.io) so anyone can paste in a review and see the prediction.

> **Live demo:** https://sentiment-analysis-kt9auxsyox4oa4hnwnn5zv.streamlit.app/

---

## How it works

In one sentence: **the model learns a +/- score for each word, then sums up the scores of words present in the review to make a decision.**

Top weighted words my trained model learned:

| Positive words (score) | Negative words (score) |
|---|---|
| great `+7.87` | not `−7.01` |
| love `+4.21` | waste `−3.39` |
| excellent `+3.63` | bad `−3.10` |
| best `+3.51` | disappointed `−3.01` |
| perfect `+2.57` | worst `−2.75` |
| amazing `+2.04` | terrible `−2.22` |

When `great` appears, the score is pushed +7.87 toward "positive"; when `terrible` appears, it gets pushed −2.22 toward "negative". The total score goes through a sigmoid to produce a probability in [0, 1], and we threshold at 0.5.

Examples:

```
"This is a great product, I love it!"
  → great(+7.87) + love(+4.21) + small contributions = z ≈ +12
  → sigmoid(12) ≈ 0.999 → Positive (1)

"Waste of money, terrible quality."
  → waste(−3.39) + money(−3.43) + terrible(−2.22) + ... ≈ −9
  → sigmoid(−9) ≈ 0.0001 → Negative (0)
```

To see the actual weights yourself:

```powershell
python -m src.inspect_weights
```

---

## The math (straight from Andrew Ng's course)

Binary logistic regression is essentially three lines of math.

**1) Forward**
```
z = w · x + b
ŷ = σ(z) = 1 / (1 + e^(-z))    ← probability in [0, 1]
```

**2) Cost — Binary Cross-Entropy + L2 regularization**
```
J(w, b) = -1/m · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ] + (λ/2m)·||w||²
```

**3) Gradient Descent + Momentum**
```
dw = 1/m · Xᵀ(ŷ - y) + (λ/m)·w
v_w := β·v_w + (1-β)·dw         ← velocity (momentum)
w   := w - α·v_w
```

`src/model.py` (the `LogisticRegressionScratch` class) implements exactly these three lines. It accepts scipy sparse matrices so it stays fast on 50K × 30K features.

---

## Data

[`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
— 3.6M Amazon reviews labeled positive/negative.

To keep training fast, I balanced-sampled 25,000 reviews per class (50K total) and split it into train 80% / dev 10% / test 10% (stratified).

---

## Results

| Metric | train | dev | test |
|---|---|---|---|
| accuracy | 0.88 | 0.88 | 0.88 |
| f1 | 0.88 | 0.87 | 0.88 |

Train ↔ dev gap is ~1 percentage point — no real overfitting.

---

## Project structure

```
.
├── app.py                            ← Streamlit demo (deployment entry point)
├── requirements.txt
├── models/sentiment_model.joblib     ← trained model
└── src/
    ├── config.py                     ← hyperparameters
    ├── preprocessing.py              ← text cleaning
    ├── features.py                   ← TF-IDF vectorizer
    ├── model.py                      ← from-scratch LogisticRegression
    ├── download_data.py              ← downloads + samples from HuggingFace
    ├── train.py                      ← training + evaluation + saving
    ├── predict.py                    ← CLI inference
    └── inspect_weights.py            ← print learned word scores
```

---

## Running locally

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download data and train
python -m src.download_data
python -m src.train

# Inspect what the model learned
python -m src.inspect_weights

# Launch the Streamlit demo
streamlit run app.py
```

---

## What I learned

- **Set hyperparameters based on the scale of your features.** My first run used `lr=0.5, n_iters=300`, but cost only dropped from 0.693 → 0.658 (5%) — the model was barely training. TF-IDF values live in [0, 1], so gradients are small. Bumping to `lr=5.0, n_iters=1500` pushed accuracy from 85% to 89%.
- **Momentum doesn't change the effective learning rate.** With β=0.9, the EMA-style velocity converges to the gradient at steady state — so the per-step magnitude is the same as without momentum. I incorrectly lowered the LR when adding momentum the first time, which slowed training down.
- **Trigrams hurt more than they help on this dataset.** Most trigrams are too sparse, so they add noise rather than signal. `(1, 2)`-grams is the sweet spot.

## What I'd try next

- Compare with scikit-learn's `LogisticRegression` (L-BFGS) on the same data
- Swap TF-IDF for Word2Vec / fastText embeddings
- Extend to star-rating regression (1~5)
- Port to Korean review data (Naver Shopping) — just swap in a KoNLPy tokenizer via `TfidfVectorizer(tokenizer=...)`
