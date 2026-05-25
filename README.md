# Sentiment Analysis

Binary sentiment classifier for Amazon product reviews. Logistic regression implemented from scratch in numpy with TF-IDF features.

**Demo:** https://sentiment-analysis-kt9auxsyox4oa4hnwnn5zv.streamlit.app/

## How it works

One weight per word (or bigram) in a 30K vocabulary. For a review, sum the weights of the words present and push through a sigmoid:

| Positive | Negative |
|---|---|
| great `+7.87` | not `−7.01` |
| love `+4.21` | waste `−3.39` |
| excellent `+3.63` | bad `−3.10` |
| best `+3.51` | disappointed `−3.01` |
| perfect `+2.57` | worst `−2.75` |
| amazing `+2.04` | terrible `−2.22` |

```
python -m src.inspect_weights
```

## Math

```
z = w·x + b
ŷ = 1 / (1 + e^(-z))
J = -1/m · Σ [y·log ŷ + (1-y)·log(1-ŷ)] + (λ/2m)·||w||²
```

Trained with batch gradient descent + momentum (β=0.9) and L2 regularization. Forward/cost/gradient are written by hand in [src/model.py](src/model.py).

## Data

[`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) — 25K samples per class, stratified 80/10/10 train/dev/test.

## Results

| | train | dev | test |
|---|---|---|---|
| accuracy | 0.88 | 0.88 | 0.88 |
| f1 | 0.88 | 0.87 | 0.88 |

## Layout

```
app.py                              streamlit demo (deploy entrypoint)
src/
  config.py                         hyperparameters
  preprocessing.py                  text cleaning
  features.py                       tf-idf
  model.py                          logistic regression
  download_data.py                  fetch + sample dataset
  train.py                          train + evaluate + save
  predict.py                        cli inference
  inspect_weights.py                print learned word weights
models/sentiment_model.joblib       trained model
```

## Run

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.download_data
python -m src.train
streamlit run app.py
```
