# Sentiment Analysis

Amazon 상품 리뷰를 **긍정(1) / 부정(0)** 으로 분류하는 미니 프로젝트입니다.
Andrew Ng 강의에서 배운 로지스틱 회귀를 numpy 로 직접 짜보고 싶어서 만들었고,
완성된 모델을 Streamlit 으로 띄워서 누구나 리뷰를 넣고 결과를 확인할 수 있게 했어요.

## 한 줄로 어떻게 동작하나

```
리뷰 텍스트 → (전처리) → TF-IDF 벡터 → sigmoid(w·x + b) → 확률 → 0 / 1
```

- **전처리**: 소문자화, HTML/URL 제거, 알파벳만 남기기
- **TF-IDF**: 단어와 두 단어 묶음(bigram) 중 자주 등장하는 2만 개를 피처로
- **로지스틱 회귀**: 0~1 사이 확률을 뱉는 함수를 학습. 0.5 이상이면 긍정으로 판단

## 원리 (수식)

이항 분류 logistic regression 은 결국 세 줄짜리 수학입니다.

**1) 예측 (Forward)**

```
z = w · x + b
ŷ = σ(z) = 1 / (1 + e^(-z))    ← 0~1 사이 확률
```

**2) 비용 함수 — Binary Cross-Entropy + L2 정규화**

```
J(w, b) = -1/m · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ] + (λ/2m)·||w||²
```

정답이면 `log(ŷ)` 가 0 에 가깝고, 틀리면 음의 무한대로 가서 페널티가 커지는 구조.

**3) 경사하강법 (Batch Gradient Descent)**

```
dw = 1/m · Xᵀ(ŷ - y) + (λ/m)·w
db = 1/m · Σ(ŷ - y)
w := w - α·dw,    b := b - α·db
```

`src/model.py` 의 `LogisticRegressionScratch` 에 이 세 줄이 그대로 들어있습니다.
scipy sparse matrix 를 받게 해서 5만 × 2만 피처에서도 빠르게 돌아갑니다.

## 데이터

[`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
— Amazon 리뷰 360만 건에 긍정/부정 라벨이 붙어있는 공개 데이터셋.

학습 시간 단축을 위해 **클래스당 25,000개씩 균형 샘플링** 해서 5만 건으로 학습 →
train 80% / dev 10% / test 10% 로 stratified split.

## 폴더 구조

```
.
├── app.py                       ← Streamlit 데모 (배포 진입점)
├── requirements.txt
├── data/processed/              ← train/dev/test CSV (학습 후 생성)
├── models/
│   ├── sentiment_model.joblib   ← 학습된 모델
│   └── metrics.json             ← 학습 결과 지표
└── src/
    ├── config.py                ← 모든 하이퍼파라미터
    ├── preprocessing.py         ← 텍스트 클리닝
    ├── features.py              ← TF-IDF
    ├── model.py                 ← 직접 구현한 LogisticRegression
    ├── download_data.py         ← HuggingFace 에서 받아서 샘플링
    ├── train.py                 ← 학습 + 평가 + 저장
    └── predict.py               ← CLI 추론
```

## 직접 돌려보기

```powershell
# 1. 의존성
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 데이터 다운로드 + 샘플링
python -m src.download_data

# 3. 학습 (CPU 에서 1~3분)
python -m src.train

# 4. CLI 로 한번 찍어보기
python -m src.predict "Love it, works exactly as described!"

# 5. Streamlit 데모 실행
streamlit run app.py
```

## 학습하면서 배운 것

- 처음엔 `learning_rate = 0.5`, `n_iters = 300` 으로 돌렸더니 정확도 85% 에서 멈춤. 학습 곡선을 보니 cost 가 0.693 → 0.658 로 거의 안 떨어진 상태였음.
- TF-IDF 는 값이 [0, 1] 범위라 gradient signal 이 작아서 학습률을 크게 줘야 한다는 걸 깨달음. `learning_rate = 5.0`, `n_iters = 1500`, `lambda_reg = 0.01` 로 바꾸니 90% 넘김.
- 즉, **하이퍼파라미터는 데이터의 스케일을 보고 정해야 한다**는 게 이번 프로젝트의 핵심 교훈.

## 결과

```
train  acc ≈ 0.90
dev    acc ≈ 0.90
test   acc ≈ 0.90
```

학습 곡선과 confusion matrix 는 앱의 **"Model metrics"** 탭에서 확인 가능.

## 배포

[share.streamlit.io](https://share.streamlit.io) 에 올려둔 데모 → *(URL 추가 예정)*

Streamlit Cloud 에 배포할 때는 `models/sentiment_model.joblib` 도 같이 커밋해야 합니다.
모델 파일이 ~400KB 정도라서 GitHub 100MB 제한과는 무관해요.

## 다음에 해볼 것

- [ ] scikit-learn 의 `LogisticRegression` 으로 같은 데이터 학습해서 직접 구현 버전과 정확도 비교 (sanity check)
- [ ] Word2Vec / fastText 임베딩으로 피처 교체
- [ ] 별점 (1~5) 회귀로 확장
- [ ] 한국어 데이터 (네이버 쇼핑 리뷰) 로 옮기기 — 형태소 분석기 + `TfidfVectorizer(tokenizer=...)` 만 교체하면 됨
