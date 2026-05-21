"""Hand-built Logistic Regression — Andrew Ng style.

The math, mirroring the Coursera ML course:

    z = X @ w + b
    a = sigmoid(z) = 1 / (1 + exp(-z))

    Cost (binary cross-entropy with L2 regularization):
        J(w, b) = -(1/m) * sum( y*log(a) + (1-y)*log(1-a) )
                  + (lambda / 2m) * sum(w**2)

    Gradients:
        dw = (1/m) * X.T @ (a - y) + (lambda / m) * w
        db = (1/m) * sum(a - y)

    Update (batch gradient descent):
        w := w - alpha * dw
        b := b - alpha * db

We support sparse X (TF-IDF output is a scipy.sparse matrix) so the dot
products stay fast on a 50K x 20K feature matrix.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid: split positive / negative branches
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegressionScratch:
    """Binary logistic regression with batch gradient descent + L2.

    Supports momentum (Andrew Ng DL Specialization, Course 2):
        v_w := beta * v_w + (1 - beta) * dw
        w   := w - alpha * v_w
    Momentum smooths the update direction so we don't zig-zag, which
    lets the same iteration budget reach a lower cost. Set momentum=0
    to recover plain batch gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iters: int = 300,
        lambda_reg: float = 1.0,
        momentum: float = 0.9,
        print_every: int = 20,
        verbose: bool = True,
    ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.print_every = print_every
        self.verbose = verbose

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[dict] = []

    @staticmethod
    def _dot(X, w):
        # Works for both numpy arrays and scipy.sparse matrices
        if sparse.issparse(X):
            return X.dot(w)
        return X @ w

    def _forward(self, X) -> np.ndarray:
        return _sigmoid(self._dot(X, self.w) + self.b)

    def _cost(self, a: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        eps = 1e-12
        ce = -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
        reg = (self.lambda_reg / (2 * m)) * np.sum(self.w ** 2)
        return float(ce + reg)

    def fit(self, X, y, X_dev=None, y_dev=None):
        y = np.asarray(y).astype(np.float64).ravel()
        n_features = X.shape[1]
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        # Velocity terms for momentum (initialized to zero)
        v_w = np.zeros(n_features, dtype=np.float64)
        v_b = 0.0
        beta = self.momentum
        m = X.shape[0]

        for it in range(1, self.n_iters + 1):
            a = self._forward(X)                              # (m,)
            error = a - y                                     # (m,)

            # dw = (1/m) X^T (a - y) + (lambda/m) w
            if sparse.issparse(X):
                dw = (X.T.dot(error)) / m
            else:
                dw = (X.T @ error) / m
            dw = np.asarray(dw).ravel() + (self.lambda_reg / m) * self.w
            db = float(np.mean(error))

            # Momentum update: v := beta*v + (1-beta)*grad ;  param := param - alpha*v
            v_w = beta * v_w + (1.0 - beta) * dw
            v_b = beta * v_b + (1.0 - beta) * db
            self.w -= self.learning_rate * v_w
            self.b -= self.learning_rate * v_b

            if it == 1 or it % self.print_every == 0 or it == self.n_iters:
                train_cost = self._cost(a, y)
                row = {"iter": it, "train_cost": train_cost}
                if X_dev is not None and y_dev is not None:
                    a_dev = self._forward(X_dev)
                    row["dev_cost"] = self._cost(a_dev, np.asarray(y_dev).astype(np.float64))
                    row["dev_acc"] = float(np.mean((a_dev >= 0.5) == y_dev))
                self.history.append(row)
                if self.verbose:
                    msg = f"  iter {it:4d}  train_cost={train_cost:.4f}"
                    if "dev_cost" in row:
                        msg += f"  dev_cost={row['dev_cost']:.4f}  dev_acc={row['dev_acc']:.4f}"
                    print(msg)

        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


class SentimentModel:
    """Bundle of vectorizer + classifier — what we persist to disk.

    Kept separate from the raw LogisticRegressionScratch so the Streamlit
    app can load a single object and call .predict_proba(["text", ...]).
    """

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        from .preprocessing import clean_text
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        return self.classifier.predict_proba(X)

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(np.int64)
