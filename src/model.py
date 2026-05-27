import numpy as np
from scipy import sparse


def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.5, n_iters=300, lambda_reg=1.0,
                 momentum=0.9, print_every=20, verbose=True):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.print_every = print_every
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.history = []

    def _predict_raw(self, X):
        z = X.dot(self.w) if sparse.issparse(X) else X @ self.w
        return sigmoid(z + self.b)

    def _cost(self, a, y):
        eps = 1e-12
        ce = -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
        reg = self.lambda_reg / (2 * len(y)) * (self.w ** 2).sum()
        return float(ce + reg)

    def fit(self, X, y, X_dev=None, y_dev=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        v_w = np.zeros(n)
        v_b = 0.0
        beta = self.momentum

        for i in range(1, self.n_iters + 1):
            a = self._predict_raw(X)
            err = a - y
            dw = (X.T.dot(err) if sparse.issparse(X) else X.T @ err) / m
            dw = np.asarray(dw).ravel() + self.lambda_reg / m * self.w
            db = float(err.mean())

            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db
            self.w -= self.learning_rate * v_w
            self.b -= self.learning_rate * v_b

            if i == 1 or i % self.print_every == 0 or i == self.n_iters:
                row = {"iter": i, "train_cost": self._cost(a, y)}
                if X_dev is not None and y_dev is not None:
                    a_dev = self._predict_raw(X_dev)
                    y_dev_f = np.asarray(y_dev, dtype=np.float64)
                    row["dev_cost"] = self._cost(a_dev, y_dev_f)
                    row["dev_acc"] = float(((a_dev >= 0.5) == y_dev).mean())
                self.history.append(row)
                if self.verbose:
                    line = f"iter {i:5d}  train_cost={row['train_cost']:.4f}"
                    if "dev_cost" in row:
                        line += f"  dev_cost={row['dev_cost']:.4f}  dev_acc={row['dev_acc']:.4f}"
                    print(line)
        return self

    def predict_proba(self, X):
        return self._predict_raw(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)


class SklearnLR:
    """Thin wrapper so predict_proba returns 1-D (positive class only)."""

    def __init__(self, C=1.0, max_iter=1000):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self.w = None
        self.b = 0.0
        self.history = []

    def fit(self, X, y, X_dev=None, y_dev=None):
        self.clf.fit(X, y)
        self.w = self.clf.coef_.ravel()
        self.b = float(self.clf.intercept_[0])
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)


class SentimentModel:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts):
        from .preprocessing import clean_text
        X = self.vectorizer.transform([clean_text(t) for t in texts])
        return self.classifier.predict_proba(X)

    def predict(self, texts, threshold=0.5):
        return (self.predict_proba(texts) >= threshold).astype(np.int64)
