from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


def build_vectorizer():
    return TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
    )
