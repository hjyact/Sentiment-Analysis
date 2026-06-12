from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, max_df=0.7)
    return vectorizer.fit_transform(texts)