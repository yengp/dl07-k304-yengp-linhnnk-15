import pickle
from underthesea import word_tokenize
import re
import predictor



def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = word_tokenize(text, format="text")
    return text

def load_model_components():
    with open("models/tfidf_review.pkl", "rb") as f1:
        vectorizer = pickle.load(f1)
    with open("models/recommend_model.pkl", "rb") as f2:
        model = pickle.load(f2)
    return vectorizer, model, normalize_text

def predict_sentiment(text, vectorizer, model, normalize_func):
    clean_text = normalize_func(text)
    vector = vectorizer.transform([clean_text])
    proba = model.predict_proba(vector)[0]
    label = model.classes_[proba.argmax()]
    return label, proba.max()
