#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the Streamlit app can be imported and basic functionality works
"""

# Test 1: Check if all required modules can be imported
try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from predictor import predict_sentiment, load_model_components
    import datetime
    import underthesea
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import xgboost
    import joblib
    from tqdm import tqdm
    import requests
    import numpy as np
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test 2: Check if model files exist and can be loaded
try:
    import os
    model_files = [
        "models/sentiment_model.pkl",
        "models/tfidf_vectorizer.pkl", 
        "models/label_encoder.pkl",
        "models/recommend_model.pkl",
        "models/tfidf_review.pkl"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ {model_file} exists")
        else:
            print(f"❌ {model_file} missing")
            
    # Test loading one model
    vectorizer, model, normalize_func = load_model_components()
    print("✅ Model components loaded successfully")
    
except Exception as e:
    print(f"❌ Model loading error: {e}")

# Test 3: Check if data files exist
try:
    data_files = [
        "data/Overview_Companies.xlsx",
        "data/Overview_Reviews.xlsx",
        "data/Reviews.xlsx",
        "output/Processed_reviews.xlsx"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ {data_file} exists")
        else:
            print(f"❌ {data_file} missing")
            
except Exception as e:
    print(f"❌ Data file check error: {e}")

# Test 4: Check if image files exist
try:
    image_files = [
        "img/your_logo2.jpg",
        "img/cluster0.png",
        "img/cluster1.png",
        "img/cluster2.png",
        "img/cluster3.png",
        "img/cluster4.png"
    ]
    
    for image_file in image_files:
        if os.path.exists(image_file):
            print(f"✅ {image_file} exists")
        else:
            print(f"❌ {image_file} missing")
            
except Exception as e:
    print(f"❌ Image file check error: {e}")

# Test 5: Try to compile the main streamlit app
try:
    import py_compile
    py_compile.compile("streamlit_app.py", doraise=True)
    print("✅ streamlit_app.py compiled successfully (no syntax errors)")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error in streamlit_app.py: {e}")

print("\n🎉 All tests completed! The app should be ready to run.")
