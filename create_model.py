#!/usr/bin/env python3
"""
Create a model for the Docker container.
Try to train with real data if available, otherwise create a dummy model.
"""
import os
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

MODEL_PATH = 'app/models/churn_pipeline.pkl'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Try to train with real data, fallback to dummy model
try:
    if os.path.exists('data/train.csv'):
        print("Found training data, attempting to train real model...")
        import sys
        sys.path.insert(0, '/app')
        from app.services.train_model import main
        main('data/train.csv', MODEL_PATH)
        print(f"Successfully trained model at {MODEL_PATH}")
    else:
        raise FileNotFoundError('No training data found')
except Exception as e:
    print(f'Training failed or no data: {e}. Creating dummy model.')
    # Create a minimal classifier that supports predict_proba
    clf = Pipeline([("clf", DummyClassifier(strategy="prior"))])
    X = np.zeros((2, 19))  # 19 features expected by the API
    y = np.array([0, 1])
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    print(f'Created dummy model at {MODEL_PATH}')