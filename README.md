# Hybrid Breast Cancer Detection (K-Means + KNN)

This project builds a hybrid ML model combining K-Means clustering and 
KNN classification on the Breast Cancer Wisconsin dataset.

## Steps Used
1. Load and preprocess dataset
2. Apply K-Means clustering
3. Add cluster label as a new feature
4. Train KNN classifier
5. Save models (KMeans + KNN + scaler)

## How to Run
1. Place dataset in data/breast_cancer.csv
2. Run: python src/train_model.py
3. Predict using: python src/predict.py
