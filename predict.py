import sys
import os
import pickle
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.preprocessing import load_and_preprocess

# Load models
kmeans = pickle.load(open(os.path.join(ROOT_DIR, "models/kmeans_model.pkl"), "rb"))
knn = pickle.load(open(os.path.join(ROOT_DIR, "models/knn_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(ROOT_DIR, "models/scaler.pkl"), "rb"))

# Example input (replace with real values)
example = np.array([[14.5, 20.3, 96.0, 660.0, 0.091, 0.121, 0.240, 0.110]])

example_scaled = scaler.transform(example)

# Add cluster
cluster = kmeans.predict(example_scaled)
example_final = np.column_stack((example_scaled, cluster))

prediction = knn.predict(example_final)

if prediction[0] == 1:
    print("⚠ Malignant (Cancer Detected)")
else:
    print("✔ Benign (No Cancer)")
