import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# -----------------------------
# 1. Load Dataset
# -----------------------------
csv_path = "data/breast_cancer.csv"
df = pd.read_csv(csv_path)

print("Dataset Loaded Successfully!")
print(df.head())
print("\nShape =", df.shape)

# -----------------------------
# 2. Drop useless column
# -----------------------------
df = df.drop(columns=["Unnamed: 32"], errors="ignore")

print("\nDropped 'Unnamed: 32' column (empty column removed).")
print("Shape after drop:", df.shape)

# -----------------------------
# 3. Convert target column (M/B → 1/0)
# -----------------------------
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
df[df.columns] = imputer.fit_transform(df)

print("\nMissing Values Fixed!")

# -----------------------------
# 5. Heatmap (numeric only)
# -----------------------------
df_numeric = df.select_dtypes(include=["number"])

plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 6. Count Plot
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=df["diagnosis"])
plt.title("Cancer Count (0 = Benign, 1 = Malignant)")
plt.show()

# -----------------------------
# 7. Split Data
# -----------------------------
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nData Split Done!")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# -----------------------------
# 8. Train KNN Classifier
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)

print("\n🔵 KNN Accuracy:", knn_acc)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# -----------------------------
# 9. Train KMeans
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df["Cluster"] = kmeans.labels_

# -----------------------------
# 10. Cluster Visualization
# -----------------------------
plt.figure(figsize=(6, 6))
df["Cluster"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("KMeans Cluster Distribution")
plt.show()

# -----------------------------
# 11. Save Models
# -----------------------------
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(imputer, "models/imputer.pkl")

print("\nAll Models Saved Successfully!")
