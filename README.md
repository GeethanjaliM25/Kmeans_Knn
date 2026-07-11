

# 🧠 Breast Cancer Prediction using KNN & K_Means           
 
This project uses **Machine Learning algorithms — K-Nearest Neighbors (KNN) and K-Means clustering** — to analyze and predict breast cancer diagnosis based on medical data. 
It is built using **Python, Pandas, Scikit-learn, Matplotlib, and Seaborn**.
---

## 🚀 Project Objective

To build a machine learning system that:
- Predicts if a tumor is **Benign (0)** or **Malignant (1)** using **KNN**
- Groups the data into clusters using **K-Means**.
- Provides analysis, evaluation, and visualization of results. 

---

```

---

## 📊 Dataset Information

- **Rows:** 569
- **Columns:** 33 (after cleaning → 32 used)
- **Target column:** `diagnosis`
  - `M` → Malignant (1)
  - `B` → Benign (0)

Unused/removed column:
- `Unnamed: 32` (empty column – removed during preprocessing)

---

## 🛠 Technologies Used

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install requirements:

```bash
pip install -r requirements.txt
```

---

## ▶ How To Run

Go to the project folder in terminal:

```bash
cd knn_kmeans
```

Then run:

```bash
python src/train_model.py
```

If successful, you will see:

- ✅ Accuracy score
- ✅ Confusion matrix
- ✅ Classification report
- ✅ Saved models

---

## ✅ Output (Sample Result)

```text
🔵 KNN Accuracy: 0.9473 (94.7%)

Confusion Matrix:
[[68  3]
 [ 3 40]]

Classification Report:
 precision    recall  f1-score
 0.96        0.96    0.96
 0.93        0.93    0.93

All Models Saved Successfully!
```

This confirms:
✔ KNN is working  
✔ K-Means is working  
✔ Models saved successfully  

---

## 📌 Future Enhancements

- Add a **Streamlit Web App**
- Real-time prediction UI
- Graphical reports & dashboards
- Add more ML algorithms for comparison

---

## 👩‍💻 Author

**Geethanjali M**  
B.E Student | Machine Learning & Web Development Enthusiast  


---

---

⭐ *If you like this project, please give it a star on GitHub!* ⭐


