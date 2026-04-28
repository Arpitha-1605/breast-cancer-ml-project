# Breast Cancer Classification using Machine Learning

## 📌 Project Overview
This project applies multiple **supervised and unsupervised machine learning algorithms** to classify tumors as **Malignant** or **Benign** using the Breast Cancer dataset.  
It also evaluates the impact of **dimensionality reduction techniques (PCA & LDA)** on model performance.

---

## 📊 Dataset
- Breast Cancer Wisconsin Dataset (from sklearn)
- Total samples: 569
- Features: 30 numerical features
- Target:
  - 0 → Malignant (Cancerous)
  - 1 → Benign (Non-cancerous)

---

## 🤖 Machine Learning Models Used
### 🔹 Supervised Learning
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest

### 🔹 Unsupervised Learning
- K-Means Clustering

---

## 🔽 Dimensionality Reduction
- **PCA (Principal Component Analysis)**  
  Reduces feature dimensions without using labels.

- **LDA (Linear Discriminant Analysis)**  
  Reduces dimensions while preserving class separability.

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📊 Model Comparison
Model performance is evaluated across:
- Original Dataset
- PCA-transformed Dataset
- LDA-transformed Dataset

An **interactive Plotly graph** is used to compare accuracies.

---

## 📉 Visualization
- Confusion Matrix Heatmaps
- Model Accuracy Comparison (Interactive)
- K-Means Clustering Graph

---

## 🔍 Key Observations
- Random Forest and SVM achieved high accuracy.
- PCA slightly reduced accuracy due to information loss.
- LDA maintained or improved performance by preserving class separation.
- K-Means clustering showed meaningful grouping of data points.

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
python main.py
