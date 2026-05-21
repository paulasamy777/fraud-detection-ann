# 💳 AI Fraud Detection System

A deep learning model that detects fraudulent credit card transactions
using an Artificial Neural Network (ANN) built with TensorFlow and
deployed as an interactive web app with Streamlit.

---

## 🔍 About the Project

Credit card fraud is extremely rare but costly. Standard ML models fail
on such imbalanced data — they predict everything as legitimate and still
get 99% accuracy. This project solves that using Focal Loss and
Precision-Recall threshold optimization to maximize fraud detection.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Matplotlib / Seaborn
- Joblib

---

## 🚀 How to Run

pip install -r requirements.txt
streamlit run app.py

---

## 🧠 Model Highlights

- 4-layer ANN with Dropout regularization (prevents overfitting)
- Focal Loss for class imbalance (only 0.17% of transactions are fraud)
- Optimized classification threshold via Precision-Recall curve
- Evaluated with PR-AUC — not just accuracy

---

## 📊 Dataset

Credit Card Fraud Detection dataset from Kaggle.
284,807 transactions — only 492 are fraudulent (0.17%).
Features: Time, V1–V28 (PCA anonymized), Amount, Class.

streamlit run app.py
