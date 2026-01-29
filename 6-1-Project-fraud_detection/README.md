# 💳 Credit Card Fraud Detection: From Linear Models to Deep Learning

This repository presents a comprehensive machine learning workflow designed to solve the "Needle in a Haystack" problem of financial fraud detection. The project explores the transition from traditional statistical models to deep learning architectures, specifically handling extreme class imbalance.

---

## 🚀 Project Overview
In financial transaction data, fraud typically accounts for a tiny fraction of total activity. In this dataset, only **0.17%** of transactions are fraudulent. 

The primary challenge addressed in this project is why standard "Accuracy" is a deceptive metric (achieving 99.8% while missing actual fraud) and how to optimize for **Recall** and **Precision** using advanced resampling and threshold tuning techniques.

---

## 📂 Repository Structure
As seen in the project environment, the repository includes:
* **`creditcard.csv`**: The source dataset containing PCA-transformed features ($V1$–$V28$), $Amount$, and the target $Class$.
* **`Fraud_detection_Latreche.ipynb`**: The main notebook containing full EDA, model training, and evaluation.
* **`Fraud_detection_questions.ipynb`**: A lab-style version with guided exercises for students to explore the dataset and models.
* **`Fraud_detection_solutions.ipynb`**: The corresponding solution key for the lab exercises.
* **`app.py`**: A Plotly Dash application for interactive fraud monitoring and threshold visualization.

---

## 🧠 Technical Workflow

### 1. Baseline: Logistic Regression
* **Standard LR**: Achieving high accuracy but failing to catch a significant portion of fraud.
* **Balanced LR**: Implementing `class_weight='balanced'` and optimizing the decision threshold to prioritize detection over raw accuracy.

### 2. Deep Learning: MLP + SMOTE
To capture non-linear relationships, we implement a **Multi-Layer Perceptron (MLP)**.



* **Input Layer**: Features $x_1$ through $x_4$ (and others) are ingested from the transaction table.
* **Hidden Layers**: Two dense layers (64 and 32 neurons) with **ReLU** or **Tanh** activations process the signals.
* **SMOTE Resampling**: Since the MLP initially struggles with the 0.17% imbalance, we use **Synthetic Minority Over-sampling Technique (SMOTE)** to generate synthetic fraud samples for the training phase.
* **Output Layer**: Produces a probability score $f(X)$ used to trigger "Fraud Alerts".

---

## 📊 Performance Comparison
| Model | Recall | Precision | Business Insight |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Std) | 0.64 | 0.85 | Misses 36% of fraud cases. |
| Logistic Regression (Balanced) | 0.92 | 0.06 | High detection, but floods investigators with false alarms. |
| **MLP + SMOTE (Optimized)** | **0.81** | **0.56** | **Optimal balance: high detection with low false positives**. |



---

## 🛠️ Installation
```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn plotly dash