# Predictive Risk Modelling: Binary Classification of Health Outcomes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Lib-Pandas%20%7C%20Scikit--Learn%20%7C%20Matplotlib-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📋 Executive Summary
This project demonstrates the end-to-end development of a **Supervised Machine Learning pipeline** to predict high-risk binary outcomes (Diabetes: Yes/No) using large-scale unstructured public data.

1.  Ingesting and cleaning high-volume, noisy datasets.
2.  Handling **class imbalance** (where risk cases are the minority).
3.  Optimising for **Recall** to minimize False Negatives (critical in risk identification).
4.  Deriving feature importance to provide explainable insights for decision-making.

---

## 🎯 The Business Challenge
**Objective:** Identify individuals at high risk of a specific outcome based on behavioral and demographic indicators.

**The Data:** The dataset (CDC BRFSS) contains 250,000+ records with significant noise, missing values, and categorical variables. The challenge mirrors **KYC (Know Your Customer)** and **Credit Scoring** scenarios where data quality is variable and feature selection is critical.

---

## 🛠️ Technical Implementation

### 1. Data Cleaning & Preprocessing (ETL)
* **Handling Missingness:** Implemented logic to handle non-responses and null values without losing statistical power.
* **Feature Engineering:** Transformed categorical survey responses into numerical inputs suitable for regression and tree-based models.
* **Normalization:** Applied scaling to continuous variables (BMI, Age) to ensure model convergence.

### 2. Exploratory Data Analysis (EDA)
* Analyzed correlation matrices to identify multicollinearity between risk factors.
* Visualized distribution of target variables to assess class imbalance.

### 3. Modelling & Algorithms
Tested and benchmarked multiple classifiers to find the optimal balance between bias and variance:
* **Logistic Regression:** Established a baseline for interpretability (Odds Ratios).
* **Random Forest / Decision Trees:** Captured non-linear relationships between risk factors.
* **Gradient Boosting (XGBoost/LGBM):** Optimized for predictive accuracy.

---

## 📊 Key Results & Impact

| Metric | Result | Context |
| :--- | :--- | :--- |
| **Accuracy** | ~86% | Overall correctness of the model. |
| **Recall (Sensitivity)** | High | **Critical Metric:** Prioritized catching positive cases (Risk) over avoiding False Positives. |
| **Precision** | Balanced | Ensured the model didn't flag 'safe' individuals excessively. |

**Top Risk Indicators Identified:**
* High Blood Pressure (Correlated with Debt-to-Income ratio logic in finance).
* BMI / General Health Score.
* Age Demographics.

*(Note: These indicators allow stakeholders to target intervention strategies effectively, similar to targeted financial products.)*

---

🚀 How to Run
Clone the repo:

Bash

git clone [https://github.com/chanyinkwan/Predictive-Modeling-of-Diabetes-Risk-Factors.git](https://github.com/chanyinkwan/Predictive-Modeling-of-Diabetes-Risk-Factors.git)
Install dependencies:

Bash

pip install -r requirements.txt
Open the Jupyter Notebook:

Bash

jupyter notebook notebooks/main_analysis.ipynb

📬 Contact
Yin Kwan Chan

Role: Technical Business Analyst / Data Analyst

Location: United Kingdom

[LinkedIn Profile](https://www.linkedin.com/in/kessog-chan-05591a2b3/)

Open to discussion regarding Data Analysis, Risk Modelling, and FinTech opportunities.

---

## 📂 Repository Structure
```bash
├── data/               # Raw and processed datasets (GitIgnore applied to large files)
├── notebooks/          # Jupyter Notebooks (EDA, Modeling, Evaluation)
├── graph/             # Visualizations and Confusion Matrices
└── README.md           # Project documentation
