# Credit Card Fraud Detection

## Overview

Credit card fraud detection is a critical concern for financial institutions, aiming to minimize financial losses and protect customer trust. This project delivers a robust machine learning solution to identify fraudulent transactions from a highly imbalanced dataset, typical of real-world scenarios.

## Business Problem

Financial institutions face significant losses and reputational damage from undetected fraudulent transactions. The main business objectives are:
- **Maximize detection of fraud** while **minimizing disruption to genuine customers** (false positives).
- **Automate and scale** the detection process for real-time decision-making.

## Project Pipeline

### 1. Data Preprocessing

**Key Steps:**
- **Data Cleaning:** Verified no missing values or obvious outliers.
- **Feature Preparation:** Since features are PCA-transformed for confidentiality, only `Time`, `Amount`, and `Class` are directly interpretable.
- **Feature Scaling:** Used `StandardScaler` to normalize `Amount` and `Time` for consistent model input.
- **Class Imbalance Handling:** 
  - Initially experimented with **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
  - Ultimately, adopted the model's **class_weight='balanced'** approach, letting algorithms internally adjust for class imbalance based on sample frequency.

**Why Class Weight over SMOTE?**
- **SMOTE** can sometimes introduce synthetic patterns not present in the real data, which may lead to overfitting or unrealistic fraud detection.
- **Class weighting** is less intrusive, making the model more generalizable by penalizing misclassification of the minority class more during training.

### 2. Exploratory Data Analysis (EDA) & Insights

- **Imbalance Confirmed:** Fraud cases are less than 0.2% of all transactions.
- **Correlation Analysis:** Due to PCA, features are mostly uncorrelated.

**Insight:** The imbalance and subtlety of fraud highlights the need for careful metric selection (focus on recall and precision, not accuracy).

### 3. Model Selection

**Models Explored:**
- Logistic Regression
- Random Forests
- XGBoost

**Final Choice:**  
**Random Forest with `class_weight='balanced'`**

**Why?**
- **Business Value:** By balancing precision and recall, the model supports business goals of minimizing financial loss (catching more fraud) and reducing customer friction (fewer false positives).
- **Class Weighting:** By using `class_weight='balanced'`, the model focuses on correctly identifying the minority class (fraud) without needing artificial data generation.
- **Robustness:** Random Forest is known for its strong generalization, resistance to overfitting

### 4. Evaluation & Tuning

- **Metrics Used:** Precision, Recall, F1-score, ROC-AUC.
- **Threshold Tuning:** Adjusted probability thresholds to prioritize fraud recall (catch more frauds, even with some false positives).
- **Business Alignment:** False negatives (missed fraud) are more costly than false positives (flagging a genuine transaction).

### 5. Business Outcome

- **Reduced Losses:** Early detection of fraud saves money and resources.
- **Better Customer Experience:** Fewer false alarms mean less inconvenience for genuine customers.
- **Actionable Insights:** Feature importance and model outputs can guide further investigation and process improvements.

## Conclusion

This project demonstrates a practical, business-aligned approach to credit card fraud detection:
- **Thoughtful preprocessing** and **careful handling of class imbalance** using model-driven weighting (`class_weight='balanced'`).
- **Interpretable, high-performing models** that align with business goals.

---

**For further details, see the code and notebooks in this repository.**
