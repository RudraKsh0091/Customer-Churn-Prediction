# Bank Customer Churn Prediction

Predicting which bank customers are likely to churn using an **Artificial Neural Network (ANN)** and **Random Forest**, with a focus on handling class imbalance and proper evaluation metrics.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| ANN | ~86% | **0.864** |
| Random Forest | ~86% | 0.861 |

ANN marginally outperforms Random Forest on ROC-AUC.

---

## Dataset

[Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset) — 10,000 rows, 14 features including Age, Balance, Geography, NumOfProducts, etc.

---

## Key Techniques

- **Class imbalance** (~20% churn) handled via `class_weight='balanced'`
- **Stratified train/test split** to preserve churn ratio
- **ROC-AUC + F1-score** used as primary metrics (accuracy alone is misleading on imbalanced data)
- **Dropout + EarlyStopping** on ANN to prevent overfitting
- **Feature importance** analysis via Random Forest

---

## Project Structure

```
customer-churn-prediction/
├── customer_churn_prediction.ipynb
├── models/
│   ├── ann_churn_model.keras
│   ├── rf_churn_model.pkl
│   └── scaler.pkl
├── plots/
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── class_distribution.png
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/yourusername/customer-churn-prediction
cd customer-churn-prediction
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn joblib
# Open and run customer_churn_prediction.ipynb
```

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn
