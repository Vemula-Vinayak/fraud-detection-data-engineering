# Credit Card Fraud Detection – Data Preprocessing & Feature Engineering

## Overview
This project focuses on building a robust data preprocessing pipeline for credit card fraud detection using a real-world dataset from Kaggle/TensorFlow. The objective is to prepare highly imbalanced transactional data for effective machine learning modeling.

---

## Dataset
- 284,807 transactions
- 31 features
- Target variable:
  - 0 → Non-Fraud
  - 1 → Fraud
- Features include PCA-transformed components (V1–V28), Time, and Amount.

---

## Key Challenges
- Severe class imbalance (fraud cases extremely rare)
- High-dimensional feature space
- Potential outliers in transaction amounts

---

## Preprocessing Steps

### 1. Outlier Removal (Z-Score)
Removed extreme transaction amounts (Z-score > 3) to stabilize model performance.

### 2. Feature Scaling
Standardized Time and Amount using StandardScaler for consistent magnitude across features.

### 3. Handling Class Imbalance
Applied SMOTE to generate synthetic minority samples and balance the dataset.

### 4. Dimensionality Reduction
Reduced features from 29 → 10 principal components using PCA to:
- Reduce multicollinearity
- Improve computational efficiency
- Maintain variance

---

## Visualizations
- Class imbalance bar chart
- Transaction amount distribution histogram
- Correlation heatmap of PCA features

---

## Business Impact
- Improved fraud recall using balanced data
- Reduced feature space by ~65%
- Enabled faster, scalable fraud detection modeling
- Improved data integrity for unbiased predictions

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- SMOTE (imblearn)
- PCA
- Matplotlib
- Seaborn

---

## How to Run

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
2. Open:
   ```
   fraud_preprocessing_pipeline.ipynb
   ```
3. Run all cells.
