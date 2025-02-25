# Coronary Heart Disease Prediction using Machine Learning

## Overview
This project explores machine learning algorithms to predict coronary heart disease (CHD) using a dataset from a high-risk region in the Western Cape, South Africa.

## Dataset
- **462 individuals** with **9 features**: 
  - `sbp` (systolic blood pressure)
  - `tobacco` (tobacco usage)
  - `ldl` (low-density lipoprotein cholesterol)
  - `adiposity`
  - `famhist` (family history: Present/Absent)
  - `typea` (Type A behavior)
  - `obesity`
  - `alcohol` (alcohol consumption)
  - `age`
- **Target variable:** `chd` (1: Presence, 0: Absence)

## Methodology
### 1. Exploratory Data Analysis (EDA)
- All features are numerical except 'famhist'.
- 'famhist' column converted to numerical (Present → 1, Absent → 0).
- Data free from unreasonable scale and no negative values.
- Positively skewed features: 'sbp,' 'tobacco,' 'ldl,' 'alcohol'.
- No significant outliers.
- No strong correlations (|r| < 0.8) among features.
- VIF scores < 5 (no multicollinearity).
- Imbalance in Target Variable 'chd': 65% class ‘0’, 35% class ‘1’.

### 2. Modeling and Evaluation
- Data Standardization
- Data Split: 80% for training and 20% for testing
- Base Model: Logistic Regression (with Ridge Penalty)
- Evaluated using **accuracy, precision, recall, and F1-score**

- Compared with 9 other machine learning models
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Adaboost
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - Linear Discriminant Analysis (LDA)
  - Quadratic Discriminant Analysis (QDA)
  - Naïve Bayes
 
## Results
### Model Performance Comparison

| Index | ML Classification Model              | Accuracy Score | ROC AUC Score | F1 Score (Class 0) | F1 Score (Class 1) |
|-------|--------------------------------------|---------------|--------------|------------------|------------------|
| 1     | **Logistic Regression (Ridge)**     | ✅ **78.49%**  | ✅ **80.11%** | ✅ **0.84**       | ✅ **0.66**       |
| 2     | K Nearest Neighbors                 | 72.04%        | 73.65%       | 0.81             | 0.43             |
| 3     | Decision Tree                        | 🔴 **63.44%**  | 🔴 **50.00%** | 0.78             | 0.00             |
| 4     | Random Forest                        | 70.97%        | 65.90%       | 0.79             | 0.54             |
| 5     | Adaboost                             | 70.97%        | 67.15%       | 0.78             | 0.57             |
| 6     | Gradient Boosting                    | 72.04%        | 64.88%       | 0.81             | 0.50             |
| 7     | Support Vector Machine               | 77.42%        | 72.23%       | 0.84             | 0.63             |
| 8     | LDA                                  | 76.34%        | 71.39%       | 0.83             | 0.62             |
| 9     | QDA                                  | 76.34%        | 72.01%       | 0.83             | 0.63             |
| 10    | Naïve Bayes                          | 75.27%        | 75.52%       | 0.79             | 0.69             |

**Notes:**
- ✅ **Logistic Regression with Ridge Penalty** achieved the **best performance** across all metrics.
- 🔴 **Decision Tree** had the **lowest accuracy (63.44%)** and **ROC AUC score (50.00%)**, indicating poor classification ability.
- **Naïve Bayes** had the highest **F1 Score for Class 1 (69%)**, making it more effective in identifying positive cases.

## Key Takeaways
- **Feature importance analysis** showed that **age** was the most influential predictor.
- **Regularization (Ridge)** reduced model complexity and improved generalization.
- **Logistic Regression outperformed complex models**, proving that simpler models can be effective for small datasets.

## Limitations & Future Work
- **Class imbalance** (65% `0`, 35% `1`) could be addressed with resampling techniques.
- **Dataset size** (462 observations) is small; collecting more data could improve model reliability.
- **Feature engineering** (e.g., polynomial features, interactions) might enhance performance.


