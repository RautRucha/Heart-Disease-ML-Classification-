# Heart Disease ML Classification

This project applies supervised machine learning techniques to predict the likelihood of coronary heart disease (CHD) using a dataset from a high-risk region in South Africa.

## 📊 Dataset Overview

- Consists of **462 male individuals** from the Western Cape, South Africa.
- Target variable: `chd` (1 = presence, 0 = absence of coronary heart disease).
- Features include:
  - Continuous: `sbp`, `tobacco`, `ldl`, `adiposity`, `typea`, `obesity`, `alcohol`, `age`
  - Categorical: `famhist` (converted to binary)
- The dataset was clean (no missing or duplicate entries) and numerically formatted.

🔍 Data Exploration Summary
Skewness & Outliers:

sbp, tobacco, ldl, and alcohol showed positive skewness.

A few outliers were found but retained due to the small dataset size.

Correlation Analysis:

No strong correlation found among the 9 features (all |r| < 0.8).

Multicollinearity Check (VIF):

All features had VIF < 5, indicating no serious multicollinearity.

Adiposity was most correlated with obesity (r = 0.72) and had the highest VIF (3.59).

Target Variable Imbalance:

Class distribution: 65% = class 0 (no CHD), 35% = class 1 (CHD).

No extreme imbalance; no class-balancing techniques required.


## 🔍 Process

1. **Exploratory Data Analysis (EDA):**
   - Histograms, boxplots, and pair plots were used.
   - Identified positive skewness and minor outliers (retained due to small dataset size).
   - Checked for correlation and multicollinearity (VIF < 5 for all features).

2. **Feature Engineering:**
   - Converted `famhist` to binary.
   - Standardized data due to varying scales across features.

3. **Train/Test Split:**
   - Used an 80:20 split for model evaluation.

4. **Model Selection:**
   - Compared 10 classification algorithms using accuracy, precision, recall, F1-score, and ROC AUC.

5. **Hyperparameter Tuning:**
   - Applied grid search for logistic regression to optimize the regularization parameter `C`.

## 🤖 Machine Learning Models Used

- Logistic Regression with Ridge Penalty (L2 regularization)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- Naïve Bayes
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

## 🧠 Key Findings

1. **Logistic Regression (Ridge)** performed best overall with:
   - Accuracy: ~78.5%
   - ROC AUC: 80%
   - Precision (Class 1): 79%
   - Recall (Class 1): 56%
2. SVM was the next-best model, close in accuracy but lower recall.
3. Models generally classified class 0 better than class 1.
4. Adaboost and Gradient Boosting underperformed in F1-score.
5. KNN and Decision Tree were least effective due to low AUC and poor recall.

## ✅ Conclusion

Logistic Regression with Ridge regularization offered the best balance between precision and recall for identifying heart disease cases. While the dataset size limited model complexity, it demonstrated the effectiveness of classical machine learning methods for binary classification tasks in healthcare data.

---
