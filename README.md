# 🎮 Online Gaming Behavior Prediction

## 📌 Overview

This project focuses on predicting online gaming behavior using a real-world dataset. By applying multiple supervised machine learning algorithms, the project aims to analyze player activity patterns and accurately classify or predict gaming behavior.

The study also compares the performance of different models to identify the most effective approach for this problem.

## 🎯 Objectives
- Analyze online gaming behavior data
- Perform data preprocessing and feature selection
- Train multiple machine learning models
- Compare model performance
- Identify the best-performing model for prediction

## 🤖 Machine Learning Models Used

The following supervised learning algorithms are implemented:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

## 📂 Project Structure

- `data/` - dataset files
- `src/preprocessing.py` - shared preprocessing pipeline
- `src/svm_model.py` - runnable SVM pipeline script (training, tuning, evaluation)
- `notebooks/svm.ipynb` - SVM notebook for assignment submission
- `notebooks/decision_tree.ipynb` - Decision Tree model notebook
- `notebooks/random_forest.ipynb` - Random Forest model notebook
- `members.txt` - group member details (IDs and emails)
- `submission.txt` - required submission links

## 🧪 SVM (Your Contribution)

The SVM implementation uses `LinearSVC` with:

- shared preprocessing (`src/preprocessing.py`)
- class balancing support
- hyperparameter tuning (`GridSearchCV`)
- evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- visualizations: confusion matrix and ROC curves

### Verified SVM Result Snapshot

- Accuracy: `0.7169`
- Weighted Precision: `0.7399`
- Weighted Recall: `0.7169`
- Weighted F1-score: `0.7145`
- ROC-AUC (OvR Macro): `0.8783`

## ▶️ How To Run

Run SVM script:

```bash
python src/svm_model.py
```

Open and run notebook:

```bash
jupyter notebook notebooks/svm.ipynb
```
