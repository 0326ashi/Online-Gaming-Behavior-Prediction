import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from preprocessing import preprocess_data


def run_svm_pipeline(data_path: str):
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    baseline_svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
    )
    baseline_svm.fit(X_train, y_train)
    y_pred_baseline = baseline_svm.predict(X_test)

    print("Baseline Accuracy:", round(accuracy_score(y_test, y_pred_baseline), 4))
    print(
        "Baseline Weighted F1:",
        round(f1_score(y_test, y_pred_baseline, average="weighted"), 4),
    )

    param_grid = {
        "C": [0.1, 1.0, 5.0, 10.0],
        "class_weight": [None, "balanced"],
        "max_iter": [5000, 10000],
    }

    grid = GridSearchCV(
        estimator=LinearSVC(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_

    print("Best Parameters:", grid.best_params_)
    print("Best CV Weighted F1:", round(grid.best_score_, 4))

    y_pred = best_svm.predict(X_test)
    decision_scores = best_svm.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("========== FINAL SVM EVALUATION ==========")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("svm_confusion_matrix.png", dpi=150)
    plt.close()

    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_auc = roc_auc_score(y_test_bin, decision_scores, average="macro", multi_class="ovr")
    print(f"ROC AUC (OvR, macro): {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], decision_scores[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_label} (AUC = {class_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("SVM ROC Curve (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("svm_roc_curve.png", dpi=150)
    plt.close()

    coef_abs_mean = np.mean(np.abs(best_svm.coef_), axis=0)
    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": coef_abs_mean}
    ).sort_values("Importance", ascending=False)

    print("\nTop 10 influential features:")
    print(importance_df.head(10))

    importance_df.to_csv("svm_feature_importance.csv", index=False)


if __name__ == "__main__":
    run_svm_pipeline("data/online_gaming_behavior_dataset.csv")
