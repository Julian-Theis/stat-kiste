import pandas as pd
import ast
import numpy as np
from backend.stat.ci_auc import calculate_auc_ci
from backend.stat.classification_report import run_classification_report


def ci_auc(file, y_true_col="y_true", y_score_col="y_score", alpha=0.95):
    try:
        df = pd.read_csv(file)
        y_true = df[y_true_col].to_numpy()
        df[y_score_col] = df[y_score_col].apply(ast.literal_eval)
        y_score = np.array(df[y_score_col].values.tolist())
        y_score= y_score[:, 1]

        auc, auc_cov, ci = calculate_auc_ci(y_true, y_score, alpha)

        print("AUC:", auc)
        print("AUC COV:", auc_cov)
        print((alpha * 100), "% AUC Confidence Interval:", ci)
    except Exception as e:
        print("Failed to calculate AUC CIs:", str(e))


def classification_report(file, y_true_col="y_true", y_pred_col="y_pred", y_score_col="y_score", alpha=0.95):
    try:
        df = pd.read_csv(file)
        df[y_score_col] = df[y_score_col].apply(ast.literal_eval)
        y_true = df[y_true_col].to_numpy()
        y_pred = df[y_pred_col].to_numpy()
        y_score = np.array(df[y_score_col].values.tolist())

        run_classification_report(y_true=y_true, y_pred=y_pred, y_score=y_score, alpha=alpha)

    except Exception as e:
        print("Failed to run Classification Report:", str(e))
