import pandas as pd
import ast
import numpy as np
from backend.stat.ci_auc import calculate_auc_ci
from backend.stat.classification_report import run_classification_report
from backend.stat.normality_tests import run_shapiro_wilk_normality_test, run_dagostino_pearson_test, run_anderson_darling
from backend.stat.mean_tests import run_wilcoxon_signed_rank_test, run_paired_t_test


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

def normality_test(file, value_col="values", alpha=0.05, test="shapiro"):
    try:
        df = pd.read_csv(file)
        values = df[value_col].to_numpy()

        if test == "shapiro":
            run_shapiro_wilk_normality_test(values, alpha)
        elif test == "anderson":
            run_anderson_darling(values)
        elif test == "dagostino":
            run_dagostino_pearson_test(values, alpha)
        else:
            print("Test not implemented...")

    except Exception as e:
        print("Failed to run Normality Test:", str(e))

def mean_test(file, sample1_col="sample1", sample2_col="sample2", alpha=0.05, test="wilcoxon"):
    try:
        df = pd.read_csv(file)
        sample1 = df[sample1_col].to_numpy()
        sample2 = df[sample2_col].to_numpy()

        if test == "wilcoxon":
            run_wilcoxon_signed_rank_test(sample1, sample2, alpha=alpha)
        elif test == "pairedt":
            run_paired_t_test(sample1, sample2, alpha=alpha)
        else:
            print("Test not implemented...")

    except Exception as e:
        print("Failed to run Mean Test:", str(e))
