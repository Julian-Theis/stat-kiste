import pandas as pd
from backend.stat.ci_auc import calculate_auc_ci

def ci_auc(file, y_true_col="y_true", y_pred_col="y_pred", alpha=0.95):
    df = pd.read_csv(file)
    y_true = df[y_true_col].to_numpy()
    y_pred = df[y_pred_col].to_numpy()

    auc, auc_cov, ci = calculate_auc_ci(y_true, y_pred, alpha)

    print("AUC:", auc)
    print("AUC COV:", auc_cov)
    print((alpha * 100), "% AUC Confidence Interval:", ci)
