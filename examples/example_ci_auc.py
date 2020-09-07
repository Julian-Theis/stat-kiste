import numpy as np
from backend.stat.ci_auc import calculate_auc_ci

if __name__ == "__main__":
    alpha = 0.95
    y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04, 0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
    y_true = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0,   0,    1,    0,    0,    1,    1,    0,    1,    0])

    auc, auc_cov, ci = calculate_auc_ci(y_true, y_pred, alpha)

    print("*** Statistics ***")
    print("AUC:", auc)
    print("AUC COV:", auc_cov)
    print((alpha*100),"% AUC Confidence Interval:", ci)