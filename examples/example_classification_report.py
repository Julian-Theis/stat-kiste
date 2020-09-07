import numpy as np
from backend.stat.classification_report import run_classification_report

if __name__ == "__main__":
    y_pred = np.array([1,0,0,1,1,0,0,1,1,2,0,0,2,2,2,2])
    y_true = np.array([1,1,0,1,1,0,1,0,2,2,2,2,2,1,0,2])
    y_score = np.array([
        [0.002, 0.642, 0.356],
        [0.642, 0.002, 0.356],
        [0.642, 0.002, 0.356],
        [0.002, 0.642, 0.356],
        [0.002, 0.642, 0.356],
        [0.642, 0.002, 0.356],
        [0.642, 0.002, 0.356],
        [0.002, 0.642, 0.356],
        [0.002, 0.642, 0.356],
        [0.002, 0.356, 0.642],
        [0.642, 0.002, 0.356],
        [0.642, 0.002, 0.356],
        [0.002, 0.356, 0.642],
        [0.002, 0.356, 0.642],
        [0.002, 0.356, 0.642],
        [0.002, 0.356, 0.642]])

    report_with_auc = run_classification_report(y_true=y_true, y_pred=y_pred, y_score=y_score, alpha=0.68)