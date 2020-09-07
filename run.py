import argparse
from cmd.interface import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', choices=["ci_auc", "classification_report"], help='Available functions: [ci_auc]', required=True)

    """ AUC CI SPECIFIC """
    parser.add_argument('-file', '--file', help='Input file', default="", required=False)
    parser.add_argument('-ypc', '--y_pred_col', help='y_pred column name', default="y_pred", required=False)
    parser.add_argument('-ytc', '--y_true_col', help='y_true column name', default="y_true", required=False)
    parser.add_argument('-ysc', '--y_score_col', help='y_score column name', default="y_score", required=False)
    parser.add_argument('-a', '--alpha', help='Alpha', default=0.95, required=False)

    """ Runtime """
    args = parser.parse_args()
    if args.function == "ci_auc":
        if args.file == "":
            print("-f / --file argument is missing.")
        else:
            print()
            print("Calculating AUC with Confidence Intervals...")
            ci_auc(file=args.file, y_score_col=args.y_score_col, y_true_col=args.y_true_col, alpha=float(args.alpha))

    elif args.function == "classification_report":
        if args.file == "":
            print("-f / --file argument is missing.")
        else:
            print()
            print("Running Classification Report ...")
            classification_report(file=args.file, y_pred_col=args.y_pred_col, y_true_col=args.y_true_col, y_score_col=args.y_score_col, alpha=float(args.alpha))