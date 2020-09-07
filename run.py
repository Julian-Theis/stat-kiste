import argparse
from cmd.interface import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', choices=["ci_auc", "classification_report", "normality", "mean"], help='Available functions: [ci_auc]', required=True)

    """ AUC CI SPECIFIC """
    parser.add_argument('-file', '--file', help='Input file', default="", required=False)
    parser.add_argument('-ypc', '--y_pred_col', help='y_pred column name', default="y_pred", required=False)
    parser.add_argument('-ytc', '--y_true_col', help='y_true column name', default="y_true", required=False)
    parser.add_argument('-ysc', '--y_score_col', help='y_score column name', default="y_score", required=False)
    parser.add_argument('-a', '--alpha', help='Alpha', default=0.95, required=False)

    """ NORMALITY TEST SPECIFIC"""
    parser.add_argument('-vc', '--value_col', help='values column name', default="values", required=False)
    parser.add_argument('-test', '--test', help='Test name <shapiro, dagostino, anderson, wilcoxon, pairedt>', required=False)

    """ MEAN TEST SPECIFIC"""
    parser.add_argument('-s1c', '--sample1_col', help='sample 1 column name', default="sample1", required=False)
    parser.add_argument('-s2c', '--sample2_col', help='sample 2 column name', default="sample2", required=False)

    """ Runtime """
    args = parser.parse_args()
    if args.function == "ci_auc":
        if args.file == "":
            print("-f / --file argument is missing.")
        else:
            print()
            print("Calculating AUC with Confidence Intervals...")
            ci_auc(file=args.file, y_score_col=args.y_score_col, y_true_col=args.y_true_col, y_pred_col=args.y_pred_col, alpha=float(args.alpha))

    elif args.function == "classification_report":
        if args.file == "":
            print("-f / --file argument is missing.")
        else:
            print()
            print("Running Classification Report ...")
            classification_report(file=args.file, y_pred_col=args.y_pred_col, y_true_col=args.y_true_col, y_score_col=args.y_score_col, alpha=float(args.alpha))

    elif args.function == "normality":
        if args.file == "":
            print("-f / --file argument is missing.")
        elif args.test not in ["shapiro", "anderson", "dagostino"]:
            print("-test / --test argument is missing or incorrect. Select one of <shapiro, anderson, dagostino>")
        else:
            print()
            print("Running Normality Test (" + str(args.test) + ")")
            normality_test(file=args.file, value_col=args.value_col, alpha=float(args.alpha), test=args.test)

    elif args.function == "mean":
        if args.file == "":
            print("-f / --file argument is missing.")
        elif args.test not in ["wilcoxon", "pairedt"]:
            print("-test / --test argument is missing or incorrect. Select one of <wilcoxon, pairedt>")
        else:
            print()
            print("Running Mean Test (" + str(args.test) + ")")
            mean_test(file=args.file, sample1_col=args.sample1_col, sample2_col=args.sample2_col, alpha=float(args.alpha), test=args.test)