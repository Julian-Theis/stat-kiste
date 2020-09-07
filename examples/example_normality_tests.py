from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
from backend.stat.normality_tests import run_shapiro_wilk_normality_test, run_anderson_darling, run_dagostino_pearson_test

if __name__ == "__main__":
    seed(1)
    gaussian = 5 * randn(100) + 50
    print('mean=%.3f stdv=%.3f' % (mean(gaussian), std(gaussian)))

    print()
    print("Shapiro-Wilk Normality Test")
    stat, p = run_shapiro_wilk_normality_test(gaussian, alpha = 0.05, print_results=True)

    print()
    print("D'Agostino and Pearson Test")
    stat, p = run_dagostino_pearson_test(gaussian, alpha=0.05, print_results=True)

    print()
    print("Anderson Darling Test")
    results = run_anderson_darling(gaussian, print_results=True)



