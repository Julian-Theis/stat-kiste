from numpy.random import seed
from numpy.random import randn
from backend.stat.mean_tests import run_paired_t_test, run_wilcoxon_signed_rank_test

if __name__ == "__main__":
    seed(1)
    sample1 = 5 * randn(100) + 50
    sample2 = 5 * randn(100) + 49

    print()
    print("Paired T-Test")
    stat, p = run_paired_t_test(sample1, sample2, alpha=0.05, print_results=True)

    print()
    print("Wilcoxon Signed Rank Test")
    stat, p = run_wilcoxon_signed_rank_test(sample1, sample2, alpha=0.05, print_results=True)




