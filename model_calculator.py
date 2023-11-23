from math import log, exp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class ModelCalculator:
    def __init__(self, N: int, t: float, g: float, enable_caching=True):
        """
        :param N: Total amount of potential models.
        :param t: Desired percentage bracket (as a fraction; e.g., 0.1 for top 10%).
        :param g: Guaranteed probability for finding a model in the top t%.
        :param enable_caching: Boolean flag to enable or disable caching for factorial computations.
        """

        self.N = N
        self.t = t
        self.g = g
        self.enable_caching = enable_caching

        if self.enable_caching:
            self.log_factorial_cache = dict()
        self.T = int(self.t * self.N)
        self.p = log(1 - self.g) + self.S(self.N - self.T + 1, self.N + 1)

    def S(self, a: int, b: int) -> float:
        """
        Computes the sum of logarithms of numbers in a given range.

        :param a: Start of the range (inclusive).
        :param b: End of the range (exclusive).
        :return: Sum of logarithms from a to b-1.
        """

        if not self.enable_caching:
            return sum(log(n) for n in range(a, b))

        total = 0
        for n in range(max(1, a), b):
            log_fact = self.log_factorial_cache.get(n)
            if log_fact is not None:
                total += log_fact
            else:
                log_fact = log(n)
                self.log_factorial_cache[n] = log_fact
                total += log_fact
        return total

    def f(self, m: int) -> float:
        """
        Computes a specific part of an equation as seen in the notebook.

        :param m: The amount of models to consider.
        :return: The computed value of the function f.
        """

        return self.S(self.N - self.T - m + 1, self.N - m + 1)

    def binary_search(self, low: int, high: int) -> int:
        """
        Performs a binary search to find the minimum m that satisfies the condition p >= f(m).

        :param low: Lower bound of the search interval.
        :param high: Upper bound of the search interval.
        :return: The minimum value of m that satisfies the condition.
        """

        while low < high:
            mid = (low + high) // 2
            if self.p >= self.f(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def calculate_result(self) -> int:
        """
        Calculates the minimum amount of models to be checked to guarantee finding a model in the top t%.

        :return: The minimum amount of models to check.
        """

        return self.binary_search(1, self.N)

    def plot_results(self, include_line_of_best_fit: bool = False) -> None:
        """
        Plots the probability of finding a top t% model against the amount models checked.

        :param include_line_of_best_fit: Whether to include curve fitting in the plot.
        """

        tmp_precompute = self.S(self.N - self.T + 1, self.N + 1)

        def probability_top_t_model(n: int) -> float:
            return 1 - exp(self.f(n) - tmp_precompute)

        def curve_func(x, k):
            return 1 - np.exp(-k * x)

        x_values = np.arange(1, self.N + 1)
        y_values = np.array([probability_top_t_model(m) for m in x_values])
        plt.plot(x_values, y_values)

        x_minimum = self.calculate_result()
        plt.plot([x_minimum, x_minimum], [0, self.g], color="lime", label="Minimum Models to Attain Guarantee", )

        if include_line_of_best_fit:
            popt, _ = curve_fit(curve_func, x_values, y_values)
            k_fit = popt[0]
            plt.plot(x_values, curve_func(x_values, k_fit), "r--", label=f"Fit: k={k_fit:.3f}", )

        plt.xlabel("Number of Models Checked")
        plt.ylabel(f"Likelihood of Finding Top {self.t * 100}% Model (%)")
        plt.title(f"Probability of Finding Top Model with {self.g * 100}% Guarantee")
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    N_ = 7056  # total possible models
    t_ = 0.001  # top percentage models we'd like
    g_ = 0.900  # probability guaranteed we'd get the top percentage model

    calc = ModelCalculator(N_, t_, g_)
    res = calc.calculate_result()

    print(f"If we have {N_} possible models,\n"
          f"and we want a top {t_ * 100}% performing model,\n"
          f"with a certainty of {g_ * 100}%,\n"
          f"then we will need to search {res} models.")

    # calc.plot_results(include_line_of_best_fit=False)
