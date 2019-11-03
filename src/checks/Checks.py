class Checks:

    def __init__(self):
        return

    def check_if_valid_c(self, phrase):
        while True:
            try:
                number = int(input(phrase))
                if number <= 0:
                    continue
                return number
            except ValueError:
                continue

    def check_if_valid_gamma(self, phrase):
        while True:
            try:
                step = float(input(phrase))
                return step
            except ValueError:
                continue

    def check_if_valid_k(self, phrase):
        while True:
            try:
                number = int(input(phrase))
                if number <= 0:
                    continue
                return number
            except ValueError:
                continue

    def check_if_valid_kernel(self, phrase):
        while True:
            try:
                kernel = input(phrase)
                if kernel not in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:
                    continue
                return kernel
            except ValueError:
                continue

    def check_if_valid_max_iter(self, phrase):
        while True:
            try:
                number = int(input(phrase))
                if number <= 0:
                    continue
                return number
            except ValueError:
                continue

    def check_if_valid_tol(self, phrase):
        while True:
            try:
                step = float(input(phrase))
                return step
            except ValueError:
                continue
