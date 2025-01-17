class ChecksSVM:

    def __init__(self):
        return

    def check_if_valid_c(self, phrase):
        while True:
            try:
                number = float(input(phrase))
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

    def check_if_valid_kernel(self, phrase):
        while True:
            try:
                kernel = input(phrase)
                if kernel not in ["linear", "poly", "rbf", "sigmoid"]:
                    continue
                return kernel
            except ValueError:
                continue

    def check_if_valid_degree(self, phrase):
        while True:
            try:
                number = int(input(phrase))
                if number <= 0:
                    continue
                return number
            except ValueError:
                continue

    def check_if_valid_epsilon(self, phrase):
        while True:
            try:
                step = float(input(phrase))
                return step
            except ValueError:
                continue
