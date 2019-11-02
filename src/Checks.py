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
