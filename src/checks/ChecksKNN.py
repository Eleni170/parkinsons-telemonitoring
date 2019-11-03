class ChecksKNN:

    def __init__(self):
        return

    def check_if_valid_k(self, phrase):
        while True:
            try:
                number = int(input(phrase))
                if number <= 0:
                    continue
                return number
            except ValueError:
                continue
