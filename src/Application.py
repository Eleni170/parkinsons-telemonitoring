from pandas import read_csv
from pip._vendor.distlib.compat import raw_input
import numpy as np


class Application:

    def __init__(self):
        return

    def check_if_int(self, phrase):
        while True:
            try:

                number = int(raw_input(phrase))
                if number <= 0:
                    continue
                return number
            except ValueError:
                continue

    def check_if_correct_gamma(self, phrase):
        while True:
            try:
                step = float(raw_input(phrase))
                return step
            except ValueError:
                continue


if __name__ == '__main__':
    application = Application()
    data = read_csv('../dataset/parkinsons_updrs.data', header=0).values
    NumberOfAttributes = len(data[0])
    NumberOfPatterns = len(data)
    x = np.zeros((NumberOfPatterns, 16))
    t = np.zeros(NumberOfPatterns)
    x = data[:, 6:NumberOfAttributes]
    t = data[:, 3:5]

    print('hello')
