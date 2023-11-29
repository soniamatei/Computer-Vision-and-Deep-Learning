import numpy as np


def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    # TODO your code here

    maximum = np.max(x)
    normalization_factor = np.sum(np.exp((x - maximum) / t))
    probability = np.vectorize(lambda e: np.exp((e - maximum) / t) / normalization_factor)
    result = probability(x)

    # end TODO your code here

    return result
