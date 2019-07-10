import numpy as np


def softmax(y):
    max = np.max(y)
    z = np.exp(y-max)/np.sum(np.exp(y-max))
    return z


def lose(z, t):
    e = np.abs(t*np.log(z))
    return e


z = ([0.4, 0.6])
t = ([0, 1])
E = lose(z, t)
print(E)