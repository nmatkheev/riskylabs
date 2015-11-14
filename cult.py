__author__ = 'lancer'

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

a = 4
b = 33

fig, ax = plt.subplots(1, 1)
n = 400
step = 1


p = float(1) / float(1 + b)
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
print binom.var(n, p)
print binom.expect(lambda x: x, args=(n, p))
print binom.expect(lambda x: x ** 2, args=(n, p))

# x = np.arange(binom.ppf(0.00001, n, p), binom.ppf(0.99999, n, p))
# x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
x = np.arange(binom.ppf(0.001, n, p), binom.ppf(0.999, n, p), step)
y = np.array(binom.pmf(x, n, p), dtype=float)


def squarer(pos1=1, pos2=len(x)):
    square = 0
    if pos2 > len(x): pos2 -= len(x)
    for i in range(pos1, pos2):
        square += (float(y[i - 1] + y[i]) / float(2)) * (x[i] - x[i - 1])
    return square


print("Square: ", squarer(2, 3))
print("Full square: ", squarer())

ax.plot(x, binom.pmf(x, n, p), 'bo', ms=7, label='binom pmf')
ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)

plt.show()
