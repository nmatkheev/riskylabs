__author__ = 'lancer'

import numpy as np
from numpy import power, exp, array, var
from scipy.optimize import curve_fit
from scipy.misc import factorial, comb
import matplotlib.pyplot as plt

lambdas = 0.08
a = 4
b = 33
pois = []
args = []


def poisson(k, lamb):
    return (lamb ** k / factorial(k)) * exp(-lamb)


def binomial(k):
    return comb(a + k - 1, k) * power(float(1) / float(1 + b), a) * power(float(b) / float(1 + b), k)


for x in range(0, 400, 12):
    args.append(x)
    pois.append(poisson(x, lambdas))

i = 0
mat = 0
mat2 = 0
print(sum(pois))
for x in pois:
    mat += x * i
    mat2 += power(x * i, 2)
    i += 12


var = mat2 - power(mat, 2)
print "mat: ", mat, " mat2: ", mat2, " var: ", var


argmin = min(args)
argmax = max(args)
poismin = max(pois)
poismax = max(pois)

plt.grid(True)
plt.subplot(2, 1, 1)
plt.plot(args, pois, 'yo-')
plt.title('Binomial distribution')
plt.ylabel('p')
plt.xlabel('n')

plt.show()