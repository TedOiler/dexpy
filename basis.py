import numpy as np
import sympy as smp


def evaluate(vec1, vec2, symbol=smp.symbols('t'), lower=0, upper=1):
    return float(smp.integrate(vec1 @ vec2.T, (symbol, lower, upper)).evalf())


# BASIS ----------------------------------------------------------------------------------------------------------------
def swish(x, centers):
    return np.append(1, np.array([(x + c) / (1 + smp.exp(-x - c)) for c in centers])).reshape(-1, 1)


def relu(x, centers):
    return np.append(1, np.array([smp.Max(0, x + c) for c in centers])).reshape(-1, 1)


def leaky_relu(x, centers, h):
    return np.append(1, np.array([smp.Max(h * (x + c), x + c) for c in centers])).reshape(-1, 1)


def softplus(x, centers):
    return np.append(1, np.array([smp.log(1 + smp.exp(x + c)) for c in centers])).reshape(-1, 1)


def softminus(x, centers):
    return x - softplus(x, centers).reshape(-1, 1)


def tanh(x, centers):
    return np.append(1, np.array(
        [(smp.exp(x + c) - smp.exp(-x - c)) / (smp.exp(x + c) + smp.exp(-x - c)) for c in centers])).reshape(-1, 1)


def sigmoid(x, centers):
    return np.append(1, np.array([1 / (1 + smp.exp(-(x + c))) for c in centers])).reshape(-1, 1)


def gaussian_k(x, centers, h):
    return np.append(1, np.array([smp.exp(-(h * (x + c)) ** 2) for c in centers])).reshape(-1, 1)


def step(x, low, high):
    return np.append(1, np.array([smp.Piecewise((1, ((x > l) & (x < h))), (0, True)) for l in low for h in high])).reshape(-1, 1)
