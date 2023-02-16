import numpy as np
import sympy as smp


def wrap(x, basis):
    return np.vstack([[1] * x.shape[0], np.array(basis)])


def evaluate(x_basis, b_basis, x):
    return (x_basis @ b_basis) / x.shape[0]


def reconstruct(x_basis, weights):
    return 0


# BASIS ----------------------------------------------------------------------------------------------------------------
def relu(x, centers):
    basis = [(x + c) * (x + c > 0) for c in centers]
    return wrap(x, basis)


def leaky_relu(x, centers, h):
    basis = [((x + c) * (x + c > 0) + h * (x + c) * (x + c <= 0)) for c in centers]
    return wrap(x, basis)


def softplus(x, centers):
    basis = [np.log(1 + np.exp(x + c)) for c in centers]
    return wrap(x, basis)


def softminus(x, centers):
    return x - softplus(x, centers)


def tanh(x, centers):
    basis = [(np.exp(x + c) - np.exp(-x - c)) / (np.exp(x + c) + np.exp(-x - c)) for c in centers]
    return wrap(x, basis)


def swish(x, centers):
    basis = [(x + c) / (1 + np.exp(-x - c)) for c in centers]
    return wrap(x, basis)


def sigmoid(x, centers):
    basis = [1 / (1 + np.exp(-(x + c))) for c in centers]
    return wrap(x, basis)


def gaussian_k(x, centers, h):
    basis = [np.exp(-(h * (x + c)) ** 2) for c in centers]
    return wrap(x, basis)


def step(x, low, high):
    return np.append(1,
                     np.array([smp.Piecewise((1, ((x > l) & (x < h))), (0, True)) for l in low for h in high])).reshape(
        -1, 1)
    # return wrap(x, basis)
