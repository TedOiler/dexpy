import numpy as np


def indicator(t, l, u):
    """
    Computes an indicator function with lower bound l and upper bound u. The output is 1 if the input t is in the range
    [l, u) or if t equals both l and u, and 0 otherwise.

    Args:
        t (float): Input value to be evaluated.
        l (float): Lower bound for the indicator function.
        u (float): Upper bound for the indicator function.

    Returns:
        int: 1 if t is in the range [l, u) or if t equals both l and u, 0 otherwise.
    """
    return 1 if ((t == u == 1) | ((t >= l) & (t < u))) else 0


def step(t, w):
    """
    Computes a step function with internal knots given by the weights w. The step function is piecewise constant and is
    defined by the values of the indicator function with lower and upper bounds determined by the knots.

    Args:
        t (float): Input value to be evaluated.
        w (list of float): Weights for the internal knots of the step function.

    Returns:
        float: The value of the step function at the input value t.
    """
    internal_knots = len(w) - 1
    increment = 1 / (internal_knots + 1)  # w=[0, 1, -1] -> increment = 1/(2+1)=0.33 -> [0, 0.33, 0.66, 1]
    lower = [increment * i for i in range(internal_knots + 1)]  # [0, 0.33, 0.66]
    upper = [increment * (i + 1) for i in range(internal_knots + 1)]  # [0.33, 0.66, 1]
    p = 0
    for i in range(len(w)):
        p += indicator(t, l=lower[i], u=upper[i]) * w[i]
    return p


def polynomial(t, p=0):
    """
    Calculates the value of the polynomial t^p at the given value of t.

    Args:
        t (float): The input value for the polynomial.
        p (int): The power to which t is raised. Default is 0.

    Returns:
        float: The value of the polynomial t^p evaluated at t.
    """
    return t ** p


def fourier(t, p):
    if p == 0:
        return np.sqrt(1 / 2)
    elif p % 2 == 0:
        return np.cos(p * np.pi * t)
    else:
        return np.sin((p - 1) * np.pi * t)
