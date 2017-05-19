# -*- coding:utf-8 -*-
import numpy as np


def interpolation(body, head=None, tail=None):
    """
    interpolation the body data into a set of cubic function
    the cubic function is like : f(x) = a + b * (x-xi) + c * (x-xi)^2 + d * (x-xi)^3
    :param body: the node data. as np.asarray([[y0,x0],[y1,x1].....[yn, xn]])
    :param head: the head k. if it is None, then the head is a free end
    :param tail: the tail k. if it is None, then the tail is a free end
    :return: a numpy.ndarray in the form of [[a0, b0, c0, d0],
                                             [a1, b1, c1, d1],
                                             ................,
                                             [an, bn, cn, dn]]
    """
    xs = body[:, 0]
    n = len(body)
    ys = body[:, 1]

    hs = np.full(n-1, 0., np.float64)
    for i in range(n - 1):
        hs[i] = xs[i + 1] - xs[i]

    a = np.full((n, n), 0., np.float64)
    for i in range(1, n - 1):
        a[i][i - 1] = hs[i - 1]
        a[i][i] = (hs[i] + hs[i - 1]) * 2.
        a[i][i + 1] = hs[i]

    b = np.full(n, 0, np.float64)
    for i in range(1, n - 1):
        b[i] = 6 * ((ys[i + 1] - ys[i]) / hs[i] - (ys[i] - ys[i - 1]) / hs[i - 1])

    if head is None:
        b[0] = 0
        a[0][0] = 1
    else:
        a[0][0] = 2 * hs[0]
        a[0][1] = hs[0]
        b[0] = 6*((ys[1] - ys[0]) / hs[0] - head)

    if tail is None:
        b[-1] = 0
        a[-1][-1] = 1
    else:
        a[-1][-2] = hs[-1]
        a[-1][-1] = 2*hs[-1]
        b[-1] = 6 * (tail - (ys[-1] - ys[-2]) / hs[-1])

    print(a)
    print(b.T)

    ms = np.linalg.solve(a, b)  # solve the linear equations
    ai = ys[:-1]
    bi = get_bi(ms, ys, hs)
    ci = get_ci(ms)
    di = get_di(ms, hs)
    return np.vstack([ai, bi, ci, di]).T


def get_bi(ms, ys, hs):
    n = len(ms)
    ret = np.full(n - 1, 0., np.float64)
    for i in range(len(ret)):
        ret[i] = (ys[i+1] - ys[i]) / hs[i] - hs[i]*ms[i]/2. - hs[i]*(ms[i+1] - ms[i]) / 6.
    return ret


def get_ci(ms):
    n = len(ms)
    ret = np.full(n - 1, 0., np.float64)
    for i in range(len(ret)):
        ret[i] = ms[i] / 2.
    return ret


def get_di(ms, hs):
    n = len(ms)
    ret = np.full(n - 1, 0., np.float64)
    for i in range(len(ret)):
        ret[i] = (ms[i + 1] - ms[i]) / (6 * hs[i])
    return ret
