# -*- coding:utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression


class Func(object):
    """the abstract math function class"""
    def __init__(self):
        self.func = None
        self.pfunc = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def integrate(self, a, b):
        """the integrate method, get the integrate in [a, b]"""
        raise NotImplementedError

    def vec(self):
        """get the vectorized version of the function"""
        if self.func is None:
            return None
        return np.vectorize(self.func)

    def vec_p(self):
        """get the vectorized version of primitive function"""
        if self.pfunc is None:
            return None
        return np.vectorize(self.pfunc)


class CubicFunc(Func):
    """the cubic function class"""
    def __init__(self, params):
        super(CubicFunc, self).__init__()
        (self.a, self.b, self.c, self.d) = params
        self.func = lambda x: self.a*x**3 + self.b*x**2 + self.c*x + self.d
        self.pfunc = lambda x: self.a*x**4 / 4. + self.b*x**3 / 3. + self.c*x*x/2. + self.d*x

    def integrate(self, a, b):
        return self.pfunc(b) - self.pfunc(a)

    def __call__(self, *args, **kwargs):
        return self.func(*args)

    def __str__(self):
        return 'f(x) = %f X^3 + %f X^2 + %f X + %f' % (self.a, self.b, self.c, self.d)


class LinearFunc(Func):
    """the linear function class"""
    def __init__(self, params):
        super(LinearFunc, self).__init__()
        if isinstance(params, np.ndarray):
            model = LinearRegression()
            model.fit(np.vstack(params[:, 1]), np.vstack(params[:, 0]))
            self.k = model.coef_[0][0]
            self.b = model.predict([[0]])[0][0]
        else:
            (self.k, self.b) = params
        self.func = lambda x: self.k*x + self.b
        self.pfunc = lambda x: self.k*x*x / 2. + self.b * x

    def integrate(self, a, b):
        return self.pfunc(b) - self.pfunc(a)

    def __call__(self, *args, **kwargs):
        return self.func(*args)

    def __str__(self):
        return 'f(x) = %f X + %f' % (self.k, self.b)


class FuncSet(object):
    """FuncSet represent a set of function"""
    def __init__(self, params=None):
        self.content = []
        self.lowerbound = None
        self.upperbound = None
        if params is not None:
            self.content = params

    def __getitem__(self, item):
        return self.content[item]

    def __iter__(self):
        return self.content.__iter__()

    def __str__(self):
        ret = ''
        for i in self.content:
            (x, y), f = i
            ret += '[%f, %f] %s \n' % (x, y, f.__str__())
        return ret

    def append(self, func, bound):
        """append new function to the func_set"""
        (a, b) = bound
        if self.lowerbound is None or a < self.lowerbound:
            self.lowerbound = a

        if self.upperbound is None or b > self.upperbound:
            self.upperbound = b

        self.content.append((bound, func))
        self.content.sort(key=lambda x : x[0][0])

    def integrate(self, a, b):
        """integrate the func_set on the section [a, b]"""
        if a is None:
            a = self.lowerbound
        if b is None:
            b = self.upperbound
        funcs = [i for i in self.content if i[0][0] < b and i[0][1] > a]
        s = 0.
        for i in funcs:
            lowerbound = max(a, i[0][0])
            upperbound = min(b, i[0][1])
            s += i[1].integrate(lowerbound, upperbound)
        return s

    def from_matrix(self, mat, xs):
        """build the func_set from the interpolation result"""
        if len(mat) + 1 != len(xs):
            raise ValueError('the node number and the function matrix dimension conflict')
        for i, x in enumerate(mat):
            xi = xs[i]
            [a, b, c, d] = x[::-1]
            d = d - a*xi**3 + b * xi * xi - c * xi
            c = c + 3*a*xi*xi - 2*xi*b
            b -= 3*a*xi
            self.append(CubicFunc(tuple([a, b, c, d])), (xi, xs[i+1]))
