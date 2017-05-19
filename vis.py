# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import arange


class Canvas(object):
    def __init__(self):
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        self.axes.grid(True)

    def p(self, *args, **kwargs):
        self.axes.plot(*args, **kwargs)

    def ani(self, *args, **kwargs):
        animation.ArtistAnimation(self.figure, *args, **kwargs)

    def show(self):
        self.figure.show()

    def point_set(self, x, y):
        self.axes.plot(x, y, 'ro')

    def point_line(self, x, y):
        self.axes.plot(x, y, 'b-')

    def vertical_line(self, x):
        self.axes.plot([x, x], [0, 1], 'r-')

    def function(self, f, a, b, style='b-', interval=0.02):
        x = arange(a, b, interval)
        y = f(x)
        self.axes.plot(x, y, style)

    def func_set(self, f, style='b-', interval=0.02):
        for i in f:
            (x, y), f = i
            self.function(f.vec(), x, y, style, interval)
