# -*- coding: utf-8 -*-

import math as m


def solve(a, b, c):
    """solve the equation and take the bigger root"""
    return (-b + m.sqrt(b * b - 4 * a * c)) / (2. * a)

