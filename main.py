# -*- coding:utf-8 -*-
from misc import *
import Func, vis, data 
from interpolation import interpolation as inter
import numpy as np


body= data.Rs
body_xs = body[:, 0]  # get the body section time vector
body_ys = body[:, 1]
lowerbound = np.min(body_xs)  # get the lowerbound of the time of the body section
upperbound = np.max(body_xs)  # get the upperbound ....
func_set = Func.FuncSet()  # build a empty function set
# interpolation the body section data to get the cubic functions and copy them to func_set
func_set.from_matrix(inter(body), body_xs)
print(func_set)  # print the func_set
con = vis.Canvas()  # get a canvas
con.func_set(func_set)  # draw func_set on the canvas
con.show()  # show the canvas
# get the balanced through binary search algorithm
# get the corrected temperature
input('press any key to end')
