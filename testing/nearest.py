# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:39:03 2021

@author: Joel Tiogo
"""

def nearest_square(num):
    root =0
    while(root+1)**2 <=num:
        root+=1
    return root **2