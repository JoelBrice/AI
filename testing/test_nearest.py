# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:39:03 2021

@author: Joel Tiogo
"""

from nearest import nearest_square;

def test_nearest_square_5():
    assert(nearest_square(5)==4)
    
def test_nearest_square_12():
    assert(nearest_square(9)==9)
    
def test_nearest_square_23():
    assert(nearest_square(23)==24)