# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:13:35 2021

@author: User
"""

import numpy as np

NumP_Array = np.array([[1,2,3], [4,6,7]])

NP1 = np.array([[1,3],[4,5]])

NP2 = np.array([[3,4],[5,7]])

MNP = np.dot(NP1, NP2)

NMP3 = NP1*NP2

MNP4 = NP1+NP2

Sub1 = NP1-NP2

Sub2 = np.subtract(NP1,NP2)

El_Sum1 = np.sum(NP1)

El_Sum1 = np.sum(NP2)

Broad_Nump = NP1+3