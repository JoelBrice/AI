# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:37:07 2021

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Data_set1 = pd.read_csv('Data_set.csv')

Data_set2 = pd.read_csv('Data_set.csv', header=2)

Data_set3 = Data_set2.rename(columns={'Temperature': 'Temp', 'E_Plug': 'Plug', 'E_Heat': 'Heat','No. Occupants':'Occupants' })

Data_set4 = Data_set3.drop('Occupants', axis = 1)
Data_set3.drop('Occupants', axis = 1, inplace=True)

Data_set5= Data_set4.drop(2, axis=0)
Data_set6 = Data_set5.reset_index(drop=True)
Data_set6.describe()

Min_item = Data_set6['Heat'].min()

Data_set6['Heat'][Data_set6['Heat']== Min_item]

Data_set6['Heat'].replace(-4, 21, inplace=True)