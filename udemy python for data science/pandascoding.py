# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:15 2021

@author: Joel Tiogo
"""

import pandas as pd

"""
Series
"""

Age = pd.Series([10,20,30,40], index=['Age1', 'Age2', 'Age3', 'Age4'])


Age.Age3
Age[Age>20]
Age.index = ['A1', 'A2', 'A3','A4']

print(Age)

import numpy as np
DF = np.array([[20,10,8], [25,8,10], [27,5,3],[30,9,7]])

Data_set = pd.DataFrame(DF)

Data_set = pd.DataFrame(DF, index=['S1', 'S2','S3', 'S4'], columns=['Age', 'Grade1', 'Grade2'])

Data_set['Grade3'] = [9,6,7,10]

Data_set.loc['S2']
Data_set.iloc[1][3]
Data_set.iloc[1,3]


Data_set.iloc[:,1:3]

Data_set.replace(10,12)
Data_set.replace({12:10, 9:30})

Data_set.head(3)
Data_set.tail(2)

Data_set.sort_values('Grade1', axis=0,ascending=True)
Data_set.sort_index(axis=0,ascending=False)


Data = pd.read_csv('Power_data.csv')



