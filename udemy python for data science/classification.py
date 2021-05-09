# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:13:37 2021

@author: User
"""

import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
iris.feature_names

Data_iris = iris.data
Data_iris = pd.DataFrame(Data_iris, columns=iris.feature_names)

Data_iris['label'] = iris.target

plt.scatter(Data_iris.iloc[:,2], Data_iris.iloc[:,3])


