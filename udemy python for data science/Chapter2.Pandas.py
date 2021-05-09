
"""
Pandas

"""

import pandas as pd

"""
Series

"""

Age = pd.Series([10,20,30,40],index=['age1','age2','age3','age4'])

Age.age3

Filtered_Age = Age[Age>10]


# Calling Values of the Series
Age.values


# Calling Indices of the Series
Age.index

Age.index = ['A1','A2','A3','A4']

Age.index

"""""""""
DataFrame

"""""""""

import numpy as np

DF = np.array([[20,10,8],[25,8,10],[27,5,3],[30,9,7]])

Data_Set = pd.DataFrame(DF)

Data_Set = pd.DataFrame(DF,index = ['S1','S2','S3','S4'])

Data_Set = pd.DataFrame(DF,index = ['S1','S2','S3','S4'],columns = ['Age','Grade1','Grade2'])


Data_Set['Grade3'] = [9,6,7,10]


Data_Set.loc['S2']

Data_Set.loc[1][3]

Data_Set.iloc[1][3]

Data_Set.iloc[1,3]

Data_Set.iloc[:,0]

Data_Set.iloc[:,3]

Filtered_Data = Data_Set.iloc[:,1:3]

Data_Set.drop('Grade1',axis=1)

Data_Set = Data_Set.replace(10,12)

Data_Set = Data_Set.replace({12:10, 9:30})

Data_Set.head(3)

Data_Set.tail(2)

Data_Set.sort_values('Grade1',ascending=True)

Data_Set.sort_index(axis=0, ascending = False)

Data = pd.read_csv('Data_Set.csv')




