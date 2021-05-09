
"""""""""""
NumPy Coding

"""""""""""

import numpy as np

NumP_Array = np.array([[1,2,3],[4,6,7]])

NP1 = np.array([[1,3],[4,5]])

NP2 = np.array([[3,4],[5,7]])

MNP = NP1@NP2

MNP3 = np.dot(NP1,NP2)

MNP2 = NP1*NP2

MNP4 = np.multiply(NP1,NP2)

Sum1 = NP1+NP2

Sub1 = NP1-NP2

Sub2 = np.subtract(NP1,NP2)

np.sum(NP1)

Broad_Nump = NP1+3

NP3 = np.array([[3,4]])

NP1+NP3

D = np.divide([12,14,16],5)

D = np.floor_divide([12,14,16],5)

np.math.sqrt(10)

ND = np.random.standard_normal((3,4))

UD = np.random.uniform(1,12,(3,4))

# Generate Float No.

np.random.rand()

# Generate Integer No.

Random_Ar= np.random.randint(1,50,(2,5))

Ze = np.zeros((3,4))

Ones = np.ones((3,4))


Filter_Ar = np.logical_and(Random_Ar>30,Random_Ar<50)

F_Random_Ar = Random_Ar[Filter_Ar]


Data_N = np.array([1,3,4,5,7,9])

Mean_N = np.mean(Data_N)

Median_N = np.median(Data_N)

Var_N = np.var(Data_N)

SD_N = np.std(Data_N)


NumP_Array = np.array([[1,2,3],[4,6,7]])


Var_Nump = np.var(NumP_Array,axis=1)

Var_Nump2 = np.var(NumP_Array,axis=0)











