"""""""""""""""""
SVR Hyper Parameter Tuning

"""""""""""""""""
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

x = Boston_P.data

y = Boston_P.target


from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf','linear'],
              'gamma':[1,0.1,0.01]}

grid = GridSearchCV(SVR(),parameters, refit = True, verbose=2, scoring='neg_mean_squared_error')

grid.fit(x,y)

best_params = grid.best_params_

"""""""""""""""""
K-Means Hyper Parameter Tuning

"""""""""""""""""

K_inertia = []

for i in range(1,10):
    KMNS = KMeans(n_clusters = i, random_state=44)
    KMNS.fit(Data_iris)
    K_inertia.append(KMNS.inertia_)


"""""""""""""""""
k-NN Hyper Parameter Tuning

"""""""""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now from sklearn datasets, we import the iris dataset

from sklearn.datasets import load_iris

iris = load_iris()


x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, 
                                                    train_size=0.70, random_state = 22, 
                                                    shuffle=True, stratify = y)


from sklearn.neighbors import KNeighborsClassifier

kNN_accuracy_test = []
kNN_accuracy_train = []


for k in range(1,50):
    kNN = KNeighborsClassifier(n_neighbors=k, metric= 'minkowski', p=1)
    kNN.fit(X_train,y_train)
    kNN_accuracy_train.append(kNN.score(X_train,y_train))
    kNN_accuracy_test.append(kNN.score(X_test,y_test))
    
plt.plot(np.arange(1,50), kNN_accuracy_train, label = 'train')
plt.plot(np.arange(1,50), kNN_accuracy_test, label = 'test')
plt.xlabel('k')
plt.ylabel('Score')
plt.legend()
plt.show()
    












