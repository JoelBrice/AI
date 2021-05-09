"""""""""""
Clustering

"""""""""""
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()


Data_iris = iris.data


"""""""""""
k-mean clustering

"""""""""""

from sklearn.cluster import KMeans

KMNS = KMeans(n_clusters=3)

KMNS.fit(Data_iris)


Labels = KMNS.predict(Data_iris)

Ctn = KMNS.cluster_centers_


plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.scatter(Ctn[:,2],Ctn[:,3], marker= 'o', color = 'red', s=120)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

KMNS.inertia_

K_inertia = []

for i in range(1,10):
    KMNS = KMeans(n_clusters=i, random_state=44)
    KMNS.fit(Data_iris)
    K_inertia.append(KMNS.inertia_)
    
    
plt.plot(range(1,10),K_inertia, color='green', marker= 'o')
plt.xlabel('number of k')
plt.ylabel('Inertia')
plt.show()
    
"""""""""""
DBSCAN

"""""""""""

from sklearn.cluster import DBSCAN

DBS = DBSCAN(eps = 0.7, min_samples= 4)

DBS.fit(Data_iris)

Labels = DBS.labels_

plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.show()


"""""""""""
Hierarchical Clustering

"""""""""""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

HR = linkage(Data_iris, method = 'complete')

#Dnd = dendrogram(HR)


Labels = fcluster(HR, 4, criterion = 'distance')

plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.show()



