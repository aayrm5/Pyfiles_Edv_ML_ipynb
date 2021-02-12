



import warnings
warnings.filterwarnings('ignore')

myfile='~/Dropbox/March onwards/Python Data Science/Data/winequality-red.csv'

import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

wine=pd.read_csv(myfile,sep=";")




wine=wine[["sulphates","alcohol","pH"]]

wine_std=pd.DataFrame(scale(wine),columns=list(wine.columns))



Ks=np.linspace(2,15,14)



ssw=[]
for k in Ks:
    kmeans=KMeans(n_clusters=int(k))
    kmeans.fit(wine_std)
    sil_score=silhouette_score(wine_std,kmeans.labels_)
    print("for inertia:" ,kmeans.inertia_ ,"and silhouette score:",sil_score,"number of clusters are:", int(k))
    ssw.append(kmeans.inertia_)
plt.plot(Ks,ssw)



k = 6
kmeans = KMeans(n_clusters=k)
kmeans.fit(wine_std)



labels = kmeans.labels_
wine_std["cluster"]=labels



from ggplot import *



ggplot(wine_std,aes(x='sulphates',y='alcohol'))+geom_point(aes(color='cluster'),size=50)



ggplot(wine_std,aes(x='sulphates',y='pH'))+geom_point(aes(color='cluster'),size=50)



ggplot(wine_std,aes(x='alcohol',y='pH'))+geom_point(aes(color='cluster'),size=50)







myfile='~/Dropbox/March onwards/Python Data Science/Data/Wholesale customers data.csv'

groc=pd.read_csv(myfile)

groc=groc[["Milk","Grocery"]]

groc_std=pd.DataFrame(scale(groc),columns=list(groc.columns))



from sklearn.cluster import DBSCAN
from sklearn import metrics



r=np.linspace(0.5,5)
for epsilon in r:
    db = DBSCAN(eps=epsilon, min_samples=20, metric='euclidean').fit(groc_std)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clust=len(set(labels))-1
    outlier=np.round(np.count_nonzero(labels == -1)/len(labels)*100,2)

    print('Estimated number of clusters: %d', n_clust)
    print("For epsilon =", epsilon ,", percentage of outliers is: ",outlier)





db = DBSCAN(eps=0.77, min_samples=10, metric='euclidean').fit(groc_std)
groc_std['cluster']=[str(x) for x in db.labels_]



ggplot(groc_std,aes(x='Milk',y='Grocery',color='cluster'))+geom_point()




from sklearn.decomposition import FactorAnalysis



data_file='~/Dropbox/March onwards/Python Data Science/Data/cars.csv'
cars=pd.read_csv(data_file)



X_cars=cars.drop(['Name'],1)



X_cars=pd.DataFrame(scale(X_cars),columns=X_cars.columns)



fa=FactorAnalysis(n_components=4,max_iter=1000)



fa.fit(X_cars)



nvar=fa.noise_variance_
plt.plot(nvar)



print(*zip(X_cars.columns,nvar))




X_cars=X_cars.drop(['Width'],1)

fa=FactorAnalysis(n_components=4,max_iter=1000)

fit=fa.fit(X_cars)
nvar=fa.noise_variance_
print(*zip(X_cars.columns,nvar))
plt.plot(nvar)



X_cars=X_cars.drop(['Length'],1)

fa=FactorAnalysis(n_components=4,max_iter=1000)

fit=fa.fit(X_cars)
nvar=fa.noise_variance_
print(*zip(X_cars.columns,nvar))
plt.plot(nvar)



X_cars=X_cars.drop(['Wheelbase'],1)

fa=FactorAnalysis(n_components=4,max_iter=1000)

fit=fa.fit(X_cars)
nvar=fa.noise_variance_
print(*zip(X_cars.columns,nvar))
plt.plot(nvar)



X_cars=X_cars.drop(['Horsepower'],1)

fa=FactorAnalysis(n_components=4,max_iter=1000)

fit=fa.fit(X_cars)
nvar=fa.noise_variance_
print(*zip(X_cars.columns,nvar))
plt.plot(nvar)



loadings=fa.components_
loadings




print(*zip(X_cars.columns,loadings[0,]))




print(*zip(X_cars.columns,loadings[1,]))




print(*zip(X_cars.columns,loadings[2,]))




print(*zip(X_cars.columns,loadings[3,]))


