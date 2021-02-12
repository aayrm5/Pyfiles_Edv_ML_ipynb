



import pandas as pd
import math
import numpy as np



data_file="~/Dropbox/Learning/Cash_Assistance_Engagement_Report.csv"

cash=pd.read_csv(data_file)



cash.head()




cash.dtypes



cash=cash.drop(["Month"],1)




cash=cash.drop(["Unnamed: 66"],1)




cash.corr(method='pearson', min_periods=1)




import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
get_ipython().magic('matplotlib inline')



cash.dropna(axis=0,inplace=True)



X=cash.copy()



X



X = scale(X)




pca = PCA(n_components=65)



pca.fit(X)




pca.components_



pca.components_.shape





var= pca.explained_variance_ratio_

print(var)




var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)



plt.plot(var1)




pca = PCA(n_components=8)



pca.fit(X)



X1=pca.transform(X)



pd.DataFrame(X1).corr(method='pearson', min_periods=1)




loadings=pca.components_



print(*zip(cash.columns,loadings[1,]))





from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt




data_file='~/Dropbox/March onwards/Python Data Science/Data/cars.csv'
cars=pd.read_csv(data_file)



cars.shape



cars.head()




X_cars=cars.drop(['Name'],1)



X_cars=scale(X_cars)




fa=FactorAnalysis(n_components=8,max_iter=1000)



fa.fit(X_cars)




loadings=fa.components_



print(*zip(cars.columns[1:],loadings[0,]))



print(*zip(cars.columns[1:],loadings[1,]))




nvar=fa.noise_variance_
plt.plot(nvar)



