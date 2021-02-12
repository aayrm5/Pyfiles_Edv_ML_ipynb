




import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split



data_file='~/Dropbox/March onwards/Python Data Science/Data/paydayloan_collections.csv'
pdl=pd.read_csv(data_file)




pdl["payment"]=np.where(pdl["payment"]=="Success",1,0)

k=pdl.columns

for col in k:
    if pdl[col].dtype=='object':
        temp=pd.get_dummies(pdl[col],drop_first=True,prefix=col)
        pdl=pd.concat([pdl,temp],1)
        pdl.drop([col],axis=1,inplace=True)


pdl.dropna(axis=0,inplace=True)

pdl_train, pdl_test = train_test_split(pdl, test_size = 0.2,random_state=2)

x_train=pdl_train.drop(["payment"],1)
y_train=pdl_train["payment"]

x_test=pdl_test.drop(["payment"],1)
y_test=pdl_test["payment"]

x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)




from sklearn.grid_search import GridSearchCV



param_grid1 = {'n_neighbors':list(range(5,10,15)),"p":[1,2,3,10000],
              'weights':['uniform','distance']}
grid = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid1,cv=10,scoring="roc_auc")
grid.fit(x_train,y_train)




print(grid.best_estimator_)



knn=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=1,
           weights='uniform')
knn.fit(x_train,y_train)



pd.crosstab(pdl_test['payment'],knn.predict(x_test))



from sklearn.metrics import accuracy_score



accuracy_score(y_test,knn.predict(x_test))




from sklearn import svm



svmClf=svm.SVC(class_weight='balanced')
param_grid1 = {'C':[0.01,0.1,1.0],"kernel":['linear', 'poly', 'rbf', 'sigmoid']}
grid = GridSearchCV(svm.SVC(),param_grid=param_grid1,cv=10,scoring="roc_auc")
grid.fit(x_train,y_train)



print(grid.best_estimator_)



clf=svm.SVC()
clf.fit(x_train,y_train)



pd.crosstab(pdl_test['payment'],clf.predict(x_test))



accuracy_score(y_test,clf.predict(x_test))





data_file='~/Dropbox/March onwards/Python Data Science/Data/emissions.csv'
em=pd.read_csv(data_file)




k=em.columns
for col in k:
    if em[col].dtype=='object':
        temp=pd.get_dummies(em[col],drop_first=True,prefix=col)
        em=pd.concat([em,temp],1)
        em.drop([col],axis=1,inplace=True)

em.dropna(axis=0,inplace=True)

em_train, em_test = train_test_split(em, test_size = 0.2,random_state=2)

x_train=em_train.drop(["ppm"],1)
y_train=em_train["ppm"]

x_test=em_test.drop(["ppm"],1)
y_test=em_test["ppm"]

x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)




from sklearn.neighbors import KNeighborsRegressor



param_grid1 = {'n_neighbors':list(range(5,10,15)),"p":[1,2,3],
              'weights':['uniform','distance'],'algorithm' : ['auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’]}
grid = GridSearchCV(KNeighborsRegressor(),param_grid=param_grid1,cv=10)
grid.fit(x_train,y_train)



print(grid.best_estimator_)



knn=KNeighborsRegressor()
knn.fit(x_train,y_train)



predicted=knn.predict(x_test)



residual=predicted-y_test



rmse_knn=np.sqrt(np.dot(residual,residual)/len(predicted))

rmse_knn




param_grid1 = {'C':list(range(0.1,0.2,1,10,100)),'epsilon' :[0.05,0.1,0.2,0.3,0.4],'degree':[2,3,4,5],
               "kernel":['linear', 'poly', 'rbf', 'sigmoid']}
grid = GridSearchCV(svm.SVR(),param_grid=param_grid1,cv=10)
grid.fit(x_train,y_train)



print(grid.best_estimator_)



sr=svm.SVR()
sr.fit(x_train,y_train)



from sklearn.metrics import mean_squared_error



rmse_svm=np.sqrt(mean_squared_error(y_test,sr.predict(x_test)))

rmse_svm

