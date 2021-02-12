



data_file='~/Dropbox/March onwards/Python Data Science/Data/Grades.csv'



import pandas as pd
import numpy as np



gd=pd.read_csv(data_file)



gd.head(10)



gd['class'].value_counts()



from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold,train_test_split




train,test=train_test_split(gd,test_size=0.2)



x_train=train[['shoe size','height']]
y_train=train['class']

x_test=test[['shoe size','height']]
y_test=test['class']





from ggplot import *



get_ipython().magic('matplotlib inline')



ggplot(test,aes(x='shoe size',y='height',color='class'))+geom_point()




knn=KNeighborsClassifier(n_neighbors=5,weights='distance')
knn.fit(x_train,y_train)



test['predicted']=knn.predict(x_test)



ggplot(test,aes(x='shoe size',y='height',color='class',shape='predicted'))+geom_point(size=100)



pd.crosstab(test['class'],test['predicted'])




from sklearn.metrics import accuracy_score



accuracy_score(y_test,knn.predict(x_test))




from sklearn import svm



help(svm.SVC)




clf=svm.SVC(verbose=True,cache_size=2000,C=20,
            class_weight='balanced')
clf.fit(x_train,y_train)



test['predicted_svm']=clf.predict(x_test)



pd.crosstab(test['class'],test['predicted_svm'])



accuracy_score(y_test,clf.predict(x_test))




ggplot(test,aes(x='shoe size',y='height',color='class',shape='predicted'))+geom_point(size=100)



