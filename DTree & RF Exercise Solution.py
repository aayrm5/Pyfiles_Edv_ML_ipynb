




import warnings
warnings.filterwarnings('ignore')



import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn import tree
import numpy as np
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

data_file='~/Dropbox/March onwards/Python Data Science/Data/loans data.csv'
ld=pd.read_csv(data_file)


for col in ["Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=ld[col].astype("str")
    ld[col]=[x.replace("%","") for x in ld[col]]

for col in ["Amount.Requested","Amount.Funded.By.Investors","Open.CREDIT.Lines",
            "Revolving.CREDIT.Balance",
           "Inquiries.in.the.Last.6.Months","Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=pd.to_numeric(ld[col],errors="coerce")


ld["LL_36"]=np.where(ld['Loan.Length']=="36 months",1,0)
ld.drop('Loan.Length',axis=1,inplace=True)


for i in range(len(ld.index)):
    if ld["Loan.Purpose"][i] in ["car","educational","major_purchase"]:
        ld.loc[i,"Loan.Purpose"]="cem"
    if ld["Loan.Purpose"][i] in ["home_improvement","medical","vacation","wedding"]:
        ld.loc[i,"Loan.Purpose"]="hmvw"
    if ld["Loan.Purpose"][i] in ["credit_card","house","other","small_business"]:
        ld.loc[i,"Loan.Purpose"]="chos"
    if ld["Loan.Purpose"][i] in ["debt_consolidation","moving"]:
        ld.loc[i,"Loan.Purpose"]="dm"

lp_dummies=pd.get_dummies(ld["Loan.Purpose"],prefix="LP")


lp_dummies.drop("LP_renewable_energy",1,inplace=True)


ld=pd.concat([ld,lp_dummies],1)
ld=ld.drop("Loan.Purpose",1)

ld=ld.drop(["State"],1)

ld["ho_mort"]=np.where(ld["Home.Ownership"]=="MORTGAGE",1,0)
ld["ho_rent"]=np.where(ld["Home.Ownership"]=="RENT",1,0)
ld=ld.drop(["Home.Ownership"],1)


ld['f1'], ld['f2'] = zip(*ld['FICO.Range'].apply(lambda x: x.split('-', 1)))

ld["fico"]=0.5*(pd.to_numeric(ld["f1"])+pd.to_numeric(ld["f2"]))

ld=ld.drop(["FICO.Range","f1","f2"],1)

ld["Employment.Length"]=ld["Employment.Length"].astype("str")
ld["Employment.Length"]=[x.replace("years","") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("year","") for x in ld["Employment.Length"]]

ld["Employment.Length"]=[x.replace("n/a","< 1") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("10+","10") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("< 1","0") for x in ld["Employment.Length"]]
ld["Employment.Length"]=pd.to_numeric(ld["Employment.Length"],errors="coerce")

ld.dropna(axis=0,inplace=True)

ld_train, ld_test = train_test_split(ld, test_size = 0.2,random_state=2)

x_train=ld_train.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_train=ld_train["Interest.Rate"]

x_test=ld_test.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_test=ld_test["Interest.Rate"]




x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)



from sklearn.grid_search import GridSearchCV



import time
start_time = time.time()

param_grid1 = {'max_depth':list(range(20,81,10)),'max_features':list(range(7,16,2)),
               "max_leaf_nodes":list(range(10,100,5))}
grid = GridSearchCV(tree.DecisionTreeRegressor(criterion="mse",random_state=2),param_grid=param_grid1,cv=10)
grid.fit(x_train,y_train)


print("--- %s seconds ---" % (time.time() - start_time))




print(grid.best_estimator_)




dtree=tree.DecisionTreeRegressor(criterion='mse', max_depth=20, max_features=13,
           max_leaf_nodes=40, min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=2,
           splitter='best')



dtree.fit(x_train,y_train)



predicted=dtree.predict(x_test)



residual=predicted-y_test



rmse_dtree=np.sqrt(np.dot(residual,residual)/len(predicted))

rmse_dtree






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



x_train.shape



from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV



params={'n_estimators':[100,200,500,700,1000],
       'criterion':['gini','entropy'],
       'min_samples_split':[5,6,7,8,9,10],
       'bootstrap':[True,False],
       'max_depth':[None,5,10,15,20],
       'max_features':[5,10,15,20,30,40,50],
       'min_samples_leaf':[5,6,7,8,9,10]}



start_time = time.time()
clf=RandomForestClassifier(class_weight="balanced",verbose=1,n_jobs=-1)

n_iter_search = 20


random_search = RandomizedSearchCV(clf, param_distributions=params,
                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))



print(random_search.best_estimator_)



clf=RandomForestClassifier(bootstrap=False, class_weight='balanced',
            criterion='entropy', max_depth=20, max_features=30,
            max_leaf_nodes=None, min_samples_leaf=8, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=False, random_state=None, verbose=1,
            warm_start=False)



clf.fit(x_train,y_train)



predicted=clf.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print(k)



TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)



importances = clf.feature_importances_
importances


indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list(x_train.columns)[f], importances[indices[f]]))




def part_plot(data,classf,varname):
    # we need to create a copy otherwise these changes 
    # will reflect in the original data as well
    d=data.copy()
    features=d.columns

    for f in features:
        if f==varname:pass
        else:
            d[f]=d[f].mean()

    d=d.drop_duplicates()
    d['response']=pd.Series(list(zip(*classf.predict_proba(d)))[1])


    print(ggplot(d,aes(x=varname,y='response'))+    geom_smooth(se=False,span=0.5)+xlab(varname)+    ylab('Response')+    ggtitle('Partial Dependence Plot \n Response Vs '+varname))



from ggplot import *
part_plot(x_train,clf,'var3')
part_plot(x_train,clf,'var4')
part_plot(x_train,clf,'var5')
part_plot(x_train,clf,'var6')



from sklearn.ensemble import ExtraTreesClassifier





ext=ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
            criterion='entropy', max_depth=20, max_features=30,
            max_leaf_nodes=None, min_samples_leaf=8, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            verbose=1)



ext.fit(x_train,y_train)



predicted=ext.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print(k)



TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)



importances = ext.feature_importances_
importances


indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list(x_train.columns)[f], importances[indices[f]]))



part_plot(x_train,ext,'var3')
part_plot(x_train,ext,'var4')
part_plot(x_train,ext,'var5')
part_plot(x_train,ext,'var6')






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



from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics.scorer import make_scorer




def loss_func(x, y):return((np.abs(x-y)).mean())

my_scorer = make_scorer(loss_func, greater_is_better=False)



start_time = time.time()
rg = RandomForestRegressor(n_jobs=-1,verbose=1)


param_dist = {"n_estimators":[10,100,500,700],
              "max_depth": [3,5, None],
              "max_features": sp_randint(5, 11),
              "min_samples_split": sp_randint(5, 11),
              "min_samples_leaf": sp_randint(5, 11),
              "bootstrap": [True, False]}

n_iter_search = 20
random_search = RandomizedSearchCV(rg, param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   scoring=my_scorer,
                                   cv=10,
                                   random_state=2)
random_search.fit(x_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))



print(random_search.best_estimator_)






rg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=9, max_leaf_nodes=None, min_samples_leaf=8,
           min_samples_split=8, min_weight_fraction_leaf=0.0,
           n_estimators=700, n_jobs=-1, 
           verbose=1, warm_start=False)



rg.fit(x_train,y_train)



predicted=rg.predict(x_test)



residual=predicted-y_test



rmse=np.sqrt(np.dot(residual,residual)/len(predicted))

rmse

