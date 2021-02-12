



data_file=r'/Users/lalitsachan/Dropbox/March onwards/Python Data Science/Data/Existing Base.csv'

import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn import tree
import numpy as np
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

bd=pd.read_csv(data_file)




bd.loc[bd["children"]=="Zero","children"]="0"
bd.loc[bd["children"]=="4+","children"]="4"
bd["children"]=pd.to_numeric(bd["children"],errors="coerce")
bd["y"]=1
bd.loc[bd["Revenue Grid"]==2,"y"]=0
bd=bd.drop(["Revenue Grid"],1)
for i in range(len(bd)):
    if bd["age_band"][i] in ["71+","65-70","51-55","45-50"]:
        bd.loc[i,"age_band"]="ab_10"
    if bd["age_band"][i] in ["55-60","41-45","31-35","22-25","26-30"]:
        bd.loc[i,"age_band"]="ab_11"
    if bd["age_band"][i]=="36-40":
        bd.loc[i,"age_band"]="ab_13"
    if bd["age_band"][i]=="18-21":
        bd.loc[i,"age_band"]="ab_17"
    if bd["age_band"][i]=="61-65":
        bd.loc[i,"age_band"]="ab_9"
ab_dummies=pd.get_dummies(bd["age_band"])
bd=pd.concat([bd,ab_dummies],1)
bd=bd.drop(["age_band","Unknown"],1)
bd["st_partner"]=0
bd["st_singleNm"]=0
bd["st_divSep"]=0
bd.loc[bd["status"]=="Partner","st_partner"]=1
bd.loc[bd["status"]=="Single/Never Married","st_singleNm"]=1
bd.loc[bd["status"]=="Divorced/Separated","st_divSep"]=1
bd=bd.drop(["status"],1)
for i in range(len(bd)):
    if bd["occupation"][i] in ["Unknown","Student","Secretarial/Admin",
                               "Other","Manual Worker"]:
        bd.loc[i,"occupation"]="oc_11"
    if bd["occupation"][i] in ["Professional","Business Manager"]:
        bd.loc[i,"occupation"]="oc_12"
    if bd["occupation"][i]=="Retired":
        bd.loc[i,"occupation"]="oc_10"
oc_dummies=pd.get_dummies(bd["occupation"])
bd=pd.concat([bd,oc_dummies],1)

bd=bd.drop(["occupation","Housewife"],1)
bd["ocp_10"]=0
bd["ocp_12"]=0

for i in range(len(bd)):
    if bd["occupation_partner"][i] in ["Unknown","Retired","Other"]:
        bd.loc[i,"ocp_10"]=1
    if bd["occupation_partner"][i] in ["Student","Secretarial/Admin"]:
        bd.loc[i,"ocp_12"]=1
bd=bd.drop(["occupation_partner","TVarea","post_code","post_area","region"],1)
bd["hs_own"]=0
bd.loc[bd["home_status"]=="Own Home","hs_own"]=1
del bd["home_status"]
bd["gender_f"]=0
bd.loc[bd["gender"]=="Female","gender_f"]=1
del bd["gender"]
bd["semp_yes"]=0
bd.loc[bd["self_employed"]=="Yes","semp_yes"]=1
del bd["self_employed"]
bd["semp_part_yes"]=0
bd.loc[bd["self_employed_partner"]=="Yes","semp_part_yes"]=1
del bd["self_employed_partner"]
bd["fi"]=4
bd.loc[bd["family_income"]=="< 8,000, >= 4,000","fi"]=6
bd.loc[bd["family_income"]=="<10,000, >= 8,000","fi"]=9
bd.loc[bd["family_income"]=="<12,500, >=10,000","fi"]=11.25
bd.loc[bd["family_income"]=="<15,000, >=12,500","fi"]=13.75
bd.loc[bd["family_income"]=="<17,500, >=15,000","fi"]=16.25
bd.loc[bd["family_income"]=="<20,000, >=17,500","fi"]=18.75
bd.loc[bd["family_income"]=="<22,500, >=20,000","fi"]=21.25
bd.loc[bd["family_income"]=="<25,000, >=22,500","fi"]=23.75
bd.loc[bd["family_income"]=="<27,500, >=25,000","fi"]=26.25
bd.loc[bd["family_income"]=="<30,000, >=27,500","fi"]=28.75
bd.loc[bd["family_income"]==">=35,000","fi"]=35
bd=bd.drop(["family_income"],1)




bd.dropna(axis=0,inplace=True)
bd_train, bd_test = train_test_split(bd, test_size = 0.2,random_state=2)
x_train=bd_train.drop(["y","REF_NO"],1)
y_train=bd_train["y"]
x_test=bd_test.drop(["y","REF_NO"],1)
y_test=bd_test["y"]
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)



dtree=tree.DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=10,
                                  class_weight="balanced")




dtree.fit(x_train,y_train)




from sklearn import tree

tree.export_graphviz(dtree,out_file="mytree.dot",
                     feature_names=x_train.columns,
                    class_names=["0","1"],
                     proportion=True)





predicted=dtree.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])



TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP


print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)




max_nodes=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
max_nodes



beta=2
FB_avg=[]
for max_node  in max_nodes:
    mytree = tree.DecisionTreeClassifier(criterion="entropy",
                                         max_leaf_nodes=max_node,class_weight="balanced")

    # computing average RMSE across 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    FB_total = []
    for train, test in kf:
        mytree.fit(x_train.loc[train], y_train[train])
        p = mytree.predict(x_train.loc[test])
        df=pd.DataFrame(list(zip(y_train,p)),columns=["real","predicted"])
        TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
        FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
        TN=len(df[(df["real"]==0) &(df["predicted"]==0) ])
        FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
        P=TP+FN
        N=TN+FP
        Precision=TP/(TP+FP)
        Recall=TP/P
        FB=(1+beta**2)*Precision*Recall/((beta**2)*Precision+Recall)
        FB_total.extend([FB])
    FB_avg.extend([np.mean(FB_total)])
best_max_node=np.array(max_nodes)[FB_avg==max(FB_avg)][0]

print('max_node value with best F2 score is :',best_max_node)




dtree=tree.DecisionTreeClassifier(criterion="entropy",
                                  max_leaf_nodes=best_max_node,class_weight="balanced")
dtree.fit(x_train,y_train)
predicted=dtree.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print(k)



TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

print(TP,TN,FP,FN)
print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)



from sklearn.ensemble import RandomForestClassifier





import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import RandomizedSearchCV


clf = RandomForestClassifier(verbose=1,n_jobs=-1)


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    # above line selects top n grid scores
    # for loop below , prints the rank, score and parameter combination
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

param_dist = {"n_estimators":[10,100,500,700],
              "max_depth": [3,5, None],
              "max_features": sp_randint(5, 11),
              "min_samples_split": sp_randint(5, 11),
              "min_samples_leaf": sp_randint(5, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
n_iter_search = 100

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)
report(random_search.grid_scores_)




rf=RandomForestClassifier(n_estimators=500,verbose=1,criterion='entropy',min_samples_split=7,
                         bootstrap=False,max_depth=None,max_features=8,min_samples_leaf=5,
                          class_weight="balanced")



rf.fit(x_train,y_train)



predicted=rf.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print(k)



TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

print(TP,TN,FP,FN)
print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)




importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list(x_train.columns)[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), list(x_train.columns))
plt.xlim([-1, x_train.shape[1]])
plt.show()





data=x_train.copy()

features=x_train.columns

for f in features:
    if f=='Average Credit Card Transaction':pass
    else:
        data[f]=data[f].mean()

data=data.drop_duplicates()
data['response']=pd.Series(list(zip(*rf.predict_proba(x_train)))[1])



from ggplot import *
ggplot(data,aes(x='Average Credit Card Transaction',y='response'))+geom_smooth(se=False,span=0.2)+xlab("Average Credit Card Transaction")+ylab('Response')+ggtitle('Partial Dependence Plot \n Response Vs Average Credit Card transactions')




