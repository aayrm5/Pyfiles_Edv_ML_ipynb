



data_file='~/Dropbox/March onwards/Python Data Science/Data/diabetic_data_small.csv'

import pandas as pd
import numpy as np

dd=pd.read_csv(data_file)



dd.head()



dd.shape



dd['readmitted'].value_counts()




dd['readmitted']=np.where(np.logical_or(dd['readmitted']=="NO", 
                                        dd['readmitted']==">30"),
                          "NO","YES")




race_dummies=pd.get_dummies(dd['race'],prefix='race')

dd=pd.concat([dd,race_dummies],1)
drop_vars=["race_?","race"]



dd['gender_F']=np.where(dd['gender']=="Female",1,0)
drop_vars=drop_vars+['gender']




dd['age1'],dd['age2']=zip(*dd['age'].apply(lambda x: x.split('-', 1)))



dd['age1']=[x.replace('[',"") for x in dd['age1']]
dd['age2']=[x.replace(')',"") for x in dd['age2']]

dd['age']=0.5*(pd.to_numeric(dd['age1'])+pd.to_numeric(dd['age2']))

drop_vars=drop_vars+['age1','age2']




dd['weight'].value_counts()



drop_vars+=['weight']



dd['admission_type_id'].value_counts()




ad_type_dummies=pd.get_dummies(dd['admission_type_id'])
ad_type_dummies.columns=["ad_type_"+str(x) for x in 
                         ad_type_dummies.columns]
dd=pd.concat([dd,ad_type_dummies],1)
drop_vars+=['ad_type_7']+['admission_type_id']



dd['discharge_disposition_id'].value_counts()




dis_dispo_dummies=pd.get_dummies(dd['discharge_disposition_id'])
dis_dispo_dummies.columns=['dis_dispo_'+str(x) for x 
                           in dis_dispo_dummies.columns]
dd=pd.concat([dd,dis_dispo_dummies],1)
drop_vars+=['dis_dispo_'+str(x) for x in 
            [15,9,17,27,24,28,8,14,13,23,7,25,4]]+['discharge_disposition_id']



dd['admission_source_id'].value_counts()




ad_source_dummies=pd.get_dummies(dd['admission_source_id'])
ad_source_dummies.columns=['ad_source_'+str(x) for 
                           x in ad_source_dummies.columns]
dd=pd.concat([dd,ad_source_dummies],1)
drop_vars+=['ad_source_'+str(x) for x in 
            [8,10,11,9,20,3,5]]+['admission_source_id']



dd['payer_code'].value_counts()




payer_code_dummies=pd.get_dummies(dd['payer_code'])
payer_code_dummies.columns=['payer_code_'+str(x) for 
                            x in payer_code_dummies.columns]
dd=pd.concat([dd,payer_code_dummies],1)
drop_vars+=['payer_code_'+str(x) for x in 
            ['OT','MP','SI','CH','WC','DM','PO']]+['payer_code']



dd['medical_specialty'].value_counts()




for ms in ['?','InternalMedicine','Emergency/Trauma',
           'Family/GeneralPractice','Cardiology','Surgery-General',
          'Nephrology','Orthopedics','Orthopedics-Reconstructive',
           'Radiologist']:
    v_name=ms.replace("?","NoRecord").replace("/","_").replace("-","_")
    dd['ms_'+v_name]=np.where(dd['medical_specialty']==ms,1,0)

    drop_vars+=['medical_specialty']+['diag_1','diag_2','diag_3',
                                  'examide','citoglipton']




dd=dd.drop(drop_vars,1)
drop_vars=[]

for col in dd.columns:
    if(dd[col].dtype=="object"):
        temp=pd.get_dummies(dd[col],drop_first=True)
        temp.columns=[col+'_'+str(x) for x in temp.columns]
        drop_vars+=[col]
        dd=pd.concat([dd,temp],1)
dd=dd.drop(drop_vars,1)


dd.columns=[x.replace("-","_").replace("?","_") for x in dd.columns]




from sklearn.cross_validation import train_test_split

dd_train, dd_test = train_test_split(dd, test_size = 0.2,random_state=2)
y_train=dd_train['readmitted_YES']
x_train=dd_train.drop(['readmitted_YES','encounter_id','patient_nbr'],1)

y_test=dd_test['readmitted_YES']
x_test=dd_test.drop(['readmitted_YES','encounter_id','patient_nbr'],1)




from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4,
                    random_state=2,class_weight='balanced'),
                         algorithm="SAMME",
                         n_estimators=1000,
                        learning_rate=0.1,
                        random_state=2)



from sklearn.metrics import roc_auc_score,accuracy_score,f1_score



bdt.fit(x_train,y_train)



roc_auc_score(y_test,bdt.predict(x_test))



accuracy_score(y_test,bdt.predict(x_test))



f1_score(y_test,bdt.predict(x_test))




import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import RandomizedSearchCV


clf=XGBClassifier(objective="binary:logistic",silent=False)




def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
param_dist = {
              "max_depth": [2,3,4,5,6],
              "learning_rate":[0.01,0.05,0.1,0.3,0.5],
    "min_child_weight":[4,5,6],
              "subsample":[i/10.0 for i in range(6,10)],
 "colsample_bytree":[i/10.0 for i in range(6,10)],
               "reg_alpha":[1e-5, 1e-2, 0.1, 1, 100],
              "gamma":[i/10.0 for i in range(0,5)],
    "n_estimators":[100,500,700,1000],
    'scale_pos_weight':[2,3,4,5,6,7,8,9]

              }
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring='roc_auc')
random_search.fit(x_train, y_train)
report(random_search.grid_scores_)



clf=XGBClassifier(objective="binary:logistic",silent=False,gamma=0.4,
                  colsample_bytree=0.6,
                  subsample=0.8,n_estimators=100,
                  reg_alpha=1,max_depth=3,learning_rate=0.01,
                 min_child_weight=5,scale_pos_weight=9)



clf.fit(x_train,y_train)



roc_auc_score(y_test,clf.predict(x_test))



accuracy_score(y_test,clf.predict(x_test))



f1_score(y_test,clf.predict(x_test))




