

import os
os.chdir(r'E:/Riz/Edvancer/Python/Data')
#data_file=r'E:/Riz/Edvancer/Python/Data/loans data.csv'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn.model_selection import KFold
#get_ipython().magic('matplotlib inline')

ld=pd.read_csv('loans data.csv')

ld.head()




for col in ["Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=ld[col].astype("str")
    ld[col]=[x.replace("%","") for x in ld[col]]




ld.dtypes




for col in ["Amount.Requested","Amount.Funded.By.Investors","Open.CREDIT.Lines","Revolving.CREDIT.Balance",
           "Inquiries.in.the.Last.6.Months","Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=pd.to_numeric(ld[col],errors="coerce")



ld.dtypes




ld["Loan.Length"].value_counts()




ll_dummies=pd.get_dummies(ld["Loan.Length"])



ll_dummies.head()




ld["LL_36"]=ll_dummies["36 months"]




get_ipython().magic('reset_selective ll_dummies')




who




ld=ld.drop('Loan.Length',axis=1)



ld.dtypes




ld["Loan.Purpose"].value_counts()




round(ld.groupby("Loan.Purpose")["Interest.Rate"].mean())




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



lp_dummies.head()




ld=pd.concat([ld,lp_dummies],1)
ld=ld.drop(["Loan.Purpose","LP_renewable_energy"],1)



ld.dtypes




ld["State"].nunique()




ld=ld.drop(["State"],1)




ld["Home.Ownership"].value_counts()



ld["ho_mort"]=np.where(ld["Home.Ownership"]=="MORTGAGE",1,0)
ld["ho_rent"]=np.where(ld["Home.Ownership"]=="RENT",1,0)
ld=ld.drop(["Home.Ownership"],1)




ld["FICO.Range"].head()




ld['f1'], ld['f2'] = zip(*ld['FICO.Range'].apply(lambda x: x.split('-', 1)))




ld["fico"]=0.5*(pd.to_numeric(ld["f1"])+pd.to_numeric(ld["f2"]))

ld=ld.drop(["FICO.Range","f1","f2"],1)




ld["Employment.Length"].value_counts()



ld["Employment.Length"]=ld["Employment.Length"].astype("str")
ld["Employment.Length"]=[x.replace("years","") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("year","") for x in ld["Employment.Length"]]




round(ld.groupby("Employment.Length")["Interest.Rate"].mean(),2)




ld["Employment.Length"]=[x.replace("n/a","< 1") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("10+","10") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("< 1","0") for x in ld["Employment.Length"]]
ld["Employment.Length"]=pd.to_numeric(ld["Employment.Length"],errors="coerce")



ld.dtypes




ld.shape



ld.dropna(axis=0,inplace=True)



ld.shape




ld_train, ld_test = train_test_split(ld, test_size = 0.2,random_state=2)



lm=LinearRegression()




x_train=ld_train.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_train=ld_train["Interest.Rate"]
x_test=ld_test.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_test=ld_test["Interest.Rate"]




lm.fit(x_train,y_train)




p_test=lm.predict(x_test)

residual=p_test-y_test

rmse_lm=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_lm





coefs=lm.coef_

features=x_train.columns

list(zip(features,coefs))




alphas=np.linspace(.0001,10,100)
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)



rmse_list=[]
for a in alphas:
    ridge = Ridge(fit_intercept=True, alpha=a)

    # computing average RMSE across 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    xval_err = 0
    for train, test in kf:
        ridge.fit(x_train.loc[train], y_train[train])
        p = ridge.predict(x_train.loc[test])
        err = p - y_train[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(x_train))
    # uncomment below to print rmse values for individidual alphas
    rmse_list.extend([rmse_10cv])
best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is : ',best_alpha )





ridge=Ridge(fit_intercept=True,alpha=best_alpha)

ridge.fit(x_train,y_train)

p_test=ridge.predict(x_test)

residual=p_test-y_test

rmse_ridge=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_ridge



list(zip(x_train.columns,ridge.coef_))




alphas=np.linspace(0.0001,1,100)
rmse_list=[]
for a in alphas:
    lasso = Lasso(fit_intercept=True, alpha=a,max_iter=10000)

    # computing RMSE using 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    xval_err = 0
    for train, test in kf:
        lasso.fit(x_train.loc[train], y_train[train])
        p =lasso.predict(x_train.loc[test])
        err = p - y_train[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(x_train))
    rmse_list.extend([rmse_10cv])
    # Uncomment below to print rmse values of individual alphas
    print('{:.3f}\t {:.4f}\t '.format(a,rmse_10cv))
best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is : ',best_alpha )



lasso=Lasso(fit_intercept=True,alpha=best_alpha)

lasso.fit(x_train,y_train)

p_test=lasso.predict(x_test)

residual=p_test-y_test

rmse_lasso=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_lasso



list(zip(x_train.columns,lasso.coef_))






from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



data_file=r'/Users/lalitsachan/Dropbox/March onwards/Python Data Science/Data/Existing Base.csv'
bd=pd.read_csv(data_file)



bd.head()



bd["children"].value_counts()




bd.loc[bd["children"]=="Zero","children"]="0"
bd.loc[bd["children"]=="4+","children"]="4"
bd["children"]=pd.to_numeric(bd["children"],errors="coerce")



bd["Revenue Grid"].value_counts()



bd["y"]=np.where(bd["Revenue Grid"]==2,0,1)
bd=bd.drop(["Revenue Grid"],1)




round(bd.groupby("age_band")["y"].mean(),2)



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
ab_dummies.head()




bd=pd.concat([bd,ab_dummies],1)
bd=bd.drop(["age_band","Unknown"],1)



bd["status"].value_counts()



bd["st_partner"]=np.where(bd["status"]=="Partner",1,0)
bd["st_singleNm"]=np.where(bd["status"]=="Single/Never Married",1,0)
bd["st_divSep"]=np.where(bd["status"]=="Divorced/Separated",1,0)
bd=bd.drop(["status"],1)



round(bd.groupby("occupation")["y"].mean(),2)



for i in range(len(bd)):
    if bd["occupation"][i] in ["Unknown","Student","Secretarial/Admin","Other","Manual Worker"]:
        bd.loc[i,"occupation"]="oc_11"
    if bd["occupation"][i] in ["Professional","Business Manager"]:
        bd.loc[i,"occupation"]="oc_12"
    if bd["occupation"][i]=="Retired":
        bd.loc[i,"occupation"]="oc_10"
oc_dummies=pd.get_dummies(bd["occupation"])
oc_dummies.head()



bd=pd.concat([bd,oc_dummies],1)

bd=bd.drop(["occupation","Housewife"],1)



round(bd.groupby("occupation_partner")["y"].mean(),2)



bd["ocp_10"]=0
bd["ocp_12"]=0
for i in range(len(bd)):
    if bd["occupation_partner"][i] in ["Unknown","Retired","Other"]:
        bd.loc[i,"ocp_10"]=1
    if bd["occupation_partner"][i] in ["Student","Secretarial/Admin"]:
        bd.loc[i,"ocp_12"]=1

bd=bd.drop(["occupation_partner","TVarea","post_code","post_area","region"],1)




bd["home_status"].value_counts()



bd["hs_own"]=np.where(bd["home_status"]=="Own Home",1,0)
del bd["home_status"]




bd["gender"].value_counts()



bd["gender_f"]=np.where(bd["gender"]=="Female",1,0)
del bd["gender"]



bd["self_employed"].value_counts()



bd["semp_yes"]=np.where(bd["self_employed"]=="Yes",1,0)
del bd["self_employed"]



bd["self_employed_partner"].value_counts()



bd["semp_part_yes"]=np.where(bd["self_employed_partner"]=="Yes",1,0)
del bd["self_employed_partner"]



bd["family_income"].value_counts()




round(bd.groupby("family_income")["y"].mean(),4)



bd["fi"]=4 # by doing this , we have essentially clubbed <4000 and Unknown values . How?
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



bd.dtypes




bd.dropna(axis=0,inplace=True)
bd_train, bd_test = train_test_split(bd, test_size = 0.2,random_state=2)



x_train=bd_train.drop(["y","REF_NO"],1)
y_train=bd_train["y"]
x_test=bd_test.drop(["y","REF_NO"],1)
y_test=bd_test["y"]



logr=LogisticRegression(penalty="l1",class_weight="balanced",random_state=2)



logr.fit(x_train,y_train)



roc_auc_score(y_test,logr.predict(x_test))




prob_score=pd.Series(list(zip(*logr.predict_proba(x_train)))[1])




cutoffs=np.linspace(0,1,100)




KS_cut=[]
for cutoff in cutoffs:
    predicted=pd.Series([0]*len(y_train))
    predicted[prob_score>cutoff]=1
    df=pd.DataFrame(list(zip(y_train,predicted)),columns=["real","predicted"])
    TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
    FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
    TN=len(df[(df["real"]==0) &(df["predicted"]==0) ])
    FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
    P=TP+FN
    N=TN+FP
    KS=(TP/P)-(FP/N)
    KS_cut.append(KS)

cutoff_data=pd.DataFrame(list(zip(cutoffs,KS_cut)),columns=["cutoff","KS"])

KS_cutoff=cutoff_data[cutoff_data["KS"]==cutoff_data["KS"].max()]["cutoff"]




prob_score_test=pd.Series(list(zip(*logr.predict_proba(x_test)))[1])

predicted_test=pd.Series([0]*len(y_test))
predicted_test[prob_score_test>float(KS_cutoff)]=1

df_test=pd.DataFrame(list(zip(y_test,predicted_test)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print('confusion matrix :\n \n ',k)
TN=k.iloc[0,0]
TP=k.iloc[1,1]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP



(TP+TN)/(P+N)



TP/P



TN/N




cutoffs=np.linspace(0.010,0.99,100)
def Fbeta_perf(beta,cutoffs,y_train,prob_score):
    FB_cut=[]
    for cutoff in cutoffs:
        predicted=pd.Series([0]*len(y_train))
        predicted[prob_score>cutoff]=1
        df=pd.DataFrame(list(zip(y_train,predicted)),columns=["real","predicted"])

        TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
        FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
        FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
        P=TP+FN


        Precision=TP/(TP+FP)
        Recall=TP/P
        FB=(1+beta**2)*Precision*Recall/((beta**2)*Precision+Recall)
        FB_cut.append(FB)

    cutoff_data=pd.DataFrame(list(zip(cutoffs,FB_cut)),columns=["cutoff","FB"])

    FB_cutoff=cutoff_data[cutoff_data["FB"]==cutoff_data["FB"].max()]["cutoff"]

    prob_score_test=pd.Series(list(zip(*logr.predict_proba(x_test)))[1])

    predicted_test=pd.Series([0]*len(y_test))
    predicted_test[prob_score_test>float(FB_cutoff)]=1

    df_test=pd.DataFrame(list(zip(y_test,predicted_test)),columns=["real","predicted"])

    k=pd.crosstab(df_test['real'],df_test["predicted"])
    TN=k.iloc[0,0]
    TP=k.iloc[1,1]
    FP=k.iloc[0,1]
    FN=k.iloc[1,0]
    P=TP+FN
    N=TN+FP
    print('For beta :',beta)
    print('Accuracy is :',(TP+TN)/(P+N))
    print('Sensitivity is :',(TP/P))
    print('Specificity is :',(TN/N))
    print('\n \n \n')



Fbeta_perf(0.5,cutoffs,y_train,prob_score)
Fbeta_perf(1,cutoffs,y_train,prob_score)
Fbeta_perf(2,cutoffs,y_train,prob_score)



