



import pandas as pd
import math
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np

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




from ggplot import *
ggplot(aes(x="fico",y="Interest.Rate"),data=ld)+geom_point(color="red")+stat_smooth(color="blue",method="loess",span=0.2)




ld['fico_square']=(np.square(ld['fico']))



ld.head()



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



