



data_file=r'/Users/lalitsachan/Dropbox/March onwards/Python Data Science/Data/loans data.csv'

import pandas as pd
import math
import numpy as np

ld=pd.read_csv(data_file)

ld=ld[["Interest.Rate","FICO.Range","Amount.Requested"]]

ld["Interest.Rate"]=ld["Interest.Rate"].astype("str")
ld["Interest.Rate"]=[x.replace("%","") for x in ld["Interest.Rate"]]
ld["Interest.Rate"]=pd.to_numeric(ld["Interest.Rate"],errors="coerce")
ld["Amount.Requested"]=pd.to_numeric(ld["Amount.Requested"],errors="coerce")

ld['f1'], ld['f2'] = zip(*ld['FICO.Range'].apply(lambda x: x.split('-', 1)))
ld["fico"]=0.5*(pd.to_numeric(ld["f1"])+pd.to_numeric(ld["f2"]))

ld=ld.drop(["FICO.Range","f1","f2"],1)

ld=ld.dropna()
ld.reset_index(drop=True,inplace=True)



ld.head()




from sklearn.linear_model import LinearRegression

lm=LinearRegression()

y=ld["Interest.Rate"]
x=ld.drop(["Interest.Rate"],axis=1)



lm.fit(x,y)



lm.intercept_ , lm.coef_





x= np.c_[np.ones(x.shape[0]), x]




alpha=0.000000001 # step size
iters=100000



def GD(x, y, iters, alpha,thresh=1):
    costs = []
    m=y.size
    beta = np.random.rand(3) # random start
    preds = []
    init_cost=100000
    i=0
    change=2000000
    while((change>thresh) & (i<iters)):
        pred = np.dot(x, beta) # X*beta
        error = y-pred # Y - X*beta
        cost = np.sum(error ** 2)/m #averaged over all data points
        change=abs(init_cost-cost)
        init_cost=cost

        gradient = -2*x.T.dot(error)/m # averaged over all data points

        beta = beta - alpha * gradient  # update
        i=i+1
    print(i)
    return(beta)



GD(x, y, iters, alpha,thresh=.01)



alpha=0.00000000000001 # step size
iters=1000000



def SGD(x, y, iters, alpha,thresh=1):
    costs = []
    beta = np.random.rand(3) # random start
    preds = []
    init_cost=100000
    i=0
    change=2000000

    while((change>thresh) & (i<iters)):
        k=np.random.randint(1,2495)
        samp=x[k,]
        pred = np.dot(samp, beta) # X*beta
        error = y[k]-pred # Y - X*beta
        cost = np.sum(error ** 2)
        change=abs(init_cost-cost)
        init_cost=cost

        gradient = -2*samp.T.dot(error)

        beta = beta - alpha * gradient  # update
        i=i+1
    print(i)
    return(beta)



SGD(x,y,iters,alpha,.001)




for col in ld.columns:
    ld[col]=ld[col]/ld[col].std()



ld.head()



lm=LinearRegression()

y=ld["Interest.Rate"]
x=ld.drop(["Interest.Rate"],axis=1)
lm.fit(x,y)
lm.intercept_ , lm.coef_



x= np.c_[np.ones(x.shape[0]), x]



alpha=0.01 # step size
iters=100000
GD(x, y, iters, alpha,thresh=.000001)





