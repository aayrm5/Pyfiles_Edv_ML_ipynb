




import warnings
warnings.filterwarnings('ignore')



import numpy as np
import math



l = np.array([["riya","lalit","andy"],["prof_dinesh","prof_andrew", "prof_peter"],[71,75,50]])   
print(l)

d={l[0,0]:(l[1,0],l[2,0]),l[0,1]:(l[1,1],l[2,1]),l[0,2]:(l[1,2],l[2,2])} #extracting array values using index
d




np.tril(np.full((3,3),4))



a=np.tril(np.full((3,3),4,dtype=int), -1)#to get the lower triangular matrix
print(a)

np.fill_diagonal(a,4)#to get the diagonal element
a




str_array=np.array([["me", "i", "my"],["we", "us", "ours"],["you", "yours", "them"]])
print(str_array)
list=str_array.flatten() #convert the array in a single list of all elements
print(list)


l=np.array([len(element) for element in list])
np.reshape(l,(3,3)) #shaping the list to an array






import random
x = np.random.randint(low=2,high=40, size=(3, 3))
print(x)







primes=[2,3,5,7,11,13,17,19,23,29,31,37] #prime numbers between 2 and 40

is_prime=(x==2)



is_prime



for p in primes[1:] :
    is_prime=np.logical_or(is_prime,x==p)

x[is_prime]

x[x in primes]






print(x)

sum(np.diag(x))




import math

l_1=np.array([[1,-3,6]])  
l_2=np.array([[3,5,0]])
print(l_1,l_2)

dot_prod=np.dot(l_1,l_2.T) #or use l_1*l_2
print(dot_prod)

l_1s=np.square(l_1) # A.B=|A||B|cos(theta)
print(l_1s)
l_2s=np.square(l_2)
print(l_2s)

mod_1=math.sqrt(np.sum(l_1s))
mod_2=math.sqrt(np.sum(l_2s))
print((dot_prod) / ((mod_1) * (mod_2)))

angle= math.acos((dot_prod) / ((mod_1) * (mod_2))) #to obtain angle
print(angle)




import pandas as pd
get_ipython().magic('matplotlib inline')

myfile='~/Dropbox/March onwards/Python Data Science/Data/revenue.csv'
rev=pd.read_csv(myfile,index_col=False)

rev.head()




rev['DEL_NO']=rev['DEL_NO'].astype('str')
rev["DEL_NO"].dtype




rev.describe()




missing_values=rev.isnull().sum() #obtaing missing values
print(missing_values)




rev=rev.dropna()#remove missing values

mean_rev=rev[["DataRevenue_JAN","DataRevenue_FEB","DataRevenue_MAR","DataRevenue_APR","DataRevenue_May"]].mean()
print(mean_rev) #mean of all the columns of columns data_revenue

print("----")

sum_rev=rev[["DataRevenue_JAN","DataRevenue_FEB","DataRevenue_MAR","DataRevenue_APR","DataRevenue_May"]].sum()
print(sum_rev) #sum of all the columns of columns data_revenue

print("----")

mean_use=rev[["DataUsageMB_JAN","DataUsageMB_FEB","DataUsageMB_MAR","DataUsageMB_APR","DataUsageMB_May"]].mean()
print(mean_use) #mean of all the columns of columns data_usage

print("----")

sum_use=rev[["DataUsageMB_JAN","DataUsageMB_FEB","DataUsageMB_MAR","DataUsageMB_APR","DataUsageMB_May"]].sum()
print(sum_use) #sum of all the columns of columns data_usage




rev["total_rev"]=rev.DataRevenue_JAN+rev.DataRevenue_FEB+rev.DataRevenue_MAR+rev.DataRevenue_APR+rev.DataRevenue_May
rev.head()

rev["total_use"]=rev.DataUsageMB_JAN+rev.DataUsageMB_FEB+rev.DataUsageMB_MAR+rev.DataUsageMB_APR+rev.DataUsageMB_May
rev.head()

rev=rev.drop(rev.columns[[2, 3, 4,5,6,7,8,9,10,11,17,18]],axis=1)
rev.head()




from ggplot import *

ggplot(aes(x="total_use",y="total_rev"),data=rev)+geom_point()+ggtitle("Scatter Plot")




rev_cat_data=rev.select_dtypes(['object'])

for c in rev_cat_data.columns:
    if c=="DEL_NO":pass
    else:print(c,":",rev[c].nunique())




myfile='~/Dropbox/python_data/Data/exercise.csv'
ex=pd.read_csv(myfile,index_col=False)

ex.head()




ex[(ex.exercise=='g')]




new_ex=ex[["warmup","situp","running"]]
new_ex

for col in new_ex.columns:
     new_ex[col]=pd.to_numeric(new_ex[col], errors='coerce')# converting object data type to numeric

new_ex.dtypes
new_ex.head()

new_ex=new_ex.dropna() #removing missing values

new_ex["total_time"]=new_ex.warmup+new_ex.situp+new_ex.running
new_ex

new_ex[(new_ex.total_time>30)] #Filter the dataset for which total_time>30




day=ex[["Day"]]
day

week_days=day[0:7]
week_days

dList = week_days['Day'].tolist()
dList

list(enumerate(dList,start=1))




pd.get_dummies(ex, prefix=None, prefix_sep='_', dummy_na=False, columns=["Day","exercise"], drop_first=True)






