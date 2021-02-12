



import warnings
warnings.filterwarnings('ignore')




import numpy as np
import math

b = np.array([[1,2,3],[4,5,6]])   
b




b.shape                     




print(b)
b[0, 0], b[0, 1], b[1, 0]   





np.zeros((2,2))



np.ones((3,2))



np.full((2,2), math.pi)



np.full((3,2),4,dtype=int)



np.eye(5)



np.random.random((4,3))




np.random.random()




90*np.random.random()+5




a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
print("this is b",":\n",b)



c=a[:2, 1:3].copy()
print("this is c",":\n",c)




print(a)
print(a[0, 1])
print(b[0,0])



b[0, 0] = 77
print(a)




print(a[0,2])
print(c[0,1])



c[0, 1] = 99
print(a)
print(c)




a = np.array([[1,2], [3, 4], [5, 6]])
a



a[[0, 1, 2], [0, 1, 0]]




a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
a



b = np.array([0, 2, 0, 1])
c=np.arange(4)
c



a[c, b]




a



a[c, b] += 10
a




a = np.array([[1,2], [3, 4], [5, 6]])
a



c=a > 2
print(c)




print(a[c])
print(a[c].shape)




a[(a>2) | (a<5)] , a[(a>2) & (a<5)]





x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])



x



y



x+y



np.add(x,y)



print(x-y)



np.subtract(x,y)



print(x)
print("~~~~~")

print(y)
print("~~~~~")
print(x * y)



np.multiply(x, y)



print(x/y)



np.divide(x,y)




np.sqrt(x)



math.sqrt(x)




v = np.array([9,10])
v



w = np.array([11, 12])
w



v.dot(w)




print(v.shape)
print(w.shape)




v=v.reshape((1,2))
w=w.reshape((1,2))




np.dot(v,w)



print('matrix v : ',v)
print('matrix v Transpose:',v.T)
print('matrix w:',w)
print('matrix w Transpose:',w.T)
print('~~~~~~~~~')
print(np.dot(v,w.T))
print('~~~~~~~~~')
print(np.dot(v.T,w))




print(x)
v=np.array([9,10])
print("~~~~~")
print(v)
x.dot(v)



print(x)
print("~~~")
print(y)
x.dot(y)




x = np.array([[1,2],[3,4]])
x



np.sum(x)




np.sum(x, axis=0)



np.sum(x, axis=1)



x



x.T




x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])




vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
vv



print(x)
print("~~~~~")
print(vv)
x + vv




x + v # produce the same result as x + vv




v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)



v.shape



w.shape



x = np.array([[1,2,3], [4,5,6]]) # x has shape (2,3])
x.shape



x



print(x)
print("~~~~~")
print(v)
x + v




v=np.array([1,2])



x+v



x.T



print(x)
print("~~~~~")
print(w)
x.T + w



(x.T + w).T




x + np.reshape(w, (2, 1))




import pandas as pd



get_ipython().magic('matplotlib inline')




cities=["Delhi","Mumbai","Kolkata","Chennai"]
code = [11,22,33,44]

mydata=list(zip(cities,code))




mydata




df = pd.DataFrame(data=mydata,columns=["cities","codes"])
df




df.to_csv("mydata.csv",index=False,header=False)




writer=pd.ExcelWriter("mydata.xlsx")

df.to_excel(writer,"Sheet1",index=False)
df.to_excel(writer,"Sheet2")




myfile='~/Dropbox/March onwards/Python Data Science/Data/bank-full.csv'
bd=pd.read_csv(myfile,sep=";")



bd.head()




bd.dtypes




bd["month"].dtype




bd.head()




bd.describe()




bd.median()




bd["age"].describe()



bd[["age","previous"]].describe()






bd["job"].value_counts()




bd_cat_data=bd.select_dtypes(['object'])




bd_cat_data.columns



for c in bd_cat_data.columns:
    print(bd[c].value_counts())



for c in bd_cat_data.columns:
    print(c,":",bd[c].nunique())




pd.crosstab(bd["y"],bd["job"])



pd.crosstab(bd["y"],bd["job"],margins=True)




bd["age"].mean()



bd.groupby('job')["age"].mean()




bd.groupby(['housing','loan'])["age","balance"].mean()




bd.groupby(['housing','loan']).agg({'age':'mean','duration':'max','balance':'sum'})




bd["balance"].plot()




from ggplot import *
ggplot(aes(x="age",y="balance"),data=bd)




ggplot(aes(x="age",y="balance"),data=bd)+geom_point()+ggtitle("ABC")




ggplot(aes(x="age",y="balance",color="marital"),data=bd)+geom_point()+ggtitle("Scatter Plot")




meat.head()



ggplot(aes(x='date', y='beef'), data=meat) +    geom_line() +    geom_point() +    stat_smooth(color="blue",method="loess",span=0.2)




ggplot(aes(y="marital",x="balance"),data=bd)+geom_boxplot()




ggplot(bd,aes(x="balance"))+geom_histogram()+ggtitle("Histogram")



ggplot(bd,aes(x="balance"))+geom_density()+ggtitle("Density Curve")




ggplot(bd,aes(x="marital"))+geom_bar()+ggtitle("Bar Plot for Categorical Variables")





ggplot(bd,aes(x="marital",fill="housing"))+geom_bar()+ggtitle("Bar Plot for Categorical Variables")




cities=["Delhi","Mumbai","Kolkata","Chennai"]
code = ["11","22","33","4a"]

mydata=list(zip(cities,code))
df = pd.DataFrame(data=mydata,columns=["cities","codes"])

df



df.dtypes




df["codes"]=pd.to_numeric(df["codes"],errors="coerce")



df




df["cities2"]=[x.replace("a","6") for x in df["cities"]]
df



df["code_log"]=[math.log(x) for x in df["codes"]]
df



df["new"]=df.codes+df.code_log
df["new2"]=df.new+2
df



df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
df



df['color'] = np.where(df['Set']=='Z', 'green', 'red')
df




df['abc'] = np.where(df['Set']=='Z', df['Type'], df['Set'])
df





df



df=df.drop("abc",axis=1) 




df



df.drop("color",axis=1,inplace=True)



df



df.columns[0]



df=df.drop(df.columns[0],axis=1)



df




df=df.drop([3],axis=0)



df




df=df[df["Type"]=="B"]



df




df.index




df['col2']=[3,4]
df.reset_index(drop=True)



df.iloc[0]



print(df.iloc[0,1])
print(df.iloc[0]['Type'])





df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                       'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']}
                        )



df2 = pd.DataFrame({'A': ['A4', 'A1', 'A2', 'A3'],
                       'B': ['B4', 'B1', 'B2', 'B3'],
                        'C': ['C4', 'C1', 'C2', 'C3'],
                        'D': ['D4', 'D1', 'D2', 'D3']}
                        )



df1



df2




newdata_long=pd.concat([df1,df2],axis=0)



newdata_long



newdata_long.reset_index(drop=True)
newdata_long



df3 = pd.DataFrame({'E': ['A4', 'A1', 'A2', 'A3',"ab"],
                       'F': ['B4', 'B1', 'B2', 'B3',"ab"],
                        'G': ['C4', 'C1', 'C2', 'C3',"ab"],
                        'H': ['D4', 'D1', 'D2', 'D3',"ab"]}
                        )
df3



newdata_wide=pd.concat([df1,df3],axis=1)
newdata_wide




df1=pd.DataFrame({"custid":[1,2,3,4,5],
                 "product":["Radio","Radio","Fridge","Fridge","Phone"]})
df2=pd.DataFrame({"custid":[3,4,5,6,7],
                 "state":["UP","UP","UP","MH","MH"]})



df1



df2



inner=pd.merge(df1,df2,on=["custid"])
inner



outer=pd.merge(df1,df2,on=["custid"],how='outer')
outer



left=pd.merge(df1,df2,on=["custid"],how='left')
left



right=pd.merge(df1,df2,on=["custid"],how='right')
right


