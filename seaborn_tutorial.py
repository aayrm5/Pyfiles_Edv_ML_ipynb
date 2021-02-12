



import pandas as pd
import math
import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



import seaborn as sns
sns.set(color_codes=True)



data_file='~/Dropbox/March onwards/Python Data Science/Data/Existing Base.csv'

eb=pd.read_csv(data_file)



eb=eb.head(100) # for the sake of simplicity i have taken only first 100 observations, 
                #you can go ahead and can consider the whole dataset




eb_port=eb["Portfolio Balance"]



sns.distplot(eb_port,kde=False) # I have taken kde=False just to obtain histogram



sns.distplot(eb_port, kde=False, rug=True) #rug plot will draw a small vertical tick at each observation.



sns.distplot(eb_port,kde=False,norm_hist=True) #density on y axis rather than count



sns.distplot(eb_port,bins=20, kde=False, rug=True)#binning, number of bins can be chosen explictly



sns.distplot(eb_port, hist=False)#density plot



sns.distplot(eb_port) #both density and histogram plots




sns.kdeplot(eb["Portfolio Balance"], shade=True)
sns.kdeplot(eb["Investment in Commudity"], shade=True);





sns.jointplot(x="Investment in Commudity", y="Portfolio Balance", data=eb)




sns.jointplot(x="Investment in Commudity", y="Portfolio Balance", kind="hex",size=5,color="g",data=eb)




sns.jointplot(x="Investment in Commudity", y="Portfolio Balance", kind="kde",size=5,data=eb)



s=sns.jointplot(x="Investment in Commudity", y="Portfolio Balance", kind="kde",size=5,data=eb)
s.plot_joint(plt.scatter, c="r", s=30, linewidth=0.5, marker="*")
s.ax_joint.collections[0].set_alpha(0);




sns.lmplot('Investment in Commudity', 'Portfolio Balance', 
           data=eb,palette="Set1",
           fit_reg=False)




sns.lmplot('Investment in Commudity', 'Portfolio Balance', 
           data=eb,palette="Set1")



sns.lmplot('Investment in Commudity', 'Portfolio Balance', 
           data=eb, 
           order=3)




sns.lmplot('Investment in Commudity', 'Portfolio Balance', hue="gender",data=eb,fit_reg=False)
plt.title("myplot", fontname='Ubuntu', fontsize=14,
            fontstyle='italic', fontweight='bold')






sns.lmplot('Investment in Commudity', 'Portfolio Balance', hue="gender",col="status",data=eb,fit_reg=False,size=3)



sns.lmplot('Investment in Commudity', 'Portfolio Balance', hue="gender",col="status",row="self_employed",
            data=eb,fit_reg=False,size=4)





sns.stripplot(x='status', y="Portfolio Balance", data=eb,jitter=True)




sns.swarmplot(x='status', y="Portfolio Balance", data=eb)




sns.swarmplot(x='status', y="Portfolio Balance",hue="gender", data=eb)




sns.boxplot(x='status', y="Portfolio Balance", data=eb)




sns.violinplot(y="Portfolio Balance", x="children", data=eb)




sns.violinplot(x="Portfolio Balance", y="children",hue="status", data=eb)




sns.countplot(x="age_band", data=eb,palette="Greens_d")



sns.countplot(x="age_band", data=eb,hue="children",palette="Greens_d")




sns.barplot(x="status", y="Investment in Commudity",hue="children",data=eb)



sns.barplot(x="Investment in Commudity", y="status",hue="gender",data=eb)




sns.set()
flights = sns.load_dataset("flights")
flights.dropna().head()



flights_long = flights.pivot("month", "year", "passengers")
flights_long




sns.heatmap(flights_long, annot=True, fmt="d", linewidths=.5)





