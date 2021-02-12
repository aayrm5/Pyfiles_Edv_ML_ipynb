



import pandas as pd
import numpy as np



import warnings
warnings.filterwarnings('ignore')



data_file='~/Dropbox/March onwards/Python Data Science/Data/SMSSpamCollection.txt'



f=open(data_file,"r")

target=[]
sms=[]

for line in f:
    line=line.strip()

    if line=="":continue

    if line[0:4]=="spam":
        sms.append(line.split('spam')[1].strip())
        target.append("spam")

    if line[0:3]=="ham":
        sms.append(line.split('ham')[1].strip())
        target.append("ham")

f.close()

mydata=pd.DataFrame(list(zip(target,sms)),columns=['target','sms'])



mydata.head()



mydata['target'].value_counts()



all_sms=" ".join(mydata['sms'])
ham_sms=" ".join(mydata.loc[mydata['target']=="ham","sms"])
spam_sms=" ".join(mydata.loc[mydata['target']=="spam","sms"])



from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



STOPWORDS



wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', 
                      width=4000,height=2000).generate(all_sms)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()



wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', 
                      width=4000,height=2000).generate(ham_sms)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()



wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', 
                      width=4000,height=2000).generate(spam_sms)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()





mydata['length']=[len(x) for x in mydata['sms']]



mydata.head()



from ggplot import *



ggplot(mydata,aes(x='length',color='target'))+geom_density()






for word in ['sale','call','will','ur','gt','now','ok']:
    print(word)
    mydata[word]=0
    for i in range(len(mydata.index)):
        if word in mydata['sms'][i].lower():
            mydata.loc[i,word]=1



mydata.head()



mydata['call'].value_counts()



from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import tree



mydata_train,mydata_test =train_test_split(mydata,test_size=0.2,random_state=2)

x_train=mydata_train.drop(['sms','target'],1)
y_train=mydata_train['target']


x_test=mydata_test.drop(['sms','target'],1)
y_test=mydata_test['target']

clf=tree.DecisionTreeClassifier()

clf.fit(x_train,y_train)

predictions=pd.DataFrame(list(zip(y_test,clf.predict(x_test))),columns=['real','predicted'])

pd.crosstab(predictions['real'],predictions['predicted'])



from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(class_weight='balanced',verbose=1,n_estimators=100)



clf.fit(x_train,y_train)

predictions=pd.DataFrame(list(zip(y_test,clf.predict(x_test))),columns=['real','predicted'])

pd.crosstab(predictions['real'],predictions['predicted'])

















from sklearn.tree import DecisionTreeClassifier



clf=DecisionTreeClassifier()



clf.fit(x_train,y_train)

predictions=pd.DataFrame(list(zip(y_test,clf.predict(x_test))),columns=['real','predicted'])

pd.crosstab(predictions['real'],predictions['predicted'])




from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop



def split_into_lemmas(message):
    message=message.lower()
    words = TextBlob(message).words
    words_sans_stop=[]
    for word in words :
        if word in stop:continue
        words_sans_stop.append(word)
    return [word.lemma for word in words_sans_stop]



tfidf= TfidfVectorizer(analyzer=split_into_lemmas)



tfidf.fit(mydata_train['sms'])



train_tfidf=tfidf.transform(mydata_train['sms'])



clf= MultinomialNB()
clf.fit(train_tfidf, y_train)



test_tfidf=tfidf.transform(mydata_test['sms'])



test_tfidf




predictions=pd.DataFrame(list(zip(y_test,clf.predict(test_tfidf))),columns=['real','predicted'])

pd.crosstab(predictions['real'],predictions['predicted'])




1. Try NB on non-text data
2. Try tfidf variables with other algorithm 




def print_top10(vectorizer, clf):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    indices=np.argsort(clf.coef_)[0][-10:]
    for i in range(10):
        print(feature_names[indices[i]])



print_top10(tfidf,clf)







