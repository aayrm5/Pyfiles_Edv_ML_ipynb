




from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream 



twitter_info="riya_twit_info.txt"
f=open(twitter_info,"r")
mykeys=[]

for i,line in enumerate(f):
    mykeys.append(line.split(':')[1].strip())
print(mykeys)



CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRET=mykeys




oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)



twitter_stream = TwitterStream(auth=oauth)



import json
twitter_stream = TwitterStream(auth=oauth)

iterator = twitter_stream.statuses.filter(track="india", language="en")
tweet_count = 10
for tweet in iterator:
    tweet_count -= 1

    j=json.dumps(tweet)
    j1=json.loads(j)#json.loads() is used to process json files
    print(j1["user"]["name"] +":"+ j1["text"].split(":")[0])

    if tweet_count <= 0:
        break 




import urllib.request
from urllib.request import urlretrieve


syntax is: urllib.request.urlretrieve(url,"file.extension")

urllib.request.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz","C:/Users/Riya/Desktop/data.tar.gz" )






from facebook import *
import facebook




graph = facebook.GraphAPI(access_token='EAAQb0kyF82kBAGQrFHsAuGbvUJ2MlQ6rIYhpVFWIV9RZBRuFutgcM5BCq92qEm5mtvLnCNgS2gTn9RP69S4X2MdwQHkMQmQiFrNAgCGS3XIdexFhPiTe07bvoFoWknbXivgzZCnkNshIFYbDiV85SsniYKe7kJTlzO4CYPRgZDZD', version='2.7')



profile = graph.get_object("me",fields='id,name,about,age_range,birthday,context,cover,education,email')
print(profile['name'])
print(profile['birthday'])
print(profile['education'])
print(profile['email'])#extracting your information



friends = graph.get_connections("me", "friends")
print(friends)#number of friends you have



post = graph.get_object(id='1332799566744721_1337319076292770')
print(post['message'])#if post id is available, the message can be extracted



comments = graph.get_connections(id='1332799566744721_1337319076292770', connection_name='comments')
print(comments)



from facepy import GraphAPI #you can also use facepy to scrap facebook data



graph = GraphAPI('EAAQb0kyF82kBAGQrFHsAuGbvUJ2MlQ6rIYhpVFWIV9RZBRuFutgcM5BCq92qEm5mtvLnCNgS2gTn9RP69S4X2MdwQHkMQmQiFrNAgCGS3XIdexFhPiTe07bvoFoWknbXivgzZCnkNshIFYbDiV85SsniYKe7kJTlzO4CYPRgZDZD')

graph.get("me?fields=id,name,age_range,birthday,education,email")# get your profile information



graph.get('me/friends')#number of friends connection you have




graph.get('me/posts')#search for top posted created by you



graph.search(term='diwali', type='event', page=False)#search for the term diwali from the events




from urllib.request import urlopen



data= "C:/Users/Riya/Desktop/PORTFOLIO.csv"
my_portfolio=pd.read_csv(data)#reads the csv file



symbol_list=list(my_portfolio["Stock Symbol"])
symbol_list



pricelist=[]
for i in symbol_list:
    url='https://in.finance.yahoo.com/q/hp?s='+i
    page=urlopen(url)
    content=page.read()
    content=str(content).split(i.lower())[1]
    price=float(content.split(">")[1].split("<")[0])#reads latest prices of the symbols from yahoo finance
    pricelist.append(price)
my_portfolio["current_price"]=pricelist 
profit=(my_portfolio["current_price"]-my_portfolio["Purchase Price"])*(my_portfolio["Number of shares"])
total_profit=profit.sum()
print("total profit:" ,total_profit)
my_portfolio["%gain"]=((my_portfolio["current_price"]-my_portfolio["Purchase Price"])/my_portfolio["Purchase Price"])*100
my_portfolio["performance_rank"]=my_portfolio["%gain"].rank(ascending=False)



my_portfolio.head()



my_portfolio.to_csv("C:/Users/Riya/Desktop/My_Portfolio.csv")





