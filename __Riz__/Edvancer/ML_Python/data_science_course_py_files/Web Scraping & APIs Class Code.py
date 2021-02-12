



f=open("myfile.txt","w")





type(f)



f.write("number"+"@"+"square"+"\n")




for i in range(8):
    mytext=str(i)+'@'+str(i**2)+'\n'
    f.write(mytext)



f.close()



f=open("myfile1.txt","w")
for i in range(8):
    mytext='square of number '+str(i)+' is :'+str(i**2)+'\n'
    f.write(mytext)
f.close()




f=open("myfile1.txt","r")

for line in f:
    print(line)

f.close()



f=open("myfile1.txt","r")
numbers=[]
squares=[]

for line in f:

    number=line.split("number")[1].split("is")[0]
    numbers.append(number)
    square=line.split(":")[1]
    squares.append(square)
print(numbers)
print(squares)
f.close()



f=open("myfile1.txt","r")
numbers=[]
squares=[]
for line in f:
    line=line.strip()
    number=int(line.split("number")[1].split("is")[0])
    numbers.append(number)
    square=int(line.split(":")[1])
    squares.append(square)
print(numbers)
print(squares)
f.close()







from urllib.request import urlopen
x = urlopen('https://www.reddit.com/')
k=x.read()



k



parts=str(k).split("<a href=")



for part in parts[1:4]:
    print(part.split('"')[1])



for part in parts:
    if "</a" not in part:pass
    else:
        link=part.split('"')[1]
        print(link)



parts=str(k).split("<a href=")
for part in parts:
    link=part.split('"')[1]
    if "domain" not in link:pass
    else:
        link=link.split("domain")[1].split('/')[1]
        print(link)



parts=str(k).split("<a href=")
links=[]
for part in parts:
    link=part.split('"')[1]
    if "domain" not in link:pass
    else:
        link=link.split("domain")[1].split('/')[1]
        if link in links:pass
        else:
            print(link)
            links.append(link)






url='https://www.google.com/search?q=python+programming+tutorials'
try:
    resp = urlopen(url)
    respData = resp.read()
    print(respData)
except Exception as e:
    print(str(e))





from urllib.request import Request
url = 'https://www.google.com/search?q=python+programming+tutorials'

headers = {}
headers['User-Agent'] = r"Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"

req = Request(url, headers = headers)
resp = urlopen(req)
respData = resp.read()



print(respData)





from bs4 import BeautifulSoup



mydata=urlopen("https://in.finance.yahoo.com/q?s=SBIN.BO").read()
soup = BeautifulSoup(mydata)



soup.findAll("a")



all_links=soup.findAll("a")
for link in all_links:
    print(link.get('href'))



soup.findAll('table')[1]



mytable=soup.findAll('table')[1]
mytable.findAll('tr')



for row in mytable.findAll('tr'):
    name=row.findAll('th')[0].string
    value=row.findAll('td')[0].string
    print(name,value)





from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream



twitter_info_file="twitter_acc_info.txt"
f=open(twitter_info_file,"r")
mykeys=[]

for i,line in enumerate(f):
    mykeys.append(line.split(':')[1].strip())
print(mykeys)



CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRET=mykeys




oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter= Twitter(auth=oauth)



twitter.search.tweets(q='#BanegaSwachhIndia', result_type='recent', lang='en',
                      count=3)




twitter_stream = TwitterStream(auth=oauth)



import json
iterator = twitter_stream.statuses.sample()

tweet_count = 10
for tweet in iterator:
    tweet_count -= 1

    print(json.dumps(tweet) )

    if tweet_count <= 0:
        break 




iterator = twitter_stream.statuses.filter(track="Google", language="en")
tweet_count = 3
for tweet in iterator:
    tweet_count -= 1

    print(json.dumps(tweet) )

    if tweet_count <= 0:
        break 



twitter_userstream = TwitterStream(auth=oauth, domain='userstream.twitter.com')

iterator = twitter_userstream.statuses.filter(track="Google", language="en")
tweet_count = 3
for tweet in iterator:
    tweet_count -= 1

    print(json.dumps(tweet) )

    if tweet_count <= 0:
        break 





world_trends = twitter.trends.available(_woeid=1)
world_trends[1:3]



ott_trend=twitter.trends.place(_id = 3369)



print(json.dumps(ott_trend, indent=4))




twitter.followers.ids(screen_name="nikunj_lata")



twitter.statuses.user_timeline(screen_name="lalitsachan")






