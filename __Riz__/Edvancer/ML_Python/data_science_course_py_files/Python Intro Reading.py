




x=5
y="data science"





x



y




X






isthisareadablename=3
_or_this_is_more_readable_despite_being_longer=4




from=4




type(x)



type(y)




x="sachin"
x




x="5"
type(x)



int(x)




x="some char"
type(x)



int(x)




x="42"
int(x)



print(x)
type(x)



x=int(x)
type(x)





x=1.22
y=20
x+y



x-y



x*y



x/y




x+2,y+3,x+y,x**(x+y)





x=2
y=3
x**y





z=(x+y)**x - y/x
z




a,b,c,d=x+2,y+3,x+y,x**(x+y)



a,b,c,d




a,b,c=2




a,b,c=2,2,2
a,b,c




a,b,c=[2,2,2]
a,b,c




a,b,c=[2]*3
a,b,c




import math
math.log(x)




import math as m
m.sin(x)




x=True
y=False



type(x)



type(y)




x and y



x & y




x or y



x | y



not x





x = 'Bombay'
y = "Banglore"
x,y




len(x),len(y)




x+y



x+" "+"#*$^"+y



z=x+" to "+y
z




z.upper()



z.lower()




z.capitalize()




z.rjust(20)




z.ljust(20)




z.center(20)




z="   "+z+"    "
z



z.lstrip()



z.rstrip()



z.strip()



z




print(z)
z.replace("B","$#@")



print(z)
z=z.strip()
z



print(z)
k=z.split("a")
print(k)
print(type(k))





print(z)
z[2:4]



print(z)
z[3:]



print(z)
z[:3]



print(z)
z[:-3]



print(z)
z[-3:]



print(z)
z[3:-3]



print(z)
z[-6:-2]




print(z)
z.count("B")





print(z)
z.find("B")





x=34
y=32

x==y



x!=y



x>y



x<y



x>=y



x<=y



x="as"
y="assignment"

x in y



x not in y




x > y



x="B"
y="Aaajsa"
x>y





x = [1,2,3,70,-10,0,99,"a","b","c"]
x




print(x)
x[3:7]



print(x)
x[3:]



print(x)
x[:3]



print(x)
x[3:-3]



print(x)
x[3:-3:2] # third argument here is step



print(x)
x[::2]




print(x)

x[2]



x[2]="!!!"
x




print(x)
x[2:4]="bipin" 
print(x)




x=[23,45,67,2,5]

x[2:4]



print(x)
x[2:4]=["bipin"]
print(x)




print(x)
x.append("new")
print(x)




print(x)
x.append([1,2,3])
print(x)



x[5]




print(x)
x.extend([1,2,3])
print(x)




print(x)
x.insert(2,"another")
print(x)




print(x)
x=x+[3,4,5]
print(x)




print(x)
x.pop()

print(x)




print(x)
x.pop(3)
print(x)




print(x)
x.remove("another")
print(x)




x.remove("another")





x=[(a-5)**2 for a in range(10)]
print(x)
x.sort() # sorts in place
print(x)
x.reverse() # reverses in place
print(x)





cities = ["delhi","Banglore","mumbai","pune"]




print(len(cities[0]))
print(len(cities[1]))
print(len(cities[2]))
print(len(cities[3]))




for i in [0,1,2,3]:
    print(len(cities[i]))




cities



for city in cities:
    print(len(city))





x=range(8)




100%32



x=range(8)

x_mod_2=[]

for a in x:
	x_mod_2.append(a%2)

x_mod_2




x_mod_2=[]
x_mod_2=[a%2 for a in x]
x_mod_2




print(x)



x_ls2=[math.log(a) for a in x if a%2 !=0]
print(x_ls2)





def my_func(x):
    if x%2!=0:
        return(math.log(x))
    else:
        return(x)

x_ls3=[my_func(a) for a in x]
x_ls3




def mysum(x=10,y=10):
	return(2*x+3*y)

print(mysum(2,3))
print(mysum(2))
print(mysum())



mysum(y=4)




print(mysum(y=-10))
print(mysum(y=100,x=1))





d= {"actor":"nasir","animal":"dog","earth":1,"list":[1,2,3]}




d["animal"]




d['fish']



d['module']="intro"
print(d)



del d['actor']
print(d)



for elem in d:
	print('value for key:%s is' % (elem) ,":",d[elem])




d.items()



for a,b in d.items():
	print('value for key:%s is %s' %(a,b))




animals = {'cat', 'dog'}




'cat' in animals



'fish' in animals



animals.add('fish')
print(animals)




animals.add('cat')   
print(animals)




animals.add('cow')
print(animals)



animals.remove('cat')
print(animals)




animals = {'cat', 'dog', 'fish','here','there'}
for animal in animals:
	print(animal)




a={1,2,3,4,5,6}
b={4,5,6,7,8,9}



c=a.union(b)
c



a.intersection(b)



a.issubset(b)



a.issubset(c)



c.issuperset(b)




print(a)
print(b)
a.difference(b)



print(a)
print(b)
a.symmetric_difference(b)




t = 12345, 54321, 'hello!'
t



t =( 12345, 54321, 'hello!') # same thing
t




u = t, (1, 2, 3, 4, 5)
u




t



t[0]=21 # you can not reassign




v = ([1, 2, 3], [3, 2, 1])



v[0]=21



print(v)
v[0][0]=21
print(v)




d = {(x, x + 1): x for x in range(10)}  
print(d)



t = (5, 6)   
print(type(t))
print(d[t])





class Point(object):

    def __init__(self, x, y):
        '''Defines x and y variables'''
        self.X = x
        self.Y = y

        # self is used as reference to internal objects that get 
        # created using values supplied from outside 
        # default function _init_ is used to create internal objects
        # which can be accessed by other functions in the class

    def length(self):
        return(self.X**2+self.Y**2)

    # all function inside a class will have access to internal objects 
    # created inside the class
    # you can understand self to be default Point class object

    def distance(self, other):
        dx = self.X - other.X
        dy = self.Y - other.Y
        return math.sqrt(dx**2 + dy**2)

    # a function inside the class can take input multiple objects
    # of the same class




z=Point(2,3)
y=Point(4,10)



type(y)



type(z)




print(z.X)
print(z.Y)




print(Point.distance(y,z))
print(Point.length(y))
print(Point.length(z))



