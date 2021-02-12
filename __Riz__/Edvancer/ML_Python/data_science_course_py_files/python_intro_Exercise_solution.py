



x="xyz" *10
x




x=[]
for j in [1,2,3]:
    x.append([(i+1)**j for i in (range(7))])
x




input_list=[20, 2, 6, 7, 10]
mn=min(input_list)
mx=max(input_list)



input_list=[20,2,1,4,5,-5]

def my_func1(ml):
    mn=min(input_list)
    mx=max(input_list)
    new_list=[mn]
    if(mn==mx):
            dummy_list=[0]*10
            return(dummy_list)

    else:
            s=(mx-mn)/9
            x=mn
            while(x<mx):
                x=x+s
                new_list.append(round(x,2))
    return(new_list)
my_func1(input_list)



import numpy as np



def my_func2(m1):
    mn=min(input_list)
    mx=max(input_list)
    if(mn==mx):
        dummy_list=[0]*10
        return(dummy_list)
    else:
        return(np.linspace(mn,mx,10))



my_func2(input_list)




import operator
d= {"actor":"nasir","animal":"dog","earth":1,"list":[1,2,3]}
sorted_d_bykey = sorted(d.items(), key=operator.itemgetter(0))#sorting by key
print(sorted_d_bykey)




import operator
d= {"actor":"nasir","animal":"dog","earth":1,"list":[1,2,3]}
sorted_d_byvalues = sorted(d.items(), key=operator.itemgetter(1))#sorting by value
print(sorted_d_byvalues)




l= ['nasir','dog',1,[1,2,3]]
l.sort()
print(l)





import operator
d= {"actor":"nasir","animal":"dog","earth":"life","list":"not available"}
sorted_d_byvalues = sorted(d.items(), key=operator.itemgetter(1))
print(sorted_d_byvalues)




a={"qw":2}
b={"gf":45}
c=a+b



d_1={'india':'New delhi','Bangladesh':'dhaka','Pakistan':'Islamabad'}
d_2={'Malaysia':'Kuala Lumpur','india':34,'South Africa':'Cape Town','Country':'Capital'}
d_3={1:'one',2:'two','float':1.666667,'india':"last"}

new_d={**d_1,**d_2,**d_3}
print(new_d) # ** operator is used to unpack dictionaries






{**d_1,**d_2}




x={}
x.update(d_1)
x.update(d_2)
x.update(d_3)
print(x) 





x = {'a': 1, 'b': 2}
y = {'b': 3, 'c': 4} # here we have key 'b' both in x and y
x.update(y)
print(x)# precedence goes to key value pair in latter dictionary



def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

x = {'a': 1, 'b': 2}
y = {'b': 3, 'c': 4}
z= {"c":4,'india':56}
new_dict=merge_dicts(x,y,z)
print(new_dict)




x='string'
x[::-1] #[begin:end:step]




x='string'
x=list(x)
x



x.reverse()
x



"".join(x)





address={"home" : ["Hyderabad","Lingampally","Ph:1234567890"],
"office":["Maharashtra","Mumbai","Ghatkopar","Ph : 5432167809","Pin :400043"],
"OOI" : ["Singapore","Ph : 09876345"]}
address.items()



address={"home" : ["Hyderabad","Lingampally","Ph:1234567890"],
"office":["Maharashtra","Mumbai","Ghatkopar","Ph : 5432167809","Pin :400043"],
"OOI" : ["Singapore","Ph : 09876345"]}

phones={}

for k,v in address.items():
    temp=[x for x in v if "Ph" in x]
    phones[k]=temp

phones





def my_gcd(x,y):
    while(x%y !=0):
        (x,y)=(y,x%y)
    return(y)

my_gcd(33,77)




def sum_n_power_q(n,q):
    my_list=[i**q for i in range(n+1)]
    print(my_list)
    sum_list=sum(my_list)
    return(sum_list)

sum_n_power_q(5,2)




x=set([3*x for x in range(1,34)])
print(x)
y=set([5*x for x in range(1,20)])
print(y)



print(x.symmetric_difference(y))




def fb(n):
    if(n==2):
        return([1,1])
    else:
        k=fb(n-1)
        series=k+[k[n-2]+k[n-3]]
        return(series)

fb(10)






