# Data School - Write Pythonic Code for Better Data Science
# Topics: 
# 1. string formatting
# 2. merge dicts
# 3. tupple assignement and unpacking
# 4. ignore return value - exceptions,  errors 
# 5. lambda expression 
# 6. yield and generators methods => pipelines and dataflow
# 7. inline generators expressions
# 8. dictionary for performance 
# 9. slots - memory search! 

# intro PEP8 - guide style python! - 4 spaces, class declarations... 

import collections
import datetime

name = 'David'
age  = 26
#---------------------------------------------------------------
# 1. STRINGS 
#---------------------------------------------------------------
def strings():
    # works, but not pythonic
    print("Hi, I'm " + name + " and I'm " + str(age) + " years old.")
    # probably pythonic
    print("Hi, I'm %s and I'm %d years old." % (name, age))
    # pythonic
    print("Hi, I'm {} and I'm {} years old.".format(name, age))
    print("Hi, I'm {1} years old and my name is {0}, yeah {1}.".format(name, age))
    data = {'day': 'Saturday', 'office': 'Home office', 'other': 'UNUSED'}
    # print: On Saturday I was working in my Home office! 
    print("On {day} I was working in my {office}!".format(**data)) #** take dict and convert it to keyword args. 
    print("Hi, I'm {name} and I'm {age} years old.".format(name=name, age=age))
    # In Python 3.6
    # print(f"Hi, I'm {name} and I'm {age} years old.")

#---------------------------------------------------------------
# 2. MERGE DICT 
#---------------------------------------------------------------
def merg_dict():
    # example web: /product/271/fast-apps?id=1^render_fast=true + form = email and name
    route = {'id': 271, 'title': 'Fast apps'}
    query = {'id': 1, 'render_fast': True}
    post = {'email': 'j@j.com', 'name': 'Jeff'}
    # print("Individual dictionaries: "); print("route: {}".format(route)); print("query: {}".format(query)); print("post:  {}".format(post))
    # Non-pythonic procedural way
    m1 = {}
    for k in query:
        m1[k] = query[k]
    for k in post:
        m1[k] = post[k]
    for k in route:
        m1[k] = route[k]
    # Classic pythonic way: copy and update
    m2 = query.copy()
    m2.update(post)
    m2.update(route)
    # Via dictionary comprehensions:
    m3 = {k: v for d in [query, post, route] for k, v in d.items()}
    # Python 3.5+ pythonic way, warning crashes on Python <= 3.4:
    m4 = {**query, **post, **route}
    print(m1); print(m2);  print(m3);print(m4)
    print("Are the same? " + 'yes' if m1 == m2 and m2 == m3 and m3 == m4 else 'no')
    #  m3 = {k: v for d in [query, post, route] for k, v in d.items()}
    #  m4 = {**query, **post, **route}

#---------------------------------------------------------------
# 3. TUPPLES 
# ---------------------------------------------------------------
def tupples():
    # tuples are defined as:
    t = (7, 11, "cat", [1, 1, 3, 5, 8]);     print(t)
    t = 7, 11, "cat", [1, 1, 3, 5, 8]
    # print(t)    # t = 7,
    # print(t, len(t))

    # create a tuple, grab a value.
    print(t[2])
    n, a, _ = t # underscore is I don't care about the value... you can use as much as you want
    #    print("n={}, a={}".format(n, a))
    # can also assign on a single line:
    x, y = 1, 2 #    print(x, y)
    # You'll find this often in loops (remember numerical for-in loops):
    for idx, item in enumerate(['hat', 'cat', 'mat', 'that']):
        print("{} -> {}".format(idx, item))

#---------------------------------------------------------------
# 4. ignoring val, exceptions 
# ---------------------------------------------------------------
class DownloadService:
    def check_download_url(self):
        pass

    def check_access_allowed(self):
        pass

    def check_network(self):
        pass

    def download_file(self):
        pass

def save_file(file_name, data):
    pass

# LBYL look before you leap
def ign():
    s = DownloadService()

    if not s.check_download_url():
        print("Cannot download, invalid url")
    if not s.check_access_allowed():
        print("Cannot download, permission denied")
    if not s.check_network():
        print("Cannot download, not connected")

    data = s.download_file()
    save_file('latest.png', data)
    print('Successfully downloaded latest.png')

    # EAFF Easier to ask for forgivenes than permission 
    try:
        data = s.download_file()
        save_file('latest.png', data)
        print('Successfully downloaded latest.png')
    except SocketError as se:
        print("Sorry no network: {}".format(se))
    except Exception as x:
        print("Sorry didn't work: {}".format(x))


#---------------------------------------------------------------
# 5. LAMBDA 
# ---------------------------------------------------------------
def filter_numbers(test):
    data = []
    for n in range (50):
        if test(n): 
            data.append(n)
    return data
def lambdaa():
    evens = filter_numbers(lambda x: x % 2 == 0)
    print(evens)
    data = [1,9,-1,20,5,-100]
    data.sort(key=lambda x: abs(x) )
    print(data)
#---------------------------------------------------------------
# 6. GENERATORS  
# ---------------------------------------------------------------
def fib():
    data = []
    nxt, current = 1, 0
    # 1 try
    # while nxt < 1000:
    #         data.append(nxt)
    #         nxt, current = nxt+current, nxt
    # return data
    # 2 - generator
    while True: 
        nxt, current = nxt+current, nxt
        yield nxt

def odds(seq):
    for n in seq: 
        if n % 2 == 1:
                yield n
# the best part of it is working with infinite data and only keeping one in memory!
# also working with lot of data and then loaded it all up and process it
def generator_com(): 
    # for f in fib():
    for f in odds(fib()):
        if f>1000: 
            break
        print(f, end=', ')
#---------------------------------------------------------------
# n9 slots -> Advance topic: PERFORMANCE - SLOTS! 
# ---------------------------------------------------------------
# - the longer the name of the variables, the more memory it will take.
# 
# slot based class -
ImmutableThingTupple = collections.namedtuple("ImmutableThingTupple", "a b c d")
class ImmutableThing:
    #JUST 1 LINE! 
    __slots__ = ['a', 'b', 'c', 'd'] # it frozen the structure of the class -> less memory
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b 
        self.c = c
        self.d = d
    
def check_memory(): 
    count = 1000 # New python: easy to visualize 1.000 or 1_000_000 1 million
    data = []
    print("tupple") # THE FASTER!                       time 1sec, memory 200Mb
    # data.append((1+n, 2+n, 3+n, 4+n)) 
    print("Named tupple") #ImmutableThingTupple class   TIME 3.5s memory 200Mb
    print("Regular class") #mutable class               TIME 3.5s memory 400Mb - double memory
    print("slot class") #immutable class                TIME 3s   memory 100Mb -> less memory!!!!!!!!!
    
    t0 = datetime.datetime.now()       
    for n in range(count):
        data.append(ImmutableThing(1+n, 2+n, 3+n, 4+n))
    t1 = datetime.datetime.now()       
    print(t1-t0)


if __name__ == "__main__":
    # strings()
    # ign()
    # lambdaa()
    # generator_com()
    check_memory()

# min 55