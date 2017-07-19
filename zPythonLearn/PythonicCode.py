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
def lambdaa():



    return 



if __name__ == "__main__":
    # strings()
    ign()
    #min 55