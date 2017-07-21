# notes... sometimes I need to configure a proxy to do external calls to ws. 
# for that: 


import json
# import urllib2
# people = json.load(urllib2.urlopen(url_oData_people))


import requests
url_oData_people = "http://services.odata.org/TripPinRESTierService/(S(pk4yy1pao5a2nngmm2ecx0hy))/People"

# response = requests.get( url_oData_people )
# people   = response.json()
# print(people)


# CONVERT JOSN into object -> Pandas or dictionary array.7

movie_json = """
{
"Title":"Johnny 5",
"Year":"2001",
"Runtime":"119 min",
"Country":"USA"
}
"""

movie_data = json.loads(movie_json)
print(type(movie_data), movie_data)
    
print("The title is {}".format(movie_data.get('Title')))
movie_json_text_2 = json.dumps(movie_data)
print(type(movie_json_text_2), movie_json_text_2)
