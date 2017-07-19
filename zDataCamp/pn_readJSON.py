import json
# import urllib2
import requests
url_oData_people = "http://services.odata.org/TripPinRESTierService/(S(pk4yy1pao5a2nngmm2ecx0hy))/People"


response = requests.get( url_oData_people )
people   = response.json()
# people = json.load(urllib2.urlopen(url_oData_people))
print(people)

# CONVERT JOSN into object -> Pandas or dictionary array.7


