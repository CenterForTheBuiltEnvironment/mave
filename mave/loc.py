import urllib2
import numpy as np
import datetime
import time
import dateutil.parser as dparser
import json
import pdb
from zipfile import ZipFile
import urllib
from StringIO import StringIO

class Locate(object):
    def __init__(self, zipcode='94720', key='d3dffb3b59309a05', **kwargs):
        self.lat, self.lon, self.geocode = self.geolookup(zipcode,key)
        self.epw = self.getepw(self.lat,self.lon)
    
    def geolookup(self, zipcode, key):
        url = 'http://api.wunderground.com/api/%s/geolookup/q/%s.json'\
                                                       %(key, zipcode)
        try:
            f = urllib2.urlopen(url)
        except IOError:
            time.sleep(30)
            try:
                f = urllib2.urlopen(url)
            except:
                raise "cannot access wunderground api"
        geoinfo = json.loads(f.read())
        lat = float(geoinfo['location']['lat'])
        lon = float(geoinfo['location']['lon'])
        geocode = geoinfo['location']['nearby_weather_stations']['airport']\
                         ['station'][0]['icao']
        return lat, lon, geocode

    def getepw(self,lat,lon):
        f = open('./mave/data/epwurl.csv','r')
        csv = np.genfromtxt(f, delimiter=',', dtype=None)
        csv_lat = csv[1:,4].astype(float)
        csv_lon = csv[1:,5].astype(float)
        min_idx = np.argmin(np.sqrt([(csv_lat-lat)**2+(csv_lon-lon)**2]))+1
        un_zip_file = ZipFile(StringIO(urllib2.urlopen(csv[min_idx,6]).read()))
        epw_file = [i for i in un_zip_file.namelist() if '.epw' in i][0]
        epw = np.genfromtxt(un_zip_file.open(epw_file), delimiter=',',\
                                             dtype=None, skip_header = 8)
        return epw

if __name__ == "__main__":
    zipcode = '94720'
    key = 'd3dffb3b59309a05'
    test = Locate(zipcode,key)
    print test.lat
    print test.lon
    print test.epw
    print test.geocode
