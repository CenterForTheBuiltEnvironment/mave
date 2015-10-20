import json
import urllib2
import numpy as np
import pdb
from zipfile import ZipFile
from StringIO import StringIO
import os
import pkgutil
from datetime import datetime

class location(object):
    def __init__(self, address, **kwargs):
        self.lat, self.lon = self.get_latlon(address)
        self.geocode = self.get_geocode(self.lat, self.lon)

    def get_latlon(self,address):
        g_key = 'AIzaSyBHvrK5BitVyEzcTI72lObBUnqUR9L6O_E'
        address = address.replace(' ','+')
        url = \
          'https://maps.googleapis.com/maps/api/geocode/json?address=%s&key=%s'\
                                                               %(address,g_key)
        f = json.loads(urllib2.urlopen(url).read())
        lat = f['results'][0]['geometry']['location']['lat']
        lon = f['results'][0]['geometry']['location']['lng']
        return lat, lon
       
    def get_geocode(self, lat, lon):
        w_key = 'd3dffb3b59309a05'
        url = 'http://api.wunderground.com/api/%s/geolookup/q/%s,%s.json'\
                                                       %(w_key,lat,lon)
        try:
            f = urllib2.urlopen(url)
        except:
            time.sleep(30)
            try:
                f = urllib2.urlopen(url)
            except e:
                raise e
        geoinfo = json.loads(f.read())
        geocode = geoinfo['location']['nearby_weather_stations']\
                             ['airport']['station'][0]['icao']
        if geocode == '':
            geocode = geoinfo['location']['nearby_weather_stations']['airport']\
                             ['station'][1]['icao']
        else:
            geocode = geocode
        return geocode

class TMYData(object):
    def __init__(self, address, **kwargs):
        lat_lon = location(address)
        self.lat = lat_lon.lat
        self.lon = lat_lon.lon
        self.tmy = self.getTMY(self.lat,self.lon)

    def getTMY(self,lat,lon):
        f = StringIO(pkgutil.get_data('mave', 'data/epwurl.csv'))
        csv = np.genfromtxt(f, delimiter=',', dtype=None)
        csv_lat = csv[1:,4].astype(float)
        csv_lon = csv[1:,5].astype(float)
        min_idx = np.argmin(np.sqrt([(csv_lat-lat)**2+(csv_lon-lon)**2]))+1
        un_zip_file = ZipFile(StringIO(urllib2.urlopen(csv[min_idx,6]).read()))
        tmy_file = [i for i in un_zip_file.namelist() if '.epw' in i][0]
        outpath = os.getcwd()+'/mave/data/'
        un_zip_file.extract(tmy_file,outpath)
        names = ["year","month","day","hour","minute","datasource",\
                 "DryBulb","DewPoint","RelHum","Atmos_Pressure",\
                 "ExtHorzRad","ExtDirRad","HorzIRSky","GloHorzRad",\
                 "DirNormRad","DifHorzRad","GloHorzIllum","DirNormIllum",\
                 "DifHorzIllum","ZenLum","WindDir","WindSpd","TotSkyCvr",\
                 "OpaqSkyCvr","Visibility","Clg_Hgt","Weather_obs",\
                 "Weather_code","Precip","Aerosol_Opt_Dept","Snow_Dept",\
                 "Days_Since_Last_Snow","Albedo","Liquid_Precip_Dept",\
                 "Liquid_Precip_Q"]
        cols = list(filter(lambda x: x!="datasource",names))
        tmy = np.genfromtxt(un_zip_file.open(tmy_file), delimiter=',',\
                    dtype=None, skip_header=8, names=names, usecols=cols)
        np.place(tmy['year'],tmy['year']!=datetime.now().year,\
                 datetime.now().year)
        column_names = ','.join(cols)
        np.savetxt('./mave/data/clean_tmy.csv',\
                   tmy,delimiter=',',fmt='%s',\
                   header=column_names, comments='')
        return tmy

if __name__ == "__main__":
    address = 'wurster hall, uc berkeley'
    test = location(address)
    test2 = TMYData(address)
    print 'address:',address
    print 'lat:',test.lat,' lon:',test.lon
    print 'nearest_airport:',test.geocode
    print 'TMY Data:', test2.tmy
