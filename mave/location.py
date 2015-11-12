"""
This software downloads TMY weather file and historical weather data for 
a given location within a given time frame. It interpolates the weather 
data to a given uniform interval.

@author Taoning Wang <taoning@berkeley.edu>
@author Paul Raftery <p.raftery@berkeley.edu>
"""

import json
import urllib2
import numpy as np
from zipfile import ZipFile
from StringIO import StringIO
import os
import pkgutil
import datetime, time 
import dateutil.parser as dparser 
import pdb 
import sys 


class location(object):
    def __init__(self, address, **kwargs):
        self.lat, self.lon, self.real_addrs = self.get_latlon(address)
        self.geocode = self.get_geocode(self.lat, self.lon)

    def get_latlon(self,address):
        g_key = 'AIzaSyBHvrK5BitVyEzcTI72lObBUnqUR9L6O_E'
        address = address.replace(' ','+')
        url = \
          'https://maps.googleapis.com/maps/api/geocode/json?address=%s&key=%s'\
                                                               %(address,g_key)
        f = json.loads(urllib2.urlopen(url).read())
        real_addrs = f['results'][0]['formatted_address']
        lat = f['results'][0]['geometry']['location']['lat']
        lon = f['results'][0]['geometry']['location']['lng']
        return lat, lon, real_addrs
       
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

class weather(object):
    def __init__(self,
                 start,
                 end,
                 key,
                 geocode,
                 interp_interval,
                 save,
                 **kwargs):
        self.start=start; self.end=end; self.interp_interval=interp_interval
        if start > end:
            error_msg =  "start time has to before the end time"
            sys.exit(error_msg)
        else:
            self.target_dts = np.arange(start, end,
                                       dtype='datetime64[%s]'%interp_interval)\
                                       .astype(datetime.datetime)
        interval = self.target_dts[-1]-self.target_dts[-2]
        self.target_dts = self.target_dts + (start-self.target_dts[0])
        if self.target_dts[-1] < end - interval:
            self.target_dts = np.append(self.target_dts, \
                                        self.target_dts[-1]+ interval)
        unix_vec = np.vectorize(self.str_to_unix_api)
        self.target_unix = unix_vec(self.target_dts)
        if key == None:
            self.timestamps, self.unix, self.data = \
                                              self.get_raw(start, end, geocode)
            self.interp_data = map(lambda x: np.interp(self.target_unix,
                                        self.unix,
                                        self.data[x].astype(float)), range(0,2))
        else:
            self.timestamps, self.unix, self.data = \
                                            self.get_raw_api(start,end,\
                                                             geocode,key)

            self.interp_data = map(lambda x: np.interp(self.target_unix,
                                                    self.unix,
                                                    self.data[:,x]), range(0,2))
        if save:
            out_time = np.vstack(self.target_dts).astype(str)
            out_data = np.column_stack(self.interp_data).astype(str)
            data_frame = np.column_stack([out_time,out_data])
            np.savetxt('weather.csv',data_frame,\
                       delimiter=',',header = self.headers,\
                       fmt ='%s',comments='')

    def get_raw(self, start, end, geocode):
        # define a range of dates
        end = end + datetime.timedelta(days=1)
        dates = np.arange(start.date(), end.date(), dtype='datetime64[D]')
        # download the timestamp data for that range of dates
        raw = np.asarray(map(lambda x: self.get_daily(geocode,x),dates))
        # stack each day of downloaded data
        data = map(lambda x: np.hstack(raw[:,x]), list([1,2]))
        timestamps = np.hstack(raw[:,0])
        # convert to unix time 
        vec_parse = np.vectorize(self.str_to_unix)
        unix = vec_parse(timestamps)
        return timestamps, unix, data

    def get_daily(self,geocode,date):
        date = date.astype(datetime.datetime)
        url = ('http://www.wunderground.com/history/airport'
              '/%s/%s/%s/%s/DailyHistory.html?format=1')%\
              (geocode,date.year,date.month,date.day)
        try:
            f = urllib2.urlopen(url)
        except:
            time.sleep(30)
            try:
                f = urllib2.urlopen(url)
            except:
                raise "operation stopped", date
        raw = f.read().splitlines()
        self.headers = 'time,tempF,dpF,RH'
        raw_txt = np.genfromtxt(raw,
                                delimiter=',',
                                names=self.headers,
                                usecols=('time,tempF,dpF'),
                                dtype=None,
                                skip_header=2)
        ts = raw_txt['time']
        time_series = np.ravel(np.core.defchararray.add(str(date)+' ',ts))
        pdb.set_trace()
        raw_txt['tempF']=(raw_txt['tempF']-32)/1.8
        return time_series, raw_txt['tempF'], raw_txt['dpF']

    def str_to_unix(self,s):
        dt = dparser.parse(s)
        secs = time.mktime(dt.timetuple())
        return secs

    def get_raw_api(self, start, end, geocode, key):
        # define a range of dates
        end = end + datetime.timedelta(days=1)
        dates = np.arange(start.date(), end.date(), dtype='datetime64[D]')
        # download the timestamp data for that range of dates
        raw = np.asarray(map(lambda x: self.get_daily_api(geocode,key,x),dates))
        # stack each day of downloaded data
        data = np.vstack(raw)[:,1:4].astype(float)
        timestamps = np.vstack(raw)[:,0]
        # convert to unix time 
        vec_parse = np.vectorize(self.str_to_unix_api)
        unix = vec_parse(timestamps)
        return timestamps, unix, data

    def get_daily_api(self,geocode,key,date):
        date = date.astype(datetime.datetime)
        month = date.strftime('%m')
        day = date.strftime('%d')
        url = 'http://api.wunderground.com/api/%s/history_%s%s%s/q/%s.json'\
                               %(key,date.year,month,day,geocode)
        try:
            f = urllib2.urlopen(url)
        except IOError:
            time.sleep(30)
            try:
                f = urllib2.urlopen(url)
            except:
                raise "operation stopped", date
        raw = json.loads(f.read())
        raw = raw['history']['observations']
        raw_txt = np.vstack(map(lambda x: self.parse_obs(x),raw))
        self.headers = 'time,tempF,dpF'
        return raw_txt

    def parse_obs(self,obs):
        self.dt = datetime.datetime(int(obs['date']['year']),\
                               int(obs['date']['mon']),\
                               int(obs['date']['mday']),\
                               int(obs['date']['hour']),\
                               int(obs['date']['min']))
        self.temp = obs['temps']
        self.dewpt = obs['dewpts']
        return self.dt, self.temp, self.dewpt

    def str_to_unix_api(self,s):
        secs = time.mktime(s.timetuple())
        return secs


class TMYData(object):
    def __init__(self, lat, lon, year, interval, **kwargs):
        self.tmy_file, self.cleaned_tmy = self.getTMY(lat, lon, year, interval)

    def getTMY(self,lat,lon,year,interval):
        f = open('./mave/data/epwurl.csv','r')
        #f = StringIO(pkgutil.get_data('./mave', 'data/epwurl.csv'))
        csv = np.genfromtxt(f, delimiter=',', dtype=None)
        csv_lat = csv[1:,4].astype(float)
        csv_lon = csv[1:,5].astype(float)
        min_idx = np.argmin(np.sqrt([(csv_lat-lat)**2+(csv_lon-lon)**2]))+1
        #downloading the zipfile from 'apps1.eere.energy.gov' website takes
        #two minutes. methods tried: urllib2.urlopen; urllib.urlretrieve;
        #request.get
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
        cols = list(filter(lambda x: x=="DryBulb" and x=="DewPoint",names))
        tmy = np.genfromtxt(un_zip_file.open(tmy_file), delimiter=',',\
                    dtype=None, skip_header=8, names=names, usecols=cols)
        if year == None: 
            np.place(tmy['year'],tmy['year']!=datetime.datetime.now().year,\
                     datetime.datetime.now().year)
        else:
            np.place(tmy['year'],tmy['year']!=year,year)
        comb_dt = np.column_stack((tmy['year'],tmy['month'],\
                                   tmy['day'],(tmy['hour']-1),\
                                   tmy['minute'])).astype(str).tolist()
        dt = map(lambda x: datetime.datetime.\
                                strptime(' '.join(x),'%Y %m %d %H %M'),\
                                comb_dt)
        unix_dt = map(lambda x: time.mktime(x.timetuple()),dt)
        target_dts = np.arange(dt[0],dt[-1],\
                               dtype='datetime64[%s]'%interval)\
                               .astype(datetime.datetime)
        target_dts = np.append(target_dts,dt[-1])
        target_unix = map(lambda x: time.mktime(x.timetuple()),target_dts)
        interp_db = np.interp(target_unix,unix_dt,tmy['DryBulb'])
        interp_dp = np.interp(target_unix,unix_dt,tmy['DewPoint'])
        target_dts = map(lambda x: x.isoformat(),target_dts)
        cleaned_tmy = np.column_stack((target_dts,interp_db,interp_dp))
        column_names = ','.join(cols)
        np.savetxt('./mave/data/clean_tmy.csv',\
                   cleaned_tmy,delimiter=',',fmt='%s',\
                   header=column_names, comments='')
        return tmy_file, cleaned_tmy

if __name__ == "__main__":
    address = 'wurster hall, uc berkeley'
    test = location(address)
    print 'address:',test.real_addrs
    print 'lat:',test.lat,' lon:',test.lon
    print 'nearest_airport:',test.geocode
    start = datetime.datetime(2015,1,1,0,0)
    end = datetime.datetime(2015,2,1,0,0)
    interp_interval = '15m'
    hist_weather = weather(start,end,None,test.geocode,\
                           interp_interval,False)
    test2 = TMYData(test.lat,test.lon,None,interp_interval)
    print 'TMY file:',test2.tmy_file
    print 'TMY Data:', test2.cleaned_tmy
