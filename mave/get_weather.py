"""
This software downloads weather data for a given location between
a given start and end date. It interpolates the weather data to a
given uniform interval between those dates. 

@author Taoning Wang <taoning@berkeley.edu>
@author Paul Raftery <p.raftery@berkeley.edu>
"""

import urllib2
import numpy as np
import datetime, time
import dateutil.parser as dparser
import pdb
import json

class GetWunder(object):
    def __init__(self,
                 start = datetime.datetime(2012,1,1,0,0),
                 end = datetime.datetime(2012,2,1,0,0),
                 geocode = 'SFO',
                 zipcode = None,
                 key = None,
                 interp_interval = '15m',
                 save = True,
                 **kwargs):
        self.target_dts = np.arange(start,
                                    end,
                                    dtype='datetime64[%s]'%interp_interval)
        self.target_unix = (self.target_dts - \
                            np.datetime64('1970-01-01T00:00:00Z'))\
                           /np.timedelta64(1,'s')
        if geocode != None:
            self.timestamps, self.unix, self.data = \
                                              self.get_raw(start, end, geocode)
        else:
            self.timestamps, self.unix, self.data = \
                                            self.get_raw_api(start,end,zipcode)
            
        self.interp_data = map(lambda x:  np.interp(self.target_unix,
                                                    self.unix,
                                                    self.data[x]), range(0,3))
        if save:
            out_time = np.vstack(self.target_dts).astype(str)
            out_data = np.column_stack(self.interp_data).astype(str)
            data_frame = np.column_stack([out_time,out_data])
            np.savetxt('weather.csv',data_frame,\
                       delimiter=',',header = self.headers,\
                       fmt ='%s',comments='')

    def get_raw(self, start, end, geocode):
        # define a range of dates
        dates = np.arange(start.date(), end.date(), dtype='datetime64[D]')
        # download the timestamp data for that range of dates
        raw = np.asarray(map(lambda x: self.get_daily(geocode,x),dates))
        # stack each day of downloaded data
        data = map(lambda x: np.hstack(raw[:,x]), list([1,2,3]))
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
        except IOError:
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
                                usecols=('time,tempF,dpF,RH'),
                                dtype=None,
                                skip_header=2)
        ts = raw_txt['time']
        time_series = np.ravel(np.core.defchararray.add(str(date)+' ',ts))
        return time_series, raw_txt['tempF'], raw_txt['dpF'], raw_txt['RH']

    def str_to_unix(self,s):
        dt = dparser.parse(s)
        secs = time.mktime(dt.timetuple())
        return secs

    def get_raw_api(self, start, end, zipcode):
        # define a range of dates
        dates = np.arange(start.date(), end.date(), dtype='datetime64[D]')
        # download the timestamp data for that range of dates
        raw = np.asarray(map(lambda x: self.get_daily_api(zipcode,key,x),dates))
        # stack each day of downloaded data
        data = np.vstack(raw)[:,1:4].astype(float)
        timestamps = np.vstack(raw)[:,0]
        # convert to unix time 
        vec_parse = np.vectorize(self.str_to_unix_api)
        unix = vec_parse(timestamps)
        return timestamps, unix, data

    def get_daily_api(self,zipcode,key,date):
        date = date.astype(datetime.datetime)
        month = date.strftime('%m')
        day = date.strftime('%d')
        url = 'http://api.wunderground.com/api/%s/history_%s%s%s/q/%s.json'\
                               %(key,date.year,month,day,zipcode)
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
        return raw_txt

    def parse_obs(self,obs):
        self.dt = datetime.datetime(int(obs['date']['year']),\
                               int(obs['date']['mon']),\
                               int(obs['date']['mday']),\
                               int(obs['date']['hour']),\
                               int(obs['date']['min']))
        self.temp = obs['tempi']
        self.dewpt = obs['dewpti']
        self.hum = obs['hum']
        return self.dt, self.temp, self.dewpt, self.hum

    def str_to_unix_api(self,s):
        secs = (s-datetime.datetime(1970,1,1)).total_seconds()
        return secs


if __name__ == "__main__":
    start = datetime.datetime(2012,1,1,0,0)
    end = datetime.datetime(2012,2,1,0,0)
    geocode = 'SFO'
    key = None
    zipcode = None
    interp_interval = '60m'
    test = GetWunder(start=start, 
                     end=end, 
                     geocode=geocode, 
                     interp_interval=interp_interval,
                     save=True)
    print '\nTarget datetimes'
    print test.target_dts
    print '\nInterpolated data'
    print test.interp_data 
