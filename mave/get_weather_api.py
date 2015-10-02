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
                 end = datetime.datetime(2012,1,3,0,0),
                 zipcode = '94720',
                 key = 'd3dffb3b59309a05',
                 interp_interval = '15m',
                 save = True,
                 **kwargs):
        self.target_dts = np.arange(start,
                                    end,
                                    dtype='datetime64[%s]'%interp_interval)
        self.target_unix = (self.target_dts - \
                            np.datetime64('1970-01-01T00:00:00Z'))\
                           /np.timedelta64(1,'s')
        self.timestamps, self.unix, self.data = \
                                              self.get_raw(start, end, zipcode)
        self.interp_data = map(lambda x:  np.interp(self.target_unix,
                                     self.unix,
                                     self.data[:,x]), range(0,3))
        if save:
            out_data = np.column_stack(self.interp_data).astype(str)
            out_time = np.vstack(self.target_dts).astype(str)
            data_frame = np.column_stack([out_time,out_data])
            header = 'time,temp,dewp,rh'
            np.savetxt('weather_api.csv', data_frame, header=header,\
                       delimiter=',', fmt='%s', comments= '')
       
    def get_raw(self, start, end, zipcode):
        # define a range of dates
        dates = np.arange(start.date(), end.date(), dtype='datetime64[D]')
        # download the timestamp data for that range of dates
        raw = np.asarray(map(lambda x: self.get_daily(zipcode,key,x),dates))
        # stack each day of downloaded data
        data = np.vstack(raw)[:,1:4].astype(float)
        timestamps = np.vstack(raw)[:,0]
        # convert to unix time 
        vec_parse = np.vectorize(self.str_to_unix)
        unix = vec_parse(timestamps)
        return timestamps, unix, data

    def get_daily(self,zipcode,key,date):
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

    def str_to_unix(self,s):
        #dt = dparser.parse(s)
        secs = (s-datetime.datetime(1970,1,1)).total_seconds()
        return secs

if __name__ == "__main__":
    start = datetime.datetime(2012,1,1,0,0)
    end = datetime.datetime(2012,1,3,0,0)
    zipcode = '94720'
    key = 'd3dffb3b59309a05'
    interp_interval = '15m'
    test = GetWunder(start, end, zipcode,key, interp_interval)
    print '\nTarget datetimes'
    print test.target_dts
    print '\nInterpolated data'
    print test.interp_data
