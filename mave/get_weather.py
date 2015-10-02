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

class GetWunder(object):
    def __init__(self,
                 start = datetime.datetime(2012,1,1,0,0),
                 end = datetime.datetime(2012,2,1,0,0),
                 geocode = 'SFO',
                 interp_interval = '15m',
                 **kwargs):
        self.target_dts = np.arange(start,
                                    end,
                                    dtype='datetime64[%s]'%interp_interval)
        self.target_unix = (self. target_dts - \
                            np.datetime64('1970-01-01T00:00:00Z'))\
                           /np.timedelta64(1,'s')
        self.timestamps, self.unix, self.data = \
                                              self.get_raw(start, end, geocode)
        self.interp_data = map(lambda x:  np.interp(self.target_unix,
                                     self.unix,
                                     self.data[x]), range(0,3))

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
        raw_txt = np.genfromtxt(raw, 
                                delimiter=',',
                                names='time,tempF,dpF,RH',
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

if __name__ == "__main__":
    start = datetime.datetime(2012,1,1,0,0)
    end = datetime.datetime(2012,2,1,0,0)
    geocode = 'SFO'
    interp_interval = '15m'
    test = GetWunder(start, end, geocode, interp_interval)
    print '\nTarget datetimes'
    print test.target_dts
    print '\nInterpolated data'
    print test.interp_data 
