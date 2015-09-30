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
        self.raw_dts, self.raw_data = self.get_raw(start, end, geocode)
        self.processed_data = self.process_time(self.target_dts,
                                                self.raw_dts,
                                                self.raw_data)

    def get_raw(self, start, end, geocode):
        dates = np.arange(start.date(), end.date(), dtype='datetime64[D]')
        pdb.set_trace()
        raw = np.asarray(map(lambda x: self.get_daily(geocode,x),dates))
        raw_dts = np.hstack(raw[:,0])
        raw_data = np.hstack(raw[:,1])
        return raw_dts, raw_data

    def get_daily(self,geocode,dt):
        year = dt.astype('datetime64[Y]').astype(int)+1970
        month = dt.astype('datetime64[M]').astype(int)%12+1
        day = int((dt-dt.astype('datetime64[M]'))/np.timedelta64(1,'D'))+1
        f = urllib2.urlopen('http://www.wunderground.com/history/airport/'\
           +geocode+'/'+str(year)+'/'+str(month)+'/'+str(day)+'/'\
           +'DailyHistory.html?format=1')
        raw = f.read().splitlines()
        raw_txt = np.genfromtxt(raw, delimiter=',',\
                                dtype=None,\
                                skip_header=2)
        time = raw_txt['f0']
        dt_str = str(dt)[:10]+' '+str(dt)[11:19]
        time_series = np.ravel(np.core.defchararray.add(dt_str,time))
        return time_series, raw_txt['f1']

    def process_time(self, target_dts, current_dts, data):
        func_p = np.vectorize(dparser.parse)
        func2 = np.vectorize(self.get_unixtime)
        try:
            dt = map(lambda x: time.strptime(x,'%y-%b-%d %I:%M:%S'),current_dts)
        except:
            dt = func_p(current_dts)
        dt_float = func2(dt) 
        normal_float = (target_dts - np.datetime64('1970-01-01T00:00:00Z'))\
                       /np.timedelta64(1,'s')
        target_data = np.interp(normal_float,dt_float,data)
        return target_data

    def get_unixtime(self,dt):
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
    print test.processed_data 
