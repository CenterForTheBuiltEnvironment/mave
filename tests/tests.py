import unittest
import pdb
import sys
from datetime import datetime 
import numpy as np
sys.path.insert(0, './mave/')
sys.path.insert(0, '../mave/')
import trainers
from core import Preprocessor, ModelAggregator, MnV
from comparer import Comparer
import dataset
import location
from nose.tools import assert_equal, assert_true, assert_false, assert_almost_equal

class Test(unittest.TestCase):

    F_1 = "./mave/data/ex4_short_no_weather.csv"
    F_2 = "./mave/data/ex4_short.csv"

    def test_preprocessor(self):
        # a basic test of the preprocessor object
        f = open(self.F_1, 'Ur')
        o = Preprocessor(f)
        assert_equal(o.X.shape, (5643, 4))
        assert_equal(len(o.X), len(o.y)) 
        assert_equal(len(o.y), len(o.dts))
        assert_equal(int(o.X[5000][0]), o.dts[5000].minute)
        assert_equal(int(o.X[5000][1]), o.dts[5000].hour)
        # test if holiday feature identifies first day of year as holiday
        assert_equal(o.X[0][3], 3 )
        f.close()

    def test_changepoint_feature(self):
        # test changepoint feature 
        f = open(self.F_1, 'Ur')
        changepoints = [
                       ("2007/1/14 01:15", Preprocessor.POST_DATA_TAG),
                       ("2007/1/17 01:15", Preprocessor.IGNORE_TAG),
                       ("2007/1/18 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2007/1/31 23:45", Preprocessor.POST_DATA_TAG),
                       ]
        o = Preprocessor(f, changepoints=changepoints, use_holidays=False)
        # test if datetimes correctly dentified as pre/post/ignore 
        assert_true(datetime(2007, 1, 14, 1, 15) in o.dts_post)
        assert_true(datetime(2007, 1, 18, 1, 15) in o.dts_pre)
        assert_true(datetime(2007, 1, 17, 1, 15) in o.dts)
        assert_false(datetime(2007, 1, 17, 1, 15) in o.dts_pre)
        assert_false(datetime(2007, 1, 17, 1, 15) in o.dts_post)
        # test holiday feature not present
        assert_equal(o.X.shape[1], 3)
        assert_false('Holiday' in o.feature_names)
        f.close()

    def test_model_aggregator(self):
        # test dataset and modelaggregator 
        f = open(self.F_1, 'Ur')
        o = Preprocessor(f, test_size=0.5, use_month=True)
        # test correct number of features present (min,hr,day,hol,mnt)
        assert_equal(o.X.shape[1], 5)
        assert_true('Month' in o.feature_names)
        A = dataset.Dataset(dataset_type='A',
                            X_s=o.X_pre_s,
                            X_standardizer=o.X_standardizer,
                            y_s=o.y_pre_s,
                            y_standardizer=o.y_standardizer,
                            dts=o.dts_pre,
                            feature_names=o.feature_names)
        m = ModelAggregator(dataset=A)
        m.train_all(k=3)
        score = m.score()
        assert_true(m.error_metrics.r2 > 0.9)
        f.close()

    def test_mnv(self):
        f = open(self.F_1, 'Ur')
        mnv = MnV(f)
        assert_true(mnv.DvsE.r2 > 0.8)

    def test_mnv_with_weather_and_tmy(self):
        f = open(self.F_2, 'Ur')
        mnv = MnV(f, use_tmy=True, address='berkeley california', ts=0.5)
        assert_true(mnv.DvsE.r2 > 0.8)
        assert_true(mnv.GvsH.r2 > 0.6)

    def test_location(self):
        o = location.Location(address='berkeley california')
        assert_equal(o.real_addrs, 'Berkeley, CA, USA')
        assert_almost_equal(o.lat, 37.87, places=2)
        assert_almost_equal(o.lon, -122.27, places=2)
    
    def test_getweather(self):
        start = datetime(2012,1,1,0,0)
        end = datetime(2012,1,2,0,0)
        o = location.Weather(start=start,
                             end=end,
                             geocode='SFO',
                             interp_interval=15,
                             save=False)
        assert_equal(o.data[0].shape,(48,))
        assert_almost_equal(float(o.data[0][0]), 8.277777777)
        assert_equal(o.interp_data[0].shape,(97,))
        assert_almost_equal(o.interp_data[-1][-1], 42.75333333)
  

if __name__ == '__main__':
    unittest.main()
