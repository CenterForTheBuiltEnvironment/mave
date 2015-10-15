import unittest
import pdb
import sys
import datetime 
import numpy as np
sys.path.insert(0, './mave/')
sys.path.insert(0, '../mave/')
import trainers
from core import Preprocessor, ModelAggregator, SingleModelMnV
from comparer import Comparer
from weather import Weather

class Test(unittest.TestCase):

    EPS = 0.001
    TEST_PATH_1 = "./mave/data/ex1.csv"
    TEST_PATH_2 = "./mave/data/ex2.csv"
    TEST_PATH_3 = "./mave/data/ex3.csv"
    TEST_PATH_4 = "./mave/data/ex4.csv"
    TEST_PATH_5 = "./mave/data/ex5.csv"
    TEST_PATH_6 = "./mave/data/ex6.csv"
    TEST_PATH_7 = "./mave/data/ex7.csv"
    TEST_PATH_8 = "./mave/data/ex8.csv"
    WEATHER_PATH = "./mave/data/weather_test.csv"

    def test_success(self):
        self.assertTrue(True)

    def test_preprocessor(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f, end_frac =0.2)
        self.assertTrue(p is not None)
        self.assertTrue(p.X.shape == (18411,8))
        self.assertTrue(len(p.X) == len(p.y)) 
        self.assertTrue(len(p.y) == len(p.dts))
        self.assertTrue(int(p.X[10000][0]) == p.dts[10000].minute)
        f.close()

    def test_preprocessor_remove_outliers(self):
        f = open(self.TEST_PATH_2, 'Ur')
        p = Preprocessor(f, end_frac=0.05, remove_outliers=False)
        f.close()
        f = open(self.TEST_PATH_2, 'Ur')
        pm = Preprocessor(f, end_frac=0.05, remove_outliers='MultipleValues')
        f.close()
        f = open(self.TEST_PATH_2, 'Ur')
        ps = Preprocessor(f, end_frac=0.05, remove_outliers='SingleValue')
        for a in (p,pm,ps):
            self.assertTrue(a is not None)
            self.assertTrue(len(a.X) == len(a.y)) 
            self.assertTrue(len(a.y) == len(a.dts))
            self.assertTrue(int(a.X[1000][0]) == a.dts[1000].minute)
        self.assertTrue(p.X.shape == (4593,8))
        self.assertTrue(pm.X.shape == (4590,8))
        self.assertTrue(ps.X.shape == (4591,8))
        f.close()
        
    def test_changepoint_feature(self):
        f = open(self.TEST_PATH_3, 'Ur')
        changepoints = [
                       ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
                       ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
                       ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2013/9/14 23:15", Preprocessor.POST_DATA_TAG),
                       ]
        p = Preprocessor(f,
                         use_holidays = True, 
                         changepoints=changepoints)
        self.assertTrue(len(p.X) == len(p.y)) 
        self.assertTrue(len(p.y) == len(p.dts))
        self.assertTrue(int(p.X[10000][0]) == p.dts[10000].minute)
        self.assertTrue(len(p.X_pre_s) == 55877)
        f.close()

    def test_model_aggregator(self):
        f = open(self.TEST_PATH_4, 'Ur')
        p = Preprocessor(f, test_size=0.75)
        m = ModelAggregator(X = p.X_pre_s,
                            y = p.y_pre_s,
                            y_standardizer = p.y_standardizer)
        self.assertTrue(m is not None)
        dummy = m.train("dummy")
        self.assertTrue(dummy is not None)
        score = m.score()
        m.train("hour_weekday")
        score = m.score()
        self.assertTrue(m.error_metrics.cvrmse < 47.0)
        f.close()

    def test_singlemnv(self):
        f = open(self.TEST_PATH_5, 'Ur')
        changepoints = [
                       ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
                       ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
                       ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2013/2/14 23:15", Preprocessor.POST_DATA_TAG),
                       ]
        mnv = SingleModelMnV(f, changepoints=changepoints)
        self.assertTrue(mnv.error_metrics.r2 > 0.25)
    
    def test_preprocessor_integer_data(self):   
        f = open(self.TEST_PATH_6, 'Ur')
        p = Preprocessor(f,use_holidays = True, start_frac=0.4, end_frac=0.6)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 18406)
        self.assertTrue(p.X.shape == (18406,8))
        f.close()

    def test_dualmnv(self):
        f = open(self.TEST_PATH_7, 'Ur')
        changepoints = [
                       ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
                       ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
                       ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2013/2/14 23:15", Preprocessor.POST_DATA_TAG),
                       ]
        mnv = SingleModelMnV(f, changepoints=changepoints)
        self.assertTrue(mnv.error_metrics.r2 >= 0.25)

    def test_random_forest(self):
        x = np.random.randint(10000, size=(10000,3))+100
        y = np.random.randint(10000, size=(10000,))+100
        random_forest_trainer = trainers.RandomForestTrainer(\
                                                           search_iterations=20)
        model = random_forest_trainer.train(x,y)
        self.assertTrue(random_forest_trainer is not None)
        X = np.random.randint(10000, size=(10000,3))+200
        y_predicted = model.predict(X)
        c = Comparer(y_predicted, y)
        self.assertTrue(c.r2 < 0.01)

    def test_preprocessor_no_hols(self):
        f = open(self.TEST_PATH_8, 'Ur')
        p = Preprocessor(f, start_frac=0.4, end_frac =0.6, use_holidays=False)
        self.assertTrue(p is not None)
        self.assertTrue(p.X.shape == (18287,7))
        self.assertTrue(len(p.X) == len(p.y)) 
        self.assertTrue(len(p.y) == len(p.dts))
        self.assertTrue(int(p.X[10000][0]) == p.dts[10000].minute)
        f.close()

    def test_getweather(self):
        start = datetime.datetime(2012,1,1,0,0)
        end = datetime.datetime(2012,1,3,0,0)
        web = Weather(start=start,end=end,geocode='SFO',zipcode=None,
                      key=None,interp_interval='15m',save=False)
        geocode = None
        api = Weather(start=start,end=end,geocode=None,zipcode='94128',
                      key='d3dffb3b59309a05',interp_interval='15m',save=False)
        f = open(self.WEATHER_PATH, 'Ur')
        txt = np.genfromtxt(f.read().splitlines(), delimiter=',',dtype = None)
        dat = txt['f5'][4:7]
        self.assertTrue(web.data != None)
        self.assertAlmostEqual(float(web.data[0][0]),46.9)  
        self.assertTrue(web.interp_data[0][3:6].all() == dat.all())
        self.assertTrue(api.interp_data[0][3:6].all() == dat.all())

if __name__ == '__main__':
    unittest.main()
