import unittest, pdb, sys
from datetime import datetime
sys.path.insert(0, './mave/')
from mave.core import Preprocessor, ModelAggregator, SingleModelMnV
import numpy as np
import trainers
from comparer import Comparer

class Test(unittest.TestCase):

    EPS = 0.001
    TEST_PATH_1 = "./mave/data/ex1.csv"
    TEST_PATH_2 = "./mave/data/ex2.csv"
    TEST_PATH_6 = "./mave/data/ex6.csv"

    def test_success(self):
        self.assertTrue(True)

    def test_preprocessor(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 91977)
        self.assertTrue(p.X.shape == (91977,8))
        self.assertTrue(len(p.X) == len(p.y)) 
        self.assertTrue(len(p.y) == len(p.datetimes))
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, start_frac=0.4, end_frac =0.6, use_holidays=False)
        self.assertTrue(p is not None)
        self.assertTrue(p.X.shape == (18403,7))
        self.assertTrue(len(p.X) == len(p.y)) 
        self.assertTrue(len(p.y) == len(p.datetimes))
        f.close()
        
    def test_preprocessor_integer_data(self):   
        f = open(self.TEST_PATH_6, 'Ur')
        p = None
        p = Preprocessor(f,use_holidays = True, start_frac=0.4, end_frac=0.6)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 18406)
        self.assertTrue(p.X.shape == (18406,8))
        f.close()

    def test_changepoint_feature(self):
        f = open(self.TEST_PATH_1, 'Ur')
        changepoints = [
                       ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
                       ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
                       ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2013/9/14 23:15", Preprocessor.POST_DATA_TAG),
                       ]
        p = Preprocessor(f,
                         use_holidays = True, 
                         changepoints=changepoints)
        m = ModelAggregator(X = p.X_pre_s,
                            y = p.y_pre_s,
                            y_standardizer = p.y_standardizer)
        self.assertTrue(m is not None)
        self.assertTrue(len(p.X) == len(p.y)) 
        f.close()

    def test_model_aggregator(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f, test_size=0.5)
        m = ModelAggregator(X = p.X_pre_s,
                            y = p.y_pre_s,
                            y_standardizer = p.y_standardizer)
        self.assertTrue(m is not None)

        dummy = m.train("dummy")
        self.assertTrue(dummy is not None)
        score = m.score()

        m.train("hour_weekday")
        score = m.score()
        f.close()

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

    def test_singlemnv(self):
        f = open(self.TEST_PATH_2, 'Ur')
        changepoints = [
                       ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
                       ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
                       ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2013/9/14 23:15", Preprocessor.POST_DATA_TAG),
                       ]
        mnv = SingleModelMnV(f, changepoints=changepoints)
        self.assertTrue(mnv.error_metrics.r2 > 0.3)
    
    def test_dualmnv(self):
        f = open(self.TEST_PATH_2, 'Ur')
        changepoints = [
                       ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
                       ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
                       ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
                       ("2013/9/14 23:15", Preprocessor.POST_DATA_TAG),
                       ]
        mnv = SingleModelMnV(f, changepoints=changepoints)
        self.assertTrue(mnv.error_metrics.r2 != 0)

if __name__ == '__main__':
    unittest.main()
