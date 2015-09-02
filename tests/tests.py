import unittest, pdb, sys
from datetime import datetime
sys.path.insert(0, './mave/')
from mave.core import Preprocessor, ModelAggregator

class Test(unittest.TestCase):

    EPS = 0.001
    TEST_PATH_1 = "./mave/data/ex1.csv"
    TEST_PATH_6 = "./mave/data/ex6.csv"

    def test_success(self):
        self.assertTrue(True)

    def test_preprocessor(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 91977)
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, use_holidays=False)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 91977)
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, start_frac=0.4, end_frac=0.6)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 18403)
        f.close()
        
    def test_preprocessor_integer_data(self):   
        f = open(self.TEST_PATH_6, 'Ur')
        p = None
        p = Preprocessor(f, start_frac=0.4, end_frac=0.6)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 18406)
        f.close()

    def test_changepoint_feature(self):
        f = open(self.TEST_PATH_1, 'Ur')
        changepoints = [
            (datetime(2012, 1, 29, 13, 15),0),
            (datetime(2013, 9, 14, 23, 15),1),
        ]
        p = Preprocessor(f, changepoints=changepoints, test_size=0.2)
        m = ModelAggregator(X=p.X_pre_s,y=p.y_pre_s,y_standardizer=p.y_standardizer)
        self.assertTrue(m is not None)

    def test_model_aggregator(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f, test_size=0.2)
        m = ModelAggregator(X=p.X_pre_s,y=p.y_pre_s,y_standardizer=p.y_standardizer)
        self.assertTrue(m is not None)

        dummy = m.train("dummy")
        self.assertTrue(dummy is not None)
        score = m.score()
        print score
        #error = abs(0.89388258838080326 - score.mae)
        #self.assertTrue(error < self.EPS)

        m.train("hour_weekday")
        score = m.score()
        print score
        #error = abs(0.33109052903614433 - score.mae)
        #self.assertTrue(error < self.EPS)

if __name__ == '__main__':
    unittest.main()
