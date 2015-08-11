import unittest
from datetime import datetime
from mave.core import Preprocessor, ModelAggregator

class Test(unittest.TestCase):

    EPS = 0.001
    TEST_PATH_1 = "./mave/data/Ex1.csv"

    def test_success(self):
        self.assertTrue(True)

    def test_preprocessor(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 92055)
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, use_holidays=False)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 92055)
        f.close()
        
        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, start_frac=0.4, end_frac=0.6)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 18415)
        f.close()

    def test_changepoint_feature(self):
        f = open(self.TEST_PATH_1, 'Ur')
        changepoints = [
            datetime(2012, 1, 29, 13, 15),
            datetime(2013, 9, 14, 23, 15),
        ]
        p = Preprocessor(f, changepoints=changepoints)
        m = ModelAggregator(p, test_size=0.2)
        self.assertTrue(m is not None)

    def test_model_aggregator(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        m = ModelAggregator(p, test_size=0.2)
        self.assertTrue(m is not None)

        dummy = m.train_dummy()
        self.assertTrue(dummy is not None)
        score = m.score()
        mae = score[3]
        error = abs(0.89388258838080326 - mae)
        self.assertTrue(error < self.EPS)

        m.train_hour_weekday()
        score = m.score()
        mae = score[3]
        error = abs(0.33109052903614433 - mae)
        self.assertTrue(error < self.EPS)

if __name__ == '__main__':
    unittest.main()
