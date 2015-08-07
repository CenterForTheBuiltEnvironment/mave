import unittest

from mave.core import Preprocessor, ModelAggregator

class Test(unittest.TestCase):

    EPS = 0.001
    TEST_PATH_1 = "mave/data/6_P_cbe_02.csv"

    def test_success(self):
        self.assertTrue(True)

    def test_preprocessor(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 35002)
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, use_holidays=False)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 35002)
        f.close()
        
        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, start_frac=0.1, end_frac=0.8)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 24501)
        f.close()

    def test_model_aggregator(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        m = ModelAggregator(p, test_size=0.2)
        self.assertTrue(m is not None)

        dummy = m.train_dummy()
        self.assertTrue(dummy is not None)
        score = m.score()
        mae = score[3]
        error = abs(0.88233783891011031 - mae)
        self.assertTrue(error < self.EPS)

        m.train_hour_weekday()
        score = m.score()
        mae = score[3]
        error = abs(0.345048548272 - mae)
        self.assertTrue(error < self.EPS)
