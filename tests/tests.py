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
        self.assertTrue(len(p.y) == len(p.datetimes))
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, use_holidays=False)
        self.assertTrue(p is not None)
        self.assertTrue(p.X.shape == (91977,7))
        f.close()

        f = open(self.TEST_PATH_1, 'Ur')
        p = None
        p = Preprocessor(f, use_holidays=True)
        self.assertTrue(p.X.shape == (91977,8))
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
        p = Preprocessor(f,use_holidays = True, start_frac=0.4, end_frac=0.6)
        self.assertTrue(p is not None)
        self.assertTrue(len(p.X) == 18406)
        f.close()

    def test_changepoint_feature(self):
        f = open(self.TEST_PATH_1, 'Ur')
        changepoints = [
            (datetime(2012, 1, 29, 13, 15), Preprocessor.PRE_DATA_TAG),
            (datetime(2012, 12, 20, 1, 15), Preprocessor.DISCARD_TAG),
            (datetime(2013, 1, 1, 1, 15), Preprocessor.PRE_DATA_TAG),
            (datetime(2013, 9, 14, 23, 15), Preprocessor.POST_DATA_TAG),
        ]
        p = Preprocessor(f,use_holidays = True, changepoints=changepoints, test_size=0.2)
        m = ModelAggregator(X=p.X_pre_s,y=p.y_pre_s,y_standardizer=p.y_standardizer)
        self.assertTrue(m is not None)
        self.assertTrue(len(p.X) == len(p.y)) 
        f.close()

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

    def test_more(self):
        f = open(self.TEST_PATH_1, 'Ur')
        p = Preprocessor(f)
        m = ModelAggregator(X=p.X_pre_s,y=p.y_pre_s,y_standardizer=p.y_standardizer)
        models = m.train_all()
        measured_post_retrofit = p.y_standardizer.inverse_transform(\
                                                             p.y_post_s)
        predicted_post_retrofit = p.y_standardizer.inverse_transform(\
                                    m.best_model.predict(p.X_post_s))
        error_metrics = Comparer(prediction=predicted_post_retrofit,\
                                 baseline=measured_post_retrofit)
        self.assertTrue(error_metrics is not None)
        self.assertTrue(error_metrics.r2 > 0.8)

    def test_singlemnv(self):
        f = open(self.TEST_PATH_2, 'Ur')
        changepoints = [
            (datetime(2012, 1, 29, 13, 15), Preprocessor.PRE_DATA_TAG),
            (datetime(2012, 12, 20, 1, 15), Preprocessor.DISCARD_TAG),
            (datetime(2013, 1, 1, 1, 15), Preprocessor.PRE_DATA_TAG),
            (datetime(2013, 9, 14, 23, 15), Preprocessor.POST_DATA_TAG),
        ]
        mnv = SingleModelMnV(f, changepoints=changepoints)
        pdb.set_trace()
        self.assertTrue(mnv.error_metrics.r2 != 0)

if __name__ == '__main__':
    unittest.main()
