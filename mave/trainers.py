import numpy as np
import pdb
import estimators
from scipy.stats import randint as sp_randint
from sklearn import cross_validation, svm, grid_search, \
        ensemble, neighbors, dummy

class ModelTrainer(object):

    def __init__(self, 
                 search_iterations=20, 
                 n_jobs=-1, 
                 k=10, 
                 verbose= False, 
                 **kwargs):
        self.search_iterations = search_iterations
        self.n_jobs = n_jobs
        self.k = k
        self.verbose = verbose
    
    def train(self, X_s, y_s, randomized_search=True):
        # using a random grid search assessed using k-fold cross validation
        if randomized_search:
            self.model = grid_search.RandomizedSearchCV(
                                     self.model, 
                                     param_distributions=self.params,
                                     n_iter=self.search_iterations,
                                     n_jobs=self.n_jobs,
                                     cv=self.k,
                                     verbose=self.verbose)
        # otherwise do an exhaustive grid search
        else:
            self.model = grid_search.GridSearchCV(
                                     self.model, 
                                     param_grid=self.params,
                                     n_jobs=self.n_jobs,
                                     cv=self.k,
                                     verbose=self.verbose)

        self.model.fit(X_s, y_s)
        return self.model

class DummyTrainer(ModelTrainer):
    params = {"strategy": ['mean', 'median']}
    model = dummy.DummyRegressor()

    def __init__(self, **kwargs):
        super(DummyTrainer, self).__init__(**kwargs) 

class HourWeekdayBinModelTrainer(ModelTrainer): 
    params = {"strategy": ['mean', 'median']}
    model = estimators.HourWeekdayBinModel()

    def __init__(self, **kwargs):
        super(HourWeekdayBinModelTrainer, self).__init__(**kwargs) 

class KNeighborsTrainer(ModelTrainer):
    params = {
                "p": [1,2],
                "n_neighbors": sp_randint(6, 40),
                "leaf_size": np.logspace(1, 2.5, 1000)
    }
    model = neighbors.KNeighborsRegressor()

    def __init__(self, **kwargs):
        super(KNeighborsTrainer, self).__init__(**kwargs) 

class SVRTrainer(ModelTrainer):
    params = {
                "C": np.logspace(-3, 1, 1000),
                "epsilon": np.logspace(-3, 0.5, 1000),
                "degree": [2,3,4],
                "gamma": np.logspace(-3, 2, 1000),
                "max_iter": [20000]
    }
    model = svm.SVR()

    def __init__(self, **kwargs):
        super(SVRTrainer, self).__init__(**kwargs) 

class RandomForestTrainer(ModelTrainer):
    max_features = 4
    params = {
                "max_depth": [4, 5, 6, 7, 8, 9, 10, None],
                "max_features": sp_randint(3, max_features),
                "min_samples_split": sp_randint(5, 500),
                "min_samples_leaf": sp_randint(5, 500),
                "bootstrap": [True, False]
    }
    model = ensemble.RandomForestRegressor()

    def __init__(self, **kwargs):
        super(RandomForestTrainer, self).__init__(**kwargs) 

class GradientBoostingTrainer(ModelTrainer):
    max_features = 4
    params = {
                "max_depth": [4, 5, 6, 7, 8, 9, 10, None],
                "n_estimators": np.logspace(1.5, 4, 1000).astype(int),
                "max_features": sp_randint(3, max_features),
                "min_samples_split": sp_randint(5, 50),
                "min_samples_leaf": sp_randint(5, 50),
                "subsample": [0.8, 1.0],
                "learning_rate": [0.05, 0.1, 0.2, 0.5]
    }
    model = ensemble.GradientBoostingRegressor()

    def __init__(self, **kwargs):
        super(GradientBoostingTrainer, self).__init__(**kwargs) 

class ExtraTreesTrainer(ModelTrainer):
    max_features = 4
    params = {
                "max_depth": [4, 5, 6, 7, 8, 9, 10, None],
                "n_estimators": sp_randint(5, 50),
                "max_features": sp_randint(3, max_features),
                "min_samples_split": sp_randint(5, 50),
                "min_samples_leaf": sp_randint(5, 50),
                "bootstrap": [True, False]
    }
    model = ensemble.ExtraTreesRegressor()

    def __init__(self, **kwargs):
        super(ExtraTreesTrainer, self).__init__(**kwargs) 

if __name__=='__main__':
    t = DummyTrainer()
