"""
This class is a wrapper for all of data required to train or evaluate a model

the dataset_type field is to help standardize notation of different datasets:
    1:'measured preretrofit data'
    2:'preretrofit data predicted using preretrofit model'
    3:'preretrofit data predicted using postretrofit model'
    4:'measured postretofit data'
    5:'postretrofit data predicted using preretrofit model'
    6:'postretrofit data predicted using postretrofit model'
    7:'tmy data predicted using preretrofit model'
    8:'tmy data predicted using postretrofit model'

typical comparisons used by mave:
    Pre-retrofit model performance = 1 vs 2
    Single model M&V = 4 vs 5
    Post retrofit model performance  = 4 vs 6
    Dual model M&V, normalized to tmy data = 7 vs 8

@author Paul Raftery <p.raftery@berkeley.edu>
"""
from sklearn import preprocessing
import numpy as np

class Dataset(object):
    def __init__(self, 
                 dataset_type=None,
                 X=None, 
                 X_s=None,
                 X_standardizer=None,
                 dts=None,
                 feature_names=None,
                 y=None,
                 y_s=None,
                 y_standardizer=None):
        assert isinstance(dataset_type,int)
        assert dataset_type in set([1,2,3,4,5,6,7,8])
        self.dataset_type = dataset_type
        # ensure standardizers are present
        assert isinstance(X_standardizer, preprocessing.data.StandardScaler)
        assert isinstance(y_standardizer, preprocessing.data.StandardScaler)
        self.X_standardizer = X_standardizer
        self.y_standardizer = y_standardizer
        # ensure both representations of X and y are present and same length
        assert isinstance(X, np.ndarray) or isinstance(X_s, np.ndarry)
        assert isinstance(y, np.ndarray) or isinstance(y_s, np.ndarry)
        self.X = X
        self.X_s = X_s
        self.y = y
        self.y_s = y_s
        if not isinstance(self.X_s,np.ndarray):
            self.X_s = self.X_standardizer.inverse_transform(self.X)
        if not isinstance(self.X,np.ndarray):
            self.X = self.X_standardizer.transform(self.X_s)
        if not isinstance(self.y_s,np.ndarray):
            self.y_s = self.y_standardizer.inverse_transform(self.y)
        if not isinstance(self.y,np.ndarray):
            self.y = self.y_standardizer.transform(self.y_s)
        assert self.X.shape[0] == len(self.y)
        # ensure a set of datetimes is present and the correct length
        assert isinstance(dts,list)
        assert len(dts) == len(self.y)
        self.dts = dts
        # ensure a set of feature names is present and of correct length
        assert isinstance(feature_names,list)
        assert len(feature_names) == X.shape[1]
        self.feature_names = feature_names
        
    def write_to_csv(self, filename='Results.csv'):
        str_date = map(lambda arr: arr.strftime(self.p.timestamp_format),
                       self.dts)
        header = self.feature_names.extend(['measured','predicted'])
        header = ','.join(map(str, header))
        output = np.column_stack((np.array(str_date),
                                  X,
                                  y,
                                  predicted,))
        np.savetxt(filename,
                   output,
                   delimiter=',',
                   header=header,
                   fmt='%s',
                   comments='')

    def __str__(self):
        desc ={1:'measured preretrofit data',
               2:'preretrofit data predicted using preretrofit model',
               3:'preretrofit data predicted using postretrofit model',
               4:'measured postretofit data',
               5:'postretrofit data predicted using preretrofit model',
               6:'postretrofit data predicted using postretrofit model',
               7:'tmy data predicted using preretrofit model',
               8:'tmy data predicted using postretrofit model'}
        return 'Dataset type: %s, %s'%(self.dataset_type,desc[self.dataset_type])

if __name__=='__main__':
   import numpy as np
   import pdb
   X = np.random.rand(6,3)
   y = np.random.rand(6,1)
   X_standardizer = preprocessing.StandardScaler().fit(X)
   y_standardizer = preprocessing.StandardScaler().fit(y)
   dts = [1,2,3,4,5,6]
   feature_names = ['Minute','Hour','DayOfWeek']
   test = Dataset(dataset_type=1,
                  X=X,
                  X_standardizer=X_standardizer,
                  y=y,
                  y_standardizer=y_standardizer,
                  dts=dts,
                  feature_names=feature_names)
   print test
   print test.X
   print test.X_s
   print test.y
   print test.y_s

