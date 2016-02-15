"""
This class is a wrapper for all of data required to train or evaluate a model

the dataset_type field is to help standardize notation of different datasets:
    A:'measured preretrofit data'
    B:'preretrofit data predicted using preretrofit model'
    C:'preretrofit data predicted using postretrofit model'
    D:'measured postretofit data'
    E:'postretrofit data predicted using preretrofit model'
    F:'postretrofit data predicted using postretrofit model'
    G:'tmy data predicted using preretrofit model'
    H:'tmy data predicted using postretrofit model'

typical comparisons used by mave:
    Pre-retrofit model performance = A vs B
    Single model M&V = D vs E
    Post retrofit model performance  = D vs F
    Dual model M&V, normalized to tmy data = G vs H

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
        assert isinstance(dataset_type,str), \
               "dataset_type is not a char: %s"%dataset_type
        assert dataset_type in set(['A','B','C','D','E','F','G','H']), \
               "dataset_type is no a character from A to H: %str"%datase_type
        self.dataset_type = dataset_type
        # ensure standardizers are present
        assert isinstance(X_standardizer, preprocessing.data.StandardScaler), \
               "X_standardizer is not an instance of preprocessing.data.StandardScaler:%s"%type(X_standardizer)
        assert isinstance(y_standardizer, preprocessing.data.StandardScaler), \
               "y_standardizer is not an instance of preprocessing.data.StandardScaler:%s"%type(y_standardizer)
        self.X_standardizer = X_standardizer
        self.y_standardizer = y_standardizer
        # ensure both representations of X and y are present and same length
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
        assert self.X.shape[0] == len(self.y), \
               "length of X (%s) doesn't match y (%s)"%(X.shape[0],len(self.y))
        # ensure a set of datetimes is present and the correct length
        assert isinstance(dts,list), \
               "dts is not a list object: %s"%type(dts)
        assert len(dts) == len(self.y), \
               "length of dts (%s) doesn't match y (%s)"%(len(dts),len(self.y))
        self.dts = dts
        # ensure a set of feature names is present and of correct length
        assert isinstance(feature_names,list), \
               "feature_names is not a list object: %s"%type(feature_names)
        assert len(feature_names) == X.shape[1], \
               "different num of feature_names than features"
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
        desc ={'A':'measured preretrofit data',
               'B':'preretrofit data predicted using preretrofit model',
               'C':'preretrofit data predicted using postretrofit model',
               'D':'measured postretofit data',
               'E':'postretrofit data predicted using preretrofit model',
               'F':'postretrofit data predicted using postretrofit model',
               'G':'tmy data predicted using preretrofit model',
               'H':'tmy data predicted using postretrofit model'}
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
   test = Dataset(dataset_type='A',
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

