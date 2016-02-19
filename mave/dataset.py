"""
This class is a wrapper for all of data required to train or evaluate a model

the dataset_type field is to help standardize notation of different datasets:
       'A':'Measured preretrofit data',
       'B':'Preretrofit prediction - preretrofit model',
       'C':'Preretrofit prediction - postretrofit model',
       'D':'Measured postretrofit data',
       'E':'Postretrofit prediction - preretrofit model',
       'F':'Postretrofit prediction - postretrofit model',
       'G':'TMY prediction: preretrofit model',
       'H':'TMY prediction: postretrofit model'}

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
    DESC ={'A':'Measured preretrofit data',
           'B':'Preretrofit prediction - preretrofit model',
           'C':'Preretrofit prediction - postretrofit model',
           'D':'Measured postretrofit data',
           'E':'Postretrofit prediction - preretrofit model',
           'F':'Postretrofit prediction - postretrofit model',
           'G':'TMY prediction - preretrofit model',
           'H':'TMY prediction - postretrofit model'}

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
               "X_standardizer is not an instance " + \
               "of preprocessing.data.StandardScaler:%s"%type(X_standardizer)
        assert isinstance(y_standardizer, preprocessing.data.StandardScaler), \
               "y_standardizer is not an instance " + \
               "of preprocessing.data.StandardScaler:%s"%type(y_standardizer)
        self.X_standardizer = X_standardizer
        self.y_standardizer = y_standardizer
        # ensure both representations of X and y are present and same length
        self.X = X
        self.X_s = X_s
        self.y = y
        self.y_s = y_s
        if not isinstance(self.X_s, np.ndarray):
            self.X_s = self.X_standardizer.transform(self.X)
        if not isinstance(self.X, np.ndarray):
            self.X = self.X_standardizer.inverse_transform(self.X_s)
        if not isinstance(self.y_s, np.ndarray):
            self.y_s = self.y_standardizer.transform(self.y)
        if not isinstance(self.y, np.ndarray):
            self.y = self.y_standardizer.inverse_transform(self.y_s)
        assert self.X.shape[0] == len(self.y), \
               "length of X (%s) doesn't match y (%s)"%(X.shape[0],len(self.y))
        # ensure datetimes are the correct length
        assert len(dts) == len(self.y), \
               "length of dts (%s) doesn't match y (%s)"%(len(dts),len(self.y))
        self.dts = dts
        # ensure a set of feature names is present and of correct length
        assert isinstance(feature_names,list), \
               "feature_names is not a list object: %s"%type(feature_names)
        assert len(feature_names) == self.X.shape[1], \
               "different num of feature_names than features"
        self.feature_names = feature_names
        
    def write_to_csv(self, 
                     filename=None, 
                     timestamp_format='%Y-%m-%d%T%H%M'):
        str_date = map(lambda arr: arr.strftime(timestamp_format),
                       self.dts)
        if not filename: filename=str(self.DESC[self.dataset_type])+'.csv' 
        header= 'Datetime,' + ','.join(self.feature_names) + ',Data'
        data = np.column_stack((np.array(str_date), self.X, self.y,))
        np.savetxt(filename,
                   data,
                   delimiter=',',
                   header=header,
                   fmt='%s',
                   comments='')

    def __str__(self):
        return 'Dataset type: %s, %s'\
                %(self.dataset_type,self.DESC[self.dataset_type])

if __name__=='__main__':
   import numpy as np
   from datetime import datetime

   X = np.random.rand(24,3)
   y = np.random.rand(24,1)
   X_standardizer = preprocessing.StandardScaler().fit(X)
   y_standardizer = preprocessing.StandardScaler().fit(y)
   dts = np.arange('2014-01-01T00:00','2014-01-02T00:00',\
                     dtype=('datetime64[h]')).astype(datetime)
   feature_names = ['Minute','Hour','DayOfWeek']
   test = Dataset(dataset_type='A',
                  X_s=X,
                  X_standardizer=X_standardizer,
                  y_s=y,
                  y_standardizer=y_standardizer,
                  dts=dts,
                  feature_names=feature_names)
   print test
   print test.X
   print test.X_s
   print test.y
   print test.y_s
   
