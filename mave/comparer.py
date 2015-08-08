"""
Quantifying the error between two arrays
Passed values can be python lists or numpy arrays,
but each must be the same length/shape

@author Paul Raftery <p.raftery@berkeley.edu>
"""
import math, os, pdb
import numpy as np

class Comparer(object):
  
  def __init__(self, prediction, baseline):
    p = np.array(prediction).astype(np.float)
    b = np.array(baseline).astype(np.float)
    # error (overpredict is negative) 
    self.e = b-p
    # normalized percentage error 
    self.npe = 100*self.e/b
    self.npe_min = np.min(self.npe)
    self.npe_max= np.max(self.npe)
    self.npe_mean = np.mean(self.npe)
    self.npe_median = np.median(self.npe)
    self.npe_10th = np.percentile(self.npe,10)
    self.npe_25th = np.percentile(self.npe,25)
    self.npe_75th = np.percentile(self.npe,75)
    self.npe_90th = np.percentile(self.npe,90)
    # normalized absolute percentage error
    self.nape = abs(self.npe)
    # mean absolute percentage error
    self.mape = np.mean(self.nape)
    # normalized mean bias error
    self.nmbe = 100*sum(self.e)/np.mean(b)/len(b)
    # root mean squared error
    self.rmse = math.sqrt(sum((self.e)**2)/len(b))
    # coefficient of root mean squared error
    self.cvrmse = 100*(self.rmse)/np.mean(b)

  def print_overview(self):
    # print an overview of how well the model performs
    print '\nNormalized Mean Bias Error: %s %%'%str(self.nmbe)
    print 'Mean Absolute Percent Error: %s %%'%str(self.mape)
    print 'CVRMSE: %s %%'%str(self.cvrmse)
    print 'Normalized error, min: %s %%'%str(self.npe_min)
    print 'Normalized error, 10th percentile: %s %%'%str(self.npe_10th)
    print 'Normalized error, 25th percentile: %s %%'%str(self.npe_25th)
    print 'Normalized error, median: %s %%'%str(self.npe_median)
    print 'Normalized error, mean: %s %%'%str(self.npe_mean)
    print 'Normalized error, 75th percentile: %s %%'%str(self.npe_75th)
    print 'Normalized error, 90th percentile: %s %%'%str(self.npe_90th)
    print 'Normalized error, max: %s %%\n'%str(self.npe_max)

if __name__=='__main__': 
  import pdb
  import numpy as np
  b = np.ones(10000,)
  p = np.random.random_sample(10000,)+0.5  
  c = Comparer(prediction=p,baseline=b)
  c.print_overview()


