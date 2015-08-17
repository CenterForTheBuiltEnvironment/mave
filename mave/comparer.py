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
        self.b_mean = np.mean(b)
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
        self.nmbe = 100*sum(self.e)/self.b_mean/len(b)
        # root mean squared error
        self.rmse = math.sqrt(sum((self.e)**2)/len(b))
        # coefficient of root mean squared error
        self.cvrmse = 100*(self.rmse)/self.b_mean
        # r2
        self.r2 = 1 - (sum(self.e**2)/sum((b - self.b_mean)**2))

    def __str__(self):
        # returns a string idescribing how closely the arrays match each other
        rv ='\nNormalized Mean Bias Error: %s %%'%str(self.nmbe)
        rv +='\nMean Absolute Percent Error: %s %%'%str(self.mape)
        rv +='\nCVRMSE: %s %%'%str(self.cvrmse)
        rv +='\nR2: %s %%'%str(self.r2)
        rv +='\nNormalized error, min: %s %%'%str(self.npe_min)
        rv +='\nNormalized error, 10th percentile: %s %%'%str(self.npe_10th)
        rv +='\nNormalized error, 25th percentile: %s %%'%str(self.npe_25th)
        rv +='\nNormalized error, median: %s %%'%str(self.npe_median)
        rv +='\nNormalized error, mean: %s %%'%str(self.npe_mean)
        rv +='\nNormalized error, 75th percentile: %s %%'%str(self.npe_75th)
        rv +='\nNormalized error, 90th percentile: %s %%'%str(self.npe_90th)
        rv +='\nNormalized error, max: %s %%\n'%str(self.npe_max)
        return rv

if __name__=='__main__': 
    import pdb
    import numpy as np
    b = np.ones(10000,)*(np.random.random_sample(10000,)+0.5)
    p = np.random.random_sample(10000,)+0.5  
    c = Comparer(prediction=p,baseline=b)
    print c


