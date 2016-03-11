"""
Quantifying the error between two arrays
Passed values can be python lists, numpy arrays,
or mave dataset object but the comparison 
and baseline data must be the same length/shape

@author Paul Raftery <p.raftery@berkeley.edu>
"""
import math, os, pdb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class Comparer(object):
    def __init__(self, comparison, baseline):
        # extract unstandardized results ('y' field) 
        # from Dataset objects if passed as args instead of arrays
        try: 
            self.c = np.array(comparison.y).astype(np.float)
        except:
            self.c = np.array(comparison).astype(np.float)
        try:
            self.b = np.array(baseline.y).astype(np.float)
        except:    
            self.b = np.array(baseline).astype(np.float)
        # amount of data
        self.n = len(self.b)
        # average of measured data
        self.b_mean = np.mean(self.b)
        # error (overpredict is negative) 
        self.e = self.b-self.c
        # total biased error
        self.tbe = np.sum(self.e)
        # normalized mean bias error
        self.nmbe = 100*np.sum(self.e)/self.b_mean/self.n
        # root mean squared error
        self.rmse = math.sqrt(np.sum((self.e)**2)/self.n)
        # coefficient of root mean squared error
        self.cvrmse = 100*(self.rmse)/self.b_mean
        # r2
        self.r2 = 1 - (np.sum(self.e**2) / np.sum((self.b - self.b_mean)**2))
        # check for zeroes before division
        if np.count_nonzero(self.b) < self.n:
            self.some_zeros = True
            self.b[self.b==0.0] = np.nan      
        else:
            self.some_zeros = False
        # normalized percentage error 
        self.npe = 100*self.e/self.b
        self.npe_min = np.min(self.npe)
        self.npe_max= np.max(self.npe)
        self.npe_mean = np.mean(self.npe)
        self.npe_median = np.median(self.npe)
        self.npe_10th = np.percentile(self.npe,10)
        self.npe_25th = np.percentile(self.npe,25)
        self.npe_75th = np.percentile(self.npe,75)
        self.npe_90th = np.percentile(self.npe,90)
        # standard deviation of percentage error
        self.npe_std = np.std(self.npe)
        # normalized absolute percentage error
        self.nape = abs(self.npe)
        # mean absolute percentage error
        self.mape = np.mean(self.nape)

    def __str__(self):
        # returns a string describing how closely the arrays match each other
        rv ='\n\nNegative indicates prediction > baseline (i.e. savings)'
        rv +='\nTotal Biased Error: %0.3f [in original units]'%self.tbe
        if self.some_zeros:
            rv +='\n\nNote: There are zero values in the baseline data' + \
                 ' rendering some typical comparison metrics meaningless.\n' 
        rv +='\nNormalized Mean Bias Error: %0.3f%%'%self.nmbe
        rv +='\nMean Absolute Percent Error: %0.3f%%'%self.mape
        rv +='\nCVRMSE: %0.3f%%'%self.cvrmse
        rv +='\nR2: %0.3f '%self.r2
        rv +='\n\nDistribution of normalized errors:'
        rv +='\n  minimum:    %+0.3f%%'%self.npe_min
        rv +='\n  10th %%ile:  %+0.3f%%'%self.npe_10th
        rv +='\n  25th %%ile:  %+0.3f%%'%self.npe_25th
        rv +='\n  median:     %+0.3f%%'%self.npe_median
        rv +='\n  75th %%ile:  %+0.3f%%'%self.npe_75th
        rv +='\n  90th %%ile:  %+0.3f%%'%self.npe_90th
        rv +='\n  maximum:    %+0.3f%%'%self.npe_max
        rv +='\n\n  mean:       %+0.3f%%'%self.npe_mean
        rv +='\n  std. dev.:  %0.3f%%'%self.npe_std
        rv +='\n  count:      %i'%self.n
        return rv

if __name__=='__main__': 
    import numpy as np

    b = np.ones(8759,)*(np.random.random_sample(8759,)+10.0)
    c = np.random.random_sample(8759,)+10.5
    comparer = Comparer(comparison=c, baseline=b)
    print comparer

