"""
Quantifying the error between two arrays
Passed values can be python lists, numpy arrays,
or mave dataset.Dataset objects but the comparison 
and baseline data must be the same length/shape

@author Paul Raftery <p.raftery@berkeley.edu>
"""
import math, os, pdb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class Comparer(object):
    DIGITS = 3
    def __init__(self, comparison, baseline):
        # extract unstandardized results ('y' field) 
        # from Dataset objects if passed as args instead of arrays
        try: 
            self.c= np.array(comparison.y).astype(np.float)
        except:
            self.c = np.array(comparison).astype(np.float)
        try:
            self.b = np.array(baseline.y).astype(np.float)
        except:    
            self.b = np.array(baseline).astype(np.float)
        self.b_mean = np.mean(self.b)
        # error (overpredict is negative) 
        self.e = self.b-self.c
        # total biased error
        self.tbe = round(sum(self.e),self.DIGITS)
        # normalized mean bias error
        self.nmbe = round(100*sum(self.e)/self.b_mean/len(self.b),self.DIGITS)
        # root mean squared error
        self.rmse = math.sqrt(sum((self.e)**2)/len(self.b))
        # coefficient of root mean squared error
        self.cvrmse = round(100*(self.rmse)/self.b_mean,self.DIGITS)
        # r2
        self.r2 = round(1 - (sum(self.e**2)/sum((self.b - self.b_mean)**2)),
                                                               self.DIGITS)
        # check for zeroes before division
        if np.count_nonzero(self.b) < len(self.b):
            self.some_zeros = True
            self.b[self.b==0.0] = np.nan      
        else:
            self.some_zeros = False
        # normalized percentage error 
        self.npe = 100*self.e/self.b
        self.npe_min = round(np.min(self.npe),self.DIGITS)
        self.npe_max= round(np.max(self.npe),self.DIGITS)
        self.npe_mean = round(np.mean(self.npe),self.DIGITS)
        self.npe_median = round(np.median(self.npe),self.DIGITS)
        self.npe_10th = round(np.percentile(self.npe,10),self.DIGITS)
        self.npe_25th = round(np.percentile(self.npe,25),self.DIGITS)
        self.npe_75th = round(np.percentile(self.npe,75),self.DIGITS)
        self.npe_90th = round(np.percentile(self.npe,90),self.DIGITS)
        # normalized absolute percentage error
        self.nape = abs(self.npe)
        # mean absolute percentage error
        self.mape = round(np.mean(self.nape),self.DIGITS)

    def __str__(self):
        # returns a string describing how closely the arrays match each other
        rv = '\n\nNote that values are negative when the model predicts'
        rv += ' a higher value than the baseline (i.e. savings)'
        if self.some_zeros:
            rv += '\n\n*** There are zero values in the baseline data' + \
                  ' rendering some typical comparison metrics meaningless.\n' 
        rv += '\nTotal Biased Error: %s [in original units]'%str(self.tbe)
        rv +='\nNormalized Mean Bias Error: %s %%'%str(self.nmbe)
        rv +='\nMean Absolute Percent Error: %s %%'%str(self.mape)
        rv +='\nCVRMSE: %s %%'%str(self.cvrmse)
        rv +='\nR2: %s '%str(self.r2)
        rv +='\n\nNormalized error, min: %s %%'%str(self.npe_min)
        rv +='\nNormalized error, 10th percentile: %s %%'%str(self.npe_10th)
        rv +='\nNormalized error, 25th percentile: %s %%'%str(self.npe_25th)
        rv +='\nNormalized error, median: %s %%'%str(self.npe_median)
        rv +='\nNormalized error, mean: %s %%'%str(self.npe_mean)
        rv +='\nNormalized error, 75th percentile: %s %%'%str(self.npe_75th)
        rv +='\nNormalized error, 90th percentile: %s %%'%str(self.npe_90th)
        rv +='\nNormalized error, max: %s %%\n'%str(self.npe_max)
        rv +="\nThe total error is: %s [in original units]"%round(self.tbe,2)
        return rv

if __name__=='__main__': 
    import numpy as np

    b = np.ones(8759,)*(np.random.random_sample(8759,)+0.5)
    c = np.random.random_sample(8759,)+0.5
    comparer = Comparer(comparison=c, baseline=b)
    print comparer

