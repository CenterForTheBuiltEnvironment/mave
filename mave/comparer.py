"""
Quantifying the error between two arrays
Passed values can be python lists or numpy arrays,
but each must be the same length/shape

@author Paul Raftery <p.raftery@berkeley.edu>
"""
import math, os, pdb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class Comparer(object):
    DIGITS = 3
    def __init__(self, prediction, baseline):
        p = np.array(prediction).astype(np.float)
        b = np.array(baseline).astype(np.float)
        self.b_mean = np.mean(b)
        # error (overpredict is negative) 
        self.e = b-p
        # total biased error
        self.tbe = round(sum(self.e),self.DIGITS)
        # normalized mean bias error
        self.nmbe = round(100*sum(self.e)/self.b_mean/len(b),self.DIGITS)
        # root mean squared error
        self.rmse = math.sqrt(sum((self.e)**2)/len(b))
        # coefficient of root mean squared error
        self.cvrmse = round(100*(self.rmse)/self.b_mean,self.DIGITS)
        # r2
        self.r2 = round(1 - (sum(self.e**2)/sum((b - self.b_mean)**2)),
                                                               self.DIGITS)
        # check for zeroes before division
        if np.count_nonzero(b) < len(b):
            self.some_zeros = True
            b[b==0.0] = np.nan      
        else:
            self.some_zeros = False
        # normalized percentage error 
        self.npe = 100*self.e/b
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
        # returns a string idescribing how closely the arrays match each other
        rv = '\n\nNote that values are negative when the model predicts'
        rv += ' a higher value than the baseline (i.e. overprediction)'
        if self.some_zeros:
            rv += '\n\n*** There are zero values in the baseline data' + \
                  ' rendering some comparisons meaningless.\n' 
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
        return rv

class Plot(object):
    def __init__(self,baseline,prediction):
        pp = PdfPages('report.pdf')
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(range(len(baseline)),baseline, s=30, c='b',\
                    label='baseline',alpha=0.7)
        ax1.scatter(range(len(prediction)),prediction, s=30, c='r',\
                    label='prediction',alpha=0.7)
        plt.legend(loc='upper right')
        
        pp.savefig()
        pp.close()
   
if __name__=='__main__': 
    import pdb
    import numpy as np
    b = np.ones(10000,)*(np.random.random_sample(10000,)+0.5)
    p = np.random.random_sample(10000,)+0.5  
    c = Comparer(prediction=p,baseline=b)
    print c
    Plot(b,p)
