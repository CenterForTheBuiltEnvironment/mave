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
    def __init__(self, prediction, baseline, **kwargs):
        self.p = np.array(prediction).astype(np.float)
        self.b = np.array(baseline).astype(np.float)
        self.b_mean = np.mean(self.b)
        # error (overpredict is negative) 
        self.e = self.b-self.p
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
                  ' rendering some comparison metrics meaningless.\n' 
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

class Plot(object):
    def __init__(self, baseline, prediction, p_X, name_list,text, fname):
        if p_X is not None and name_list is not None:
            self.names = name_list
            self.X = np.core.records.fromarrays(p_X.transpose(),\
                                                   names=self.names)
        self.p = np.array(prediction).astype(np.float)
        self.b = np.array(baseline).astype(np.float)
        if np.count_nonzero(self.b) < len(self.b):
            self.b[self.b==0.0] = np.nan      
        npe = 100*(self.b-self.p)/self.b
        e = self.b-self.p
        with PdfPages('report_%s.pdf'%(fname,)) as pdf:
            #results text
            fig0 = plt.figure()
            plt.axis([0,10,0,10])
            data = text
            plt.text(0, 0, data, fontsize=8, family='serif', wrap=True)
            plt.axis('off')
            pdf.savefig(fig0)
            plt.close()

            #scatterplot
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(1,1,1)
            ax1.set_title('Post-retrofit baseline vs. prediction')
            ax1.set_ylabel('Energy consumption [in original units]')
            ax1.set_xlabel('Data points')
            ax1.plot(np.mean(self.b),label='baseline mean',
                     c='y',rasterized=True)
            ax1.plot(e,label='baseline - prediction',c='b',rasterized=True)
           # ax1.scatter(range(len(baseline)),self.b, s=20, c='b',\
           #             label='Baseline', edgecolors='none',alpha=0.3,\
           #             rasterized=True)
           # ax1.scatter(range(len(self.p)),self.p, s=20, c='y',\
           #             label='Prediction',edgecolors='none',alpha=0.3,\
           #             rasterized=True)
            plt.legend(loc='upper right',fontsize=8,markerscale=0.6)
            plt.xlim(0,1.1*len(self.p))
            if np.mean(self.b) > 0:
                plt.ylim(0,1.1*np.nanmax(self.b))
            pdf.savefig(fig1)
            plt.close()

            #snapshot of the first and last week
            first_idx = np.where(self.X['day_of_week']==0)[0][0]
            last_idx = np.where(self.X['day_of_week']==6)[0][-1]
            if first_idx == 0:
                first_idx = np.where(self.X['day_of_week']==1)[0][0]
                last_idx = np.where(self.X['day_of_week']==0)[0][-1]
                ran = (np.where(self.X['day_of_week']==2)[0][0]-first_idx)*7
            else:
                ran = (np.where(self.X['day_of_week']==1)[0][0]-first_idx)*7
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1,1,1)
            ax2.set_title('Snapshot of the first week of data')
            ax2.set_ylabel('Energy consumption [in original units]')
            ax2.set_xlabel('Data points')
            ax2.plot(range(int(ran)),self.b[first_idx:first_idx+ran],\
                       label='Baseline',c='b',\
                       rasterized=True)
            ax2.plot(range(int(ran)),self.p[first_idx:first_idx+ran],\
                       label='Prediction',c='y',
                       rasterized=True)
            plt.legend(loc='upper right',fontsize=8,markerscale=0.6)
            plt.xlim(0,ran)
            if np.mean(self.b) > 0:
                plt.ylim(0,1.1*np.nanmax(self.b[first_idx:first_idx+ran]))
            pdf.savefig(fig2)
            plt.close()

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(1,1,1)
            ax3.set_title('Snapshot of the last week of data')
            ax3.set_ylabel('Energy consumption [in original units]')
            ax3.set_xlabel('Data points')
            ax3.plot(range(int(ran)),self.b[last_idx-ran:last_idx],\
                       label='Baseline',c='b')
            ax3.plot(range(int(ran)),self.p[last_idx-ran:last_idx],\
                       label='Prediction',c='y')
            plt.legend(loc='upper right',fontsize=8,markerscale=0.6)
            plt.xlim(0,ran)
            if np.mean(self.b) > 0:
                plt.ylim(0,1.1*np.nanmax(self.b[first_idx:first_idx+ran]))
            pdf.savefig(fig3)
            plt.close()
            
            #find the error in predicting the monthly peak value
            monthly_peak_error = []
            max_month = np.amax(self.X['month'])
            if np.greater(max_month,self.X['month'][0]-1) and \
               np.greater(max_month,self.X['month'][-1]):
                if np.less(self.X['month'][0],self.X['month'][-1]+2):
                    months = range(1,13)
                else:
                    months = range(int(self.X['month'][0]), 13)+\
                             range(1,int(self.X['month'][-1]+1))
            else:
                months = range(int(self.X['month'][0]),\
                               int(self.X['month'][-1]+1))
            for i in months:
                try:
                    max_b = np.nanmax(np.extract(self.X['month']==i, self.b))
                    max_p = np.nanmax(np.extract(self.X['month']==i, self.p))
                    monthly_peak_error.append(100*(max_b-max_p)/max_b)
                except:
                    pass
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(1,1,1)
            ax4.set_title('Monthly peak error')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Normalized percentage error [%]')
            ax4.bar(np.array(months)-0.15,monthly_peak_error,0.3,
                    rasterized=True)
            x = [1,2,3,4,5,6,7,8,9,10,11,12]
            labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',\
                      'Sep','Oct','Nov','Dec']
            plt.xticks(range(1,13),labels)
            plt.xlim(0.5,12.5)
            pdf.savefig(fig4)
            plt.close()

            #boxplot of percentage error by the hour of day
            bpdata_hour = []
            for i in range(24):
                hour_b = np.extract(self.X['hour']==i,self.b)
                hour_p = np.extract(self.X['hour']==i,self.p)
                bpdata_hour.append(100*(hour_b-hour_p)/hour_b)
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(1,1,1)
            ax5.set_title('Error by the hour of day')
            ax5.set_xlabel('Hour of day')
            ax5.set_ylabel('Normalized percentage error [%]')
            plt.ylim(-100,100)
            ax5.boxplot(bpdata_hour)
            pdf.savefig(fig5)
            plt.close()

            #boxplot of error by the hour of day
            bpdata_hour = []
            for i in range(24):
                hour_b = np.extract(self.X['hour']==i,self.b)
                hour_p = np.extract(self.X['hour']==i,self.p)
                bpdata_hour.append(hour_b-hour_p)
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(1,1,1)
            ax5.set_title('Error by the hour of day')
            ax5.set_xlabel('Hour of day')
            ax5.set_ylabel('Error [in original units]')
            ax5.boxplot(bpdata_hour)
            pdf.savefig(fig5)
            plt.close()

            #boxplot of percentage error by the day of week
            bpdata_week = []
            for i in range(7):
                week_b = np.extract(self.X['day_of_week']==i, self.b)
                week_p = np.extract(self.X['day_of_week']==i, self.p)
                bpdata_week.append(100*(week_b-week_p)/week_b)
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(1,1,1)
            ax6.set_title('Error by the day of week')
            ax6.set_xlabel('Day of week')
            ax6.set_ylabel('Normalized percentage error [%]')
            plt.ylim(-100,100)
            ax6.boxplot(bpdata_week)
            pdf.savefig(fig6)
            plt.close()

            #boxplot of error by the day of week
            bpdata_week = []
            for i in range(7):
                week_b = np.extract(self.X['day_of_week']==i, self.b)
                week_p = np.extract(self.X['day_of_week']==i, self.p)
                bpdata_week.append(week_b-week_p)
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(1,1,1)
            ax6.set_title('Error by the day of week')
            ax6.set_xlabel('Day of week')
            ax6.set_ylabel('Error [in original units]')
            ax6.boxplot(bpdata_week)
            pdf.savefig(fig6)
            plt.close()

            #OAT vs. percentage error
            if 'OutsideDryBulbTemperature' in self.names:
                fig7 = plt.figure()
                ax7 = fig7.add_subplot(1,1,1)
                ax7.set_title('Error by outside dry bulb temperature')
                ax7.set_xlabel('Outside Dry Bulb Temperature ($^\circ$C)')
                ax7.set_ylabel('Normalized percentage error [%]')
                max_temp = int(np.nanmax(self.X['OutsideDryBulbTemperature']))
                min_temp = int(np.nanmin(self.X['OutsideDryBulbTemperature']))
                if max_temp > 45:
                    max_temp = 45
                if min_temp < -35:
                    min_temp = -35
                perror_temp = []
                error_temp = []
                for i in range(min_temp,max_temp):
                    perror_temp.append(npe[np.nonzero(\
                                    (self.X['OutsideDryBulbTemperature']>=i) &\
                                    (self.X['OutsideDryBulbTemperature']<i+1))])
                    error_temp.append(e[np.nonzero(\
                                    (self.X['OutsideDryBulbTemperature']>=i) &\
                                    (self.X['OutsideDryBulbTemperature']<i+1))])
                ax7.boxplot(perror_temp)
                plt.ylim(-100,100)
                pdf.savefig(fig7)
                plt.close()
             
            #OAT vs. error
                fig7s = plt.figure()
                ax7s = fig7s.add_subplot(1,1,1)
                ax7s.set_title('Error by outside dry bulb temperature')
                ax7s.set_xlabel('Outside Dry Bulb Temperature ($^\circ$C)')
                ax7s.set_ylabel('Error [in original units]')
                ax7s.boxplot(error_temp)
                pdf.savefig(fig7s)
                plt.close()

            #holiday vs. percentage error
            if 'holiday' in self.names:
                holiday_perror=[]
                for i in range(0,4):
                    holiday_b = np.extract(self.X['holiday']==i, self.b)
                    holiday_p = np.extract(self.X['holiday']==i, self.p)
                    holiday_perror.append(100*(holiday_b-holiday_p)/holiday_b)
                fig8 = plt.figure()
                ax8 = fig8.add_subplot(1,1,1)
                ax8.set_title('Error by closeness to holidays')
                ax8.set_xlabel('Days away from holidays')
                ax8.set_ylabel('Normalized percentage error [%]')
                ax8.boxplot(holiday_perror)
                x = [0,1,2,3,4]
                label = ['','>=3 Day','2 Day','1 Days','0 Days']
                plt.xticks(x,label)
                plt.ylim(-100,100)
                pdf.savefig(fig8)
                plt.close()
            
            #holiday vs. error
                holiday_error=[]
                for i in range(0,4):
                    holiday_b = np.extract(self.X['holiday']==i, self.b)
                    holiday_p = np.extract(self.X['holiday']==i, self.p)
                    holiday_error.append(holiday_b-holiday_p)
                fig8 = plt.figure()
                ax8 = fig8.add_subplot(1,1,1)
                ax8.set_title('Error by closeness to holidays')
                ax8.set_xlabel('Days away from holidays')
                ax8.set_ylabel('Error [in original units]')
                ax8.boxplot(holiday_error)
                x = [0,1,2,3,4]
                label = ['','>=3 Days','2 Days','1 Day','0 Day']
                plt.xticks(x,label)
                pdf.savefig(fig8)
                plt.close()
           # Pivottable generation, in the case, hours are the rows and day
           # of week are the columns.     
           # rows,row_pos = np.unique(self.X[:,1],return_inverse=True)
           # cols,col_pos = np.unique(self.X[:,2],return_inverse=True)
           # pt_hour_b = np.zeros((len(rows),len(cols)),dtype=self.X.dtype)
           # pt_hour_p = pt_hour_b
           # pt_hour_b[row_pos,col_pos] = baseline
           # pt_hour_p[row_pos,col_pos] = prediction

            d = pdf.infodict()
            d['Title'] = "Report"
            d['Author'] = "repor automatically generated by Mave"

   
if __name__=='__main__': 
    import pdb
    import numpy as np
    from datetime import datetime
    def pdt(dt):
        rv = float(dt.minute),float(dt.hour),float(dt.weekday()),float(dt.month)
        return rv
    b = np.ones(8759,)*(np.random.random_sample(8759,)+0.5)
    p = np.random.random_sample(8759,)+0.5
    dts = np.arange('2010-01-01T00:00','2010-12-31T23:59',\
                     dtype=('datetime64[h]')).astype(datetime)
    vect_pdt = np.vectorize(pdt)
    dt = np.column_stack(vect_pdt(dts))
    names = ['minute','hour','day_of_week',\
             'month','holiday','OutsideDryBulbTemperature']  
    holiday = np.random.randint(4,size = 8759)
    db = np.random.random_sample(8759,)+10
    dt=np.column_stack((dt,holiday,db))
    c = Comparer(prediction=p,baseline=b,p_X=dt,names=names,plot=True)
    print c
