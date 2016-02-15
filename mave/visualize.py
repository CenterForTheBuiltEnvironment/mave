"""
Visualize the difference between two datasets 
and store as pdf.

@author Paul Raftery <p.raftery@berkeley.edu>
@author Taoning Wang <taoning@berkeley.edu>
"""
import math, os, pdb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class Visualize(object):
    def __init__(self, comparison_dataset, baseline_dataset, text, fname):
        if baseline_dataset.X is not None and \
           baseline_dataset.feature_names is not None:
            self.features = baseline_dataset.feature_names
            self.X = np.core.records.fromarrays(baseline_dataset.X.transpose(),\
                                                   names=self.features)
        self.c = np.array(comparison_dataset.y).astype(np.float)
        self.b = np.array(baseline_dataset.y).astype(np.float)
        if np.count_nonzero(self.b) < len(self.b):
            self.b[self.b==0.0] = np.nan      
        npe = 100*(self.b-self.c)/self.b
        e = self.b-self.c
        with PdfPages('report_%s.pdf'%(fname,)) as pdf:
            pdb.set_trace()
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
            ax1.set_title('Post-retrofit baseline vs. comparison')
            ax1.set_ylabel('Energy consumption [in original units]')
            ax1.set_xlabel('Data points')
            ax1.plot(np.mean(self.b),label='baseline mean',
                     c='y',rasterized=True)
            ax1.plot(e,label='baseline - comparison',c='b',rasterized=True)
           # ax1.scatter(range(len(baseline)),self.b, s=20, c='b',\
           #             label='Baseline', edgecolors='none',alpha=0.3,\
           #             rasterized=True)
           # ax1.scatter(range(len(self.c)),self.c, s=20, c='y',\
           #             label='Prediction',edgecolors='none',alpha=0.3,\
           #             rasterized=True)
            plt.legend(loc='upper right',fontsize=8,markerscale=0.6)
            plt.xlim(0,1.1*len(self.c))
            if np.mean(self.b) > 0:
                plt.ylim(0,1.1*np.nanmax(self.b))
            pdf.savefig(fig1)
            plt.close()

            #snapshot of the first and last week
            first_idx = np.where(self.X['DayOfWeek']==0)[0][0]
            last_idx = np.where(self.X['DayOfWeek']==6)[0][-1]
            pdb.set_trace()
            if first_idx == 0:
                first_idx = np.where(self.X['DayOfWeek']==1)[0][0]
                last_idx = np.where(self.X['DayOfWeek']==0)[0][-1]
                ran = (np.where(self.X['DayOfWeek']==2)[0][0]-first_idx)*7
            else:
                ran = (np.where(self.X['DayOfWeek']==1)[0][0]-first_idx)*7
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1,1,1)
            ax2.set_title('Snapshot of the first week of data')
            ax2.set_ylabel('Energy consumption [in original units]')
            ax2.set_xlabel('Data points')
            ax2.plot(range(int(ran)),self.b[first_idx:first_idx+ran],\
                       label='Baseline',c='b',\
                       rasterized=True)
            ax2.plot(range(int(ran)),self.c[first_idx:first_idx+ran],\
                       label='Prediction',c='y',
                       rasterized=True)
            plt.legend(loc='upper right',fontsize=8,markerscale=0.6)
            plt.xlim(0,ran)
            if np.mean(self.b) > 0:
                plt.ylim(0,1.1*np.nanmax(self.b[first_idx:first_idx+ran]))
            pdf.savefig(fig2)
            plt.close()

            pdb.set_trace()
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(1,1,1)
            ax3.set_title('Snapshot of the last week of data')
            ax3.set_ylabel('Energy consumption [in original units]')
            ax3.set_xlabel('Data points')
            ax3.plot(range(int(ran)),self.b[last_idx-ran:last_idx],\
                       label='Baseline',c='b')
            ax3.plot(range(int(ran)),self.c[last_idx-ran:last_idx],\
                       label='Prediction',c='y')
            plt.legend(loc='upper right',fontsize=8,markerscale=0.6)
            plt.xlim(0,ran)
            if np.mean(self.b) > 0:
                plt.ylim(0,1.1*np.nanmax(self.b[first_idx:first_idx+ran]))
            pdf.savefig(fig3)
            plt.close()
            
            #find the error in predicting the Monthly peak value
            Monthly_peak_error = []
            max_Month = np.amax(self.X['Month'])
            if np.greater(max_Month,self.X['Month'][0]-1) and \
               np.greater(max_Month,self.X['Month'][-1]):
                if np.less(self.X['Month'][0],self.X['Month'][-1]+2):
                    Months = range(1,13)
                else:
                    Months = range(int(self.X['Month'][0]), 13)+\
                             range(1,int(self.X['Month'][-1]+1))
            else:
                Months = range(int(self.X['Month'][0]),\
                               int(self.X['Month'][-1]+1))
            for i in Months:
                try:
                    max_b = np.nanmax(np.extract(self.X['Month']==i, self.b))
                    max_c = np.nanmax(np.extract(self.X['Month']==i, self.c))
                    Monthly_peak_error.append(100*(max_b-max_c)/max_b)
                except:
                    pass
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(1,1,1)
            ax4.set_title('Monthly peak error')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Normalized percentage error [%]')
            ax4.bar(np.array(Months)-0.15,Monthly_peak_error,0.3,
                    rasterized=True)
            x = [1,2,3,4,5,6,7,8,9,10,11,12]
            labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',\
                      'Sep','Oct','Nov','Dec']
            plt.xticks(range(1,13),labels)
            plt.xlim(0.5,12.5)
            pdf.savefig(fig4)
            plt.close()
            pdb.set_trace()

            #boxplot of percentage error by the Hour of day
            bcdata_Hour = []
            for i in range(24):
                Hour_b = np.extract(self.X['Hour']==i,self.b)
                Hour_c = np.extract(self.X['Hour']==i,self.c)
                bcdata_Hour.append(100*(Hour_b-Hour_c)/Hour_b)
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(1,1,1)
            ax5.set_title('Error by the Hour of day')
            ax5.set_xlabel('Hour of day')
            ax5.set_ylabel('Normalized percentage error [%]')
            plt.ylim(-100,100)
            ax5.boxplot(bcdata_Hour)
            pdf.savefig(fig5)
            plt.close()

            #boxplot of error by the Hour of day
            bcdata_Hour = []
            for i in range(24):
                Hour_b = np.extract(self.X['Hour']==i,self.b)
                Hour_c = np.extract(self.X['Hour']==i,self.c)
                bcdata_Hour.append(Hour_b-Hour_c)
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(1,1,1)
            ax5.set_title('Error by the Hour of day')
            ax5.set_xlabel('Hour of day')
            ax5.set_ylabel('Error [in original units]')
            ax5.boxplot(bcdata_Hour)
            pdf.savefig(fig5)
            plt.close()

            #boxplot of percentage error by the day of week
            bcdata_week = []
            for i in range(7):
                week_b = np.extract(self.X['DayOfWeek']==i, self.b)
                week_c = np.extract(self.X['DayOfWeek']==i, self.c)
                bcdata_week.append(100*(week_b-week_c)/week_b)
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(1,1,1)
            ax6.set_title('Error by the day of week')
            ax6.set_xlabel('Day of week')
            ax6.set_ylabel('Normalized percentage error [%]')
            plt.ylim(-100,100)
            ax6.boxplot(bcdata_week)
            pdf.savefig(fig6)
            plt.close()

            pdb.set_trace()
            #boxplot of error by the day of week
            bcdata_week = []
            for i in range(7):
                week_b = np.extract(self.X['DayOfWeek']==i, self.b)
                week_p = np.extract(self.X['DayOfWeek']==i, self.c)
                bcdata_week.append(week_b-week_c)
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(1,1,1)
            ax6.set_title('Error by the day of week')
            ax6.set_xlabel('Day of week')
            ax6.set_ylabel('Error [in original units]')
            ax6.boxplot(bcdata_week)
            pdf.savefig(fig6)
            plt.close()

            #OAT vs. percentage error
            if 'OutsideDryBulbTemperature' in self.features:
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
                cerror_temp = []
                error_temp = []
                for i in range(min_temp,max_temp):
                    cerror_temp.append(npe[np.nonzero(\
                                    (self.X['OutsideDryBulbTemperature']>=i) &\
                                    (self.X['OutsideDryBulbTemperature']<i+1))])
                    error_temp.append(e[np.nonzero(\
                                    (self.X['OutsideDryBulbTemperature']>=i) &\
                                    (self.X['OutsideDryBulbTemperature']<i+1))])
                ax7.boxplot(cerror_temp)
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

            pdb.set_trace()
            #Holiday vs. percentage error
            if 'Holiday' in self.features:
                Holiday_cerror=[]
                for i in range(0,4):
                    Holiday_b = np.extract(self.X['Holiday']==i, self.b)
                    Holiday_c = np.extract(self.X['Holiday']==i, self.c)
                    Holiday_cerror.append(100*(Holiday_b-Holiday_c)/Holiday_b)
                fig8 = plt.figure()
                ax8 = fig8.add_subplot(1,1,1)
                ax8.set_title('Error by closeness to Holidays')
                ax8.set_xlabel('Days away from Holidays')
                ax8.set_ylabel('Normalized percentage error [%]')
                ax8.boxplot(Holiday_cerror)
                x = [0,1,2,3,4]
                label = ['','>=3 Day','2 Day','1 Days','0 Days']
                plt.xticks(x,label)
                plt.ylim(-100,100)
                pdf.savefig(fig8)
                plt.close()
            
            #Holiday vs. error
                Holiday_error=[]
                for i in range(0,4):
                    Holiday_b = np.extract(self.X['Holiday']==i, self.b)
                    Holiday_c = np.extract(self.X['Holiday']==i, self.c)
                    Holiday_error.append(Holiday_b-Holiday_c)
                fig8 = plt.figure()
                ax8 = fig8.add_subplot(1,1,1)
                ax8.set_title('Error by closeness to Holidays')
                ax8.set_xlabel('Days away from Holidays')
                ax8.set_ylabel('Error [in original units]')
                ax8.boxplot(Holiday_error)
                x = [0,1,2,3,4]
                label = ['','>=3 Days','2 Days','1 Day','0 Day']
                plt.xticks(x,label)
                pdf.savefig(fig8)
                plt.close()
           # Pivottable generation, in the case, Hours are the rows and day
           # of week are the columns.     
           # rows,row_pos = np.unique(self.X[:,1],return_inverse=True)
           # cols,col_pos = np.unique(self.X[:,2],return_inverse=True)
           # pt_Hour_b = np.zeros((len(rows),len(cols)),dtype=self.X.dtype)
           # pt_Hour_p = pt_Hour_b
           # pt_Hour_b[row_pos,col_pos] = baseline
           # pt_Hour_p[row_pos,col_pos] = comparison

            d = pdf.infodict()
            d['Title'] = "Report"
            d['Author'] = "repor automatically generated by Mave"

   
if __name__=='__main__': 
    import pdb
    import numpy as np
    from dataset import Dataset
    from datetime import datetime
    from sklearn import preprocessing

    b = np.ones(8759,)*(np.random.random_sample(8759,)+0.5)
    c = np.random.random_sample(8759,)+0.5

    def pdt(dt):
        rv = float(dt.minute),float(dt.hour),float(dt.weekday()),float(dt.month)
        return rv
    dts = np.arange('2010-01-01T00:00','2010-12-31T23:59',\
                     dtype=('datetime64[h]')).astype(datetime)
    vect_pdt = np.vectorize(pdt)
    dt = np.column_stack(vect_pdt(dts))
    features = ['Minute','Hour','DayOfWeek',\
                'Month','Holiday','OutsideDryBulbTemperature']  
    Holiday = np.random.randint(4,size = 8759)
    db = np.random.random_sample(8759,)+10
    X=np.column_stack((dt,Holiday,db))
    X_standardizer = preprocessing.StandardScaler().fit(X)
    y_standardizer = preprocessing.StandardScaler().fit(b)
    baseline_dataset = Dataset(dataset_type='A',
                       X=X,
                       X_standardizer=X_standardizer,
                       y=b,
                       y_standardizer=y_standardizer,
                       dts=dts,
                       feature_names=features)
    print baseline_dataset
    y_c_standardizer = preprocessing.StandardScaler().fit(c)
    comparison_dataset = Dataset(dataset_type='B',
                         X=X,
                         X_standardizer=X_standardizer,
                         y=c,
                         y_standardizer=y_c_standardizer,
                         dts=dts,
                         feature_names=features)
    print comparison_dataset
    text= """
===== Pre-retrofit model training summary =====
=== Selected model ===
Best cross validation score on training data: 0.75904349278
Best model:
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=8,
           max_features=3, max_leaf_nodes=None, min_samples_leaf=25,
           min_samples_split=88, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
The relative importances of input features are:
[ 0.00148272  0.6377029   0.07292095  0.02694483  0.00892741  0.1343837
  0.07458231  0.04305519]
"""
    p = Visualize(comparison_dataset=comparison_dataset,
                  baseline_dataset=baseline_dataset,
                  text=text,
                  fname='test')

    print '\ndone\n'
