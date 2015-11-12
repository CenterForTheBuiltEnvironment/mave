"""
Building Energy Prediction

This software reads an input file (a required argument) containing 
building energy data in a format similar to example file. 
It then trains a model and estimates the error associated
with predictions using the model.

@author Paul Raftery <p.raftery@berkeley.edu>
@author Tyler Hoyt <thoyt@berkeley.edu>
"""

import pdb
import csv
import os
import cPickle as pickle
import dateutil.parser
import numpy as np
import pprint
from math import sqrt
from datetime import datetime, timedelta
from sklearn import preprocessing, cross_validation, metrics
from holidays import holidays
import trainers 
import comparer
import location

class Preprocessor(object):

    HISTORICAL_DATA_COLUMN_NAMES = ['OutsideDryBulbTemperature']
    TARGET_COLUMN_NAMES = ['EnergyConsumption']
    IGNORE_TAG = -1
    PRE_DATA_TAG = 0
    POST_DATA_TAG = 1

    def __init__(self, 
                 input_file, 
                 verbose=False,
                 use_holidays=True,
                 use_month=False,
                 start_frac=0.0,
                 end_frac=1.0,
                 changepoints=None,
                 test_size=0.25,
                 timestamp_format='%Y-%m-%d%T%H%M',
                 datetime_column_name = 'LocalDateTime',
                 holiday_keys = ['USFederal'],
                 dayfirst = False,
                 yearfirst = False,
                 zipcode = None,
                 hist_db_column_names = ['OutsideDryBulbTemperature'],
                 hist_dp_column_names = None,
                 target_column_names = ['EnergyConsumption'],
                 remove_outliers = 'SingleValue',
                 X_standardizer = None,
                 **kwargs):
        self.timestamp_format = timestamp_format    
        self.datetime_column_name = datetime_column_name
        self.holiday_keys = holiday_keys
        self.use_holidays = use_holidays
        self.use_month = use_month
        self.input_file = input_file
        self.verbose = verbose
        self.X_standardizer = X_standardizer
        # process the headers
        self.headers, self.named_cols = self.process_headers()

        # identify holidays to use (if any)
        self.holidays = set([])
        if use_holidays:
            for key in self.holiday_keys:
                self.holidays = self.holidays.union(holidays[key])

        # read in the input data
        data = np.genfromtxt(self.input_file, 
                             delimiter=',',
                             dtype=None, 
                             skip_header=len(self.headers)-1, 
                             usecols=self.named_cols,
                             names=True, 
                             missing_values='NA')
        # shrink the input data by start_frac and end_frac
        data_L = len(data)
        start_index = int(start_frac * data_L)
        end_index = int(end_frac * data_L)
        data = data[ start_index : end_index ]
        # parse datetimes
        dcn = self.datetime_column_name
        try: 
            dts = map(lambda d: datetime.strptime(d,
                                                  self.timestamp_format),
                                                   data[dcn])
        except ValueError:
            dts = map(lambda d: dateutil.parser.parse(d, 
                                                      dayfirst=dayfirst,
                                                      yearfirst=yearfirst),
                                                       data[dcn])
        dtypes = data.dtype.descr
        dtypes[0] = dtypes[0][0], '|S20' # force 20 char strings for datetimes
        for i in range(1,len(dtypes)):
            dtypes[i] = dtypes[i][0], 'f8' # parse all other data as float
        data = data.astype(dtypes)

        data, dts, self.interval_seconds, self.vals_per_hr = \
            self.standardize_datetimes(data, dts)
        vectorized_process_datetime = np.vectorize(self.process_datetime)
        d = np.column_stack(vectorized_process_datetime(dts))

        # add other (non datetime related) input features
        data, target_col_ind = self.append_input_features(data, d,\
                                                 hist_db_column_names,\
                                                 target_column_names)
        # remove data
        self.X, self.y, self.dts = \
            self.clean_data(data, dts, target_col_ind, remove_outliers)
        # ensure that the datetimes match the input features
        if (self.X[:,0] != np.array([dt.minute for dt in self.dts])).any() or \
            (self.X[:,1] != np.array([dt.hour for dt in self.dts])).any():
            raise Error(" - The datetimes in the datetimes array do not \
                match those in the input features array")
        self.cps = self.changepoint_feature(changepoints=changepoints, **kwargs)
        self.split_dataset(test_size=test_size)

    def process_headers(self):
        # reads up to the first 100 lines of self.input_file and returns
        # the headers and the column names 
        reader = csv.reader(self.input_file, delimiter=',')
        headers = []
        for _ in range(100):
            row = reader.next()
            headers.append(row)
            if len(row)>0: 
                if self.datetime_column_name in row: 
                    named_cols = tuple(np.where(np.array(row) !='')[0])
                    break
        self.input_file.seek(0) # rewind the file 
        return headers, named_cols 

    def clean_data(self,
                   data, 
                   datetimes, 
                   target_col_ind, 
                   remove_outliers='SingleValue'):
        # remove any row with missing data, identified by nan
        keep_inds = ~np.isnan(data).any(axis=1)
        num_to_del = len(keep_inds[~keep_inds])
        if num_to_del > 0: 
            datetimes = datetimes[keep_inds]
            data = data[keep_inds]
        # split the data into input and target arrays
        y = data[:,target_col_ind]
        X = np.hstack((data[:,:target_col_ind], data[:,target_col_ind+1:]))
        # remove outliers
        if remove_outliers == 'SingleValue':
            keep_inds = self.is_single_value_outlier(y, med_diff_multiple=100)
        elif remove_outliers == 'MultipleValues':
            keep_inds = self.is_outlier(y, threshold=10)
        else:
            keep_inds = np.ones(len(y),dtype=bool)
        if self.verbose: 
            outliers = y[~keep_inds]
            outlier_ts =  map(lambda l: str(l),datetimes[~keep_inds])
            print '\nRemoved the following %s outlier values:\n%s'%\
                  (len(outliers),zip(outlier_ts,outliers))
        X = X[keep_inds]
        y = y[keep_inds]
        datetimes = datetimes[keep_inds]
        return X, y, datetimes

    def append_input_features(self, data, d0, hist_data_names,\
                              target_names, historical_data_points=2):
        column_names = data.dtype.names[1:] # no need to include datetime column
        d = d0
        for s in column_names:
            if s in hist_data_names:
                d = np.column_stack( (d, data[s]) )
                if historical_data_points > 0:
                    # create input features using historical data 
                    # at the intervals defined by n_vals_in_past_day
                    for v in range(1, historical_data_points + 1):
                        past_hours = v * 24 / (historical_data_points + 1)
                        n_vals = past_hours * self.vals_per_hr
                        past_data = np.roll(data[s], n_vals)
                        # for the first day in the file 
                        # there will be no historical data
                        # use the data from the next day as a rough estimate
                        past_data[0:n_vals] = past_data[24*self.vals_per_hr: \
                                                 24*self.vals_per_hr+n_vals ]
                        d = np.column_stack( (d, past_data) )
            elif not s in target_names:
                # just add the column as an input feature 
                # without historical data
                d = np.column_stack( (d, data[s]) )

        # add the target data
        split = d.shape[1]
        for s in target_names:
            if s in column_names:
                d = np.column_stack( (d, data[s]) )
        return d, split

    def is_single_value_outlier(self, y, med_diff_multiple=100):
        # id 2 highest and lowest values (ignoring nans)
        # id a single value as an outlier if the min or max is very far 
        # (> 100 times the median difference between values)
        # from the next nearest unique value
        keep_inds = np.ones(len(y), dtype=bool) 
        mx = np.amax(y)
        mn = np.amin(y)
        median_diff = np.median(abs(np.diff(y)))
        y_unique = np.unique(y)
        diff_to_max = np.diff(y_unique[np.argpartition(y_unique, -2)][-2:])[0]
        if abs(diff_to_max) > med_diff_multiple*median_diff:
            keep_inds = y < mx
        diff_to_min = np.diff(y_unique[np.argpartition(y_unique, 2)][:2])[0]
        if abs(diff_to_min) > med_diff_multiple*median_diff:
            keep_inds = y > mn
        return keep_inds

    def is_outlier(self, y, threshold=10):
        # outliers detected based on median absolute deviation according to
        # Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        # Handle Outliers", The ASQC Basic References in Quality Control:
        # Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        if len(y.shape) == 1:
            y = y[:,None]
        median = np.median(y, axis=0)
        diff = np.sum((y - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        keep_inds = modified_z_score <= threshold
        return keep_inds

    def standardize_datetimes(self, data, dts):
        # calculate the interval between datetimes
        intervals = [int((dts[i]-dts[i-1]).seconds) for i in range(1, len(dts))]
        median_interval = int(np.median(intervals))
        vals_per_hr = 3600 / median_interval
        assert (3600 % median_interval) == 0,  \
            'Median interval between datetimes must divide evenly into an hour'
        median_interval_minutes = median_interval/60
        assert (median_interval % 60) == 0,  \
            'Median interval between datetimes must be an even num of minutes'
        # round time datetimes according to the median_interval
        vectorized_round_datetime = np.vectorize(self.round_datetime)
        dts = vectorized_round_datetime(dts, median_interval_minutes)
        # remove duplicates and sorts datetimes
        dts, inds = np.unique(dts, return_index = True) 
        data = data[inds]
        # updates intervals after datetime rounding and duplicate removal
        intervals = [int((dts[i]-dts[i-1]).seconds) for i in range(1, len(dts))]
        row_length = len(data[0])
        # add datetimes and nans when there are gaps (based on median interval)
        gaps = np.greater(intervals, median_interval)
        gap_inds = np.nonzero(gaps)[0] # contains the left indices of the gaps
        NN = 0 # accumulate offset of gap indices as entries are added
        for i in gap_inds:
            gap = intervals[i]
            gap_start = dts[i + NN]
            gap_end = dts[i + NN + 1]
            N = gap / median_interval - 1 # number of entries to add
            for j in range(1, N+1):
                new_dt = gap_start + j*timedelta(seconds=median_interval) 
                new_row = np.array([(new_dt,) + (np.nan,) * (row_length - 1)], 
                                                         dtype=data.dtype)
                #TODO: Logs 
                #print ("-- Missing datetime interval between \
                #         %s and %s" % (gap_start, gap_end))
                data = np.append(data, new_row)
                dts = np.append(dts, new_dt) 
                dts_ind = np.argsort(dts) 
                data = data[dts_ind]
            dts = dts[dts_ind] # sorts datetimes
            NN += N
        return data, dts, median_interval, vals_per_hr

    def round_datetime(self, dt, interval):
        # rounds a datetime to a given minute interval
        discard = timedelta(minutes=dt.minute % interval,
                            seconds=dt.second, 
                            microseconds = dt.microsecond)
        dt -= discard
        if discard >= timedelta(minutes=interval/2):
            dt += timedelta(minutes=interval)
        return dt

    def process_datetime(self, dt):
        # takes a datetime and returns a tuple of:
        # minute, hour, weekday, month, and (holiday)
        rv = float(dt.minute), float(dt.hour), float(dt.weekday()), float(dt.month)
        if self.holidays:
            if dt.date() in self.holidays:
                hol = 3.0 # this day is a holiday
            elif (dt + timedelta(1,0)).date() in self.holidays:
                hol = 2.0 # next day is a holiday
            elif (dt - timedelta(1,0)).date() in self.holidays:
                hol = 1.0 # previous day was a holiday
            else:
                hol = 0.0 # this day is not near a holiday
            rv += hol,
        return rv

    def changepoint_feature(self, 
                            changepoints = None, 
                            dayfirst = False,
                            yearfirst = False,
                            **kwargs
                            ):
        if changepoints is not None:
            # convert timestamps to datetimes
            cps = []
            for timestamp,tag in changepoints:
                try:
                    cp_dt = datetime.strptime(timestamp, self.timestamp_format)
                except ValueError:
                    cp_dt = dateutil.parser.parse(timestamp, 
                                                  dayfirst=dayfirst,
                                                  yearfirst=yearfirst)
                cps.append((cp_dt, tag))
            # sort by ascending datetime
            cps.sort(key=lambda tup: tup[0]) 
            feat = np.zeros(len(self.dts))
            for (cp_dt, tag) in cps:
                ind = np.where(self.dts >= cp_dt)[0][0] 
                feat[ind:] = tag
        else:
            feat = None
        return feat

    def split_dataset(self, test_size):
        if self.X_standardizer is None:
            self.X_standardizer = preprocessing.StandardScaler().fit(self.X)
        else:
            pass
        self.y_standardizer = preprocessing.StandardScaler().fit(self.y)
        self.X_s = self.X_standardizer.transform(self.X)
        self.y_s = self.y_standardizer.transform(self.y)
        if self.cps is not None:
            pre_inds = np.where(self.cps == self.PRE_DATA_TAG)
            post_inds = np.where(self.cps == self.POST_DATA_TAG)
           
            self.X_pre_s, self.X_post_s = self.X_s[pre_inds],self.X_s[post_inds]
            self.y_pre_s, self.y_post_s = self.y_s[pre_inds],self.y_s[post_inds]
            self.dts_pre, self.dts_post = \
                 self.dts[pre_inds], self.dts[post_inds]
        else:
            # handle case where no changepoint is given
            # by using a predefined fraction of the dataset
            # to split into pre and post datasets.
            # this is useful for testing the accuracy of the mmodel methods
            # for datasets in which no retrofit is known to have occurred
            pre = len(self.X_s)*(1-test_size)
            post = len(self.X_s)*test_size
            self.X_pre_s, self.X_post_s = self.X_s[:pre], self.X_s[pre:]
            self.y_pre_s, self.y_post_s = self.y_s[:pre], self.y_s[pre:]
            self.dts_pre, self.dts_post = self.dts[:pre], self.dts[pre:]


class ModelAggregator(object):

    def __init__(self, X, y, y_standardizer):
        self.X = X
        self.y = np.ravel(y)
        self.y_standardizer = y_standardizer
        self.models = []
        self.best_model = None
        self.best_score = None
        self.error_metrics = None

    def train(self, model):
        try:
            train = getattr(self, "train_%s" % model)
            m = train()
            self.select_model()
            return m
        except AttributeError:
            raise Exception("Model trainer %s not implemented") % model

    def train_dummy(self, **kwargs):
        dummy_trainer = trainers.DummyTrainer(**kwargs)
        dummy_trainer.train(self.X, 
                            self.y,
                            randomized_search=False)
        self.models.append(dummy_trainer.model)
        return dummy_trainer.model

    def train_hour_weekday(self, **kwargs):
        hour_weekday_trainer = trainers.HourWeekdayBinModelTrainer(**kwargs)
        hour_weekday_trainer.train(self.X, 
                                   self.y, 
                                   randomized_search=False)
        self.models.append(hour_weekday_trainer.model)
        return hour_weekday_trainer.model

    def train_kneighbors(self, **kwargs):
        kneighbors_trainer = trainers.KNeighborsTrainer(**kwargs)
        kneighbors_trainer.train(self.X, self.y)
        self.models.append(kneighbors_trainer.model)
        return kneighbors_trainer.model

    def train_svr(self, **kwargs):
        svr_trainer = trainers.SVRTrainer(**kwargs)
        svr_trainer.train(self.X, self.y)
        self.models.append(svr_trainer.model)
        return svr_trainer.model

    def train_gradient_boosting(self, **kwargs):
        gradient_boosting_trainer = trainers.GradientBoostingTrainer(**kwargs)
        gradient_boosting_trainer.train(self.X, self.y)
        self.models.append(gradient_boosting_trainer.model)
        return gradient_boosting_trainer.model

    def train_random_forest(self, **kwargs):
        random_forest_trainer = trainers.RandomForestTrainer(**kwargs)
        random_forest_trainer.train(self.X, self.y)
        self.models.append(random_forest_trainer.model)
        return random_forest_trainer.model

    def train_extra_trees(self, **kwargs):
        extra_trees_trainer = trainers.ExtraTreesTrainer(**kwargs)
        extra_trees_trainer.train(self.X, self.y)
        self.models.append(extra_trees_trainer.model)
        return extra_trees_trainer.model 

    def train_all(self, **kwargs):
        self.train_dummy(**kwargs)
        self.train_hour_weekday(**kwargs)
        self.train_kneighbors(**kwargs)
        self.train_random_forest(**kwargs)
        # These take forever and maybe aren't worth it?
        #self.train_svr(**kwargs)
        #self.train_gradient_boosting(**kwargs)

        self.select_model()
        self.score()
        return self.models
    
    def select_model(self):
        for model in self.models:
            if model.best_score_ > self.best_score:
                self.best_score = model.best_score_
                self.best_model = model
        return self.best_model, self.best_score

    def score(self):
        baseline = self.y_standardizer.inverse_transform(self.y)
        prediction = self.y_standardizer.inverse_transform(\
                                         self.best_model.predict(self.X))
        self.error_metrics = comparer.Comparer(\
                                      prediction=prediction,baseline=baseline)
        return self.error_metrics

    def __str__(self):
        rv = "\n=== Selected model ==="
        rv += "\nBest cross validation score on training data: %s"%\
                                                   self.best_model.best_score_
        rv += "\nBest model:\n%s"%self.best_model.best_estimator_
        try:
            imps = self.best_model.best_estimator_.feature_importances_
            rv += "\nThe relative importances of input features are:\n%s"%imps
            #TODO rv += "\nWhich corresponds to:\n%s"%self.input_feature_names
        except Exception, e:
            rv += ""
        rv += "\n\n=== Fit to the training data ==="
        rv += "\nThese error metrics represent the match between the"+ \
               " pre-retrofit data used to train the model and" + \
               " the model prediction:"
        rv += str(self.error_metrics)
        return rv

class MnV(object):
    def __init__(self, input_file, address=None, save=False, **kwargs):
        # pre-process the input data file
        self.p = Preprocessor(input_file, **kwargs)
        # build a model based on the pre-retrofit data 
        self.m = ModelAggregator(X=self.p.X_pre_s,
                                 y=self.p.y_pre_s,
                                 y_standardizer=self.p.y_standardizer) 
        self.m.train_all(**kwargs)
        if address is None:
            # evaluate the output of the model against the post-retrofit data
            measured_post_retrofit = self.p.y_standardizer.inverse_transform(\
                                                             self.p.y_post_s)
            predicted_post_retrofit = self.p.y_standardizer.inverse_transform(\
                                    self.m.best_model.predict(self.p.X_post_s))
            self.error_metrics = comparer.Comparer(\
                                         prediction=predicted_post_retrofit,
                                         baseline=measured_post_retrofit)
        else:
            # build a second model based on the post-retrofit data 
            self.m_post = ModelAggregator(X=self.p.X_post_s,
                                         y=self.p.y_post_s,
                                         y_standardizer=self.p.y_standardizer) 
            self.m_post.train_all(**kwargs)
            # evaluate the output of both models over the date range 
            # in the combined pre- & post- retrofit dataset and compare the
            # two predictions to estimate savings 
            # TODO: handle different dataset (e.g. TMY data)
            pre_model = self.p.y_standardizer.inverse_transform(
                                      self.m_pre.best_model.predict(self.p.X_s))
            post_model = self.p.y_standardizer.inverse_transform(
                                    self.m_post.best_model.predict(self.p.X_s))
            # predication basede on TMY data
            locale = location.Location(address)
            interval = str(self.p.interval_seconds/60)+'m'
            tmy_data = location.TMYData(locale.lat, locale.lon, None, interval)
            self.p_tmy = Preprocessor(tmy_data,\
                                      X-standardizer=self.X_standardizer,\
                                      **kwargs)
            pre_model_tmy = self.p.y_standardizer.inverse_transform(\
                            self.m_pre.best_model.predict(self.p_tmy.X_s))
            post_model_tmy = self.p.y_standardizer.inverse_transform(\
                            self.m_post.best_model.predict(self.p_tmy.X_s))
            self.error_metrics = comparer.Comparer(prediction=post_model,
                                               baseline=pre_model)
            self.error_metrics_tmy = comparer.Comparer(\
                                                   prediction=post_model_tmy,\
                                                   baseline=pre_model_tmy)


        if save:
            pickle.Pickler(open('model.pkl', 'wb'), -1).dump(
                                           self.m.best_model.best_estimator_)
            pickle.Pickler(open('error_metrics.pkl', 'wb'), -1).dump(
                                                          self.error_metrics)
            str_date = map(lambda arr: arr.strftime(self.p.timestamp_format),
                           self.p.dts_post)
            X_post = self.p.X_standardizer.inverse_transform(self.p.X_post_s)
            if self.p.use_holidays: 
                header = self.p.headers+'/n'+self.p.named_cols+\
                         'holiday,outsideDB,outsideDB8,'+\
                         'outsideDB16,measured,predicted'
            else:
                header = self.p.headers+'/n'+self.p.named_cols+\
                         'outsideDB,outsideDB,outsideDB16,'+\
                         'measured,predicted'
            post_data = np.column_stack((np.array(str_date),
                                         X_post,
                                         measured_post_retrofit,
                                         predicted_post_retrofit,))
            np.savetxt('post.csv', 
                       post_data, 
                       delimiter=',', 
                       header= header,
                       fmt='%s',
                       comments = '')
                

    def __str__(self):
        rv = "\n===== Pre-retrofit model training summary ====="
        rv += str(self.m)
        rv += "\n===== Results ====="
        rv += "\nThese results quantify the difference between the"+ \
              " measured post-retrofit data and the predicted" + \
              " consumption:"
        rv += str(self.error_metrics)
        rv += "\nThe total estimated energy savings in the post-retrofit" + \
              " period (also known as the avoided energy cost) are:" + \
              " %s [in the original units]"%round(self.error_metrics.tbe,2)
        return rv
 
class DualModelMnV(object):
    def __init__(self, input_file, save=False, **kwargs):
        # pre-process the input data file
        self.p = Preprocessor(input_file=input_file, **kwargs)
        # build a model based on the pre-retrofit data 
        self.m_pre = ModelAggregator(X=self.p.X_pre_s,
                                     y=self.p.y_pre_s,
                                     y_standardizer=self.p.y_standardizer) 
        self.m_pre.train_all(**kwargs)
        # build a second model based on the post-retrofit data 
        self.m_post = ModelAggregator(X=self.p.X_post_s,
                                     y=self.p.y_post_s,
                                     y_standardizer=self.p.y_standardizer) 
        self.m_post.train_all(**kwargs)
        # evaluate the output of both models over the date range 
        # in the combined pre- & post- retrofit dataset and compare the
        # two predictions to estimate savings 
        # TODO: handle different dataset (e.g. TMY data)
        pre_model = self.p.y_standardizer.inverse_transform(
                                    self.m_pre.best_model.predict(self.p.X_s))
        post_model = self.p.y_standardizer.inverse_transform(
                                    self.m_post.best_model.predict(self.p.X_s))
        self.error_metrics = comparer.Comparer(prediction=post_model,
                                               baseline=pre_model)
        if save:
            pickle.Pickler(open('model_pre.pkl', 'wb'), -1).dump(
                                           self.m.best_model.best_estimator_)
            pickle.Pickler(open('model_post.pkl', 'wb'), -1).dump(
                                           self.m.best_model.best_estimator_)
            pickle.Pickler(open('error_metrics.pkl', 'wb'), -1).dump(
                                                          self.error_metrics)
            str_date = map(lambda arr: arr.strftime(self.p.timestamp_format),
                           self.p.dts_post)
            if self.p.use_holidays: 
                header = 'datetime,minute,hour,dayofweek,'+\
                         'month,holiday,outsideDB,outsideDB8,'+\
                         'outsideDB16,measured,predicted'
            else:
                header = 'datetime,minute,hour,dayofweek,'+\
                         'month,outsideDB,outsideDB,outsideDB16,'+\
                         'measured,predicted'
            post_data = np.column_stack((np.array(str_date),
                                         self.p.X,
                                         pre_model,
                                         post_model,))
            np.savetxt('d_post.csv', post_data,\
                        delimiter=',', header= header,\
                        fmt='%s',comments = '')
            
    def __str__(self):
        rv = "\n===== Pre-retrofit model training summary ====="
        rv += str(self.m_pre)
        rv += "\n===== Post-retrofit model training summary ====="
        rv += str(self.m_post)
        rv += "\n===== Results ====="
        rv += "\nThese results compare the energy consumption predicted " + \
              "by both models over the entire date range in the input file." + \
              " One model was trained on the pre-retrofit data and the" + \
              " other was trained on the post-retrofit data:"
        rv += str(self.error_metrics)
        return rv
 
if __name__=='__main__': 
    f = open('data/ex2.csv', 'Ur')
    cps = [
           ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
           ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
           ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
           ("2013/9/14 23:15", Preprocessor.POST_DATA_TAG),
          ]
    mnv = SingleModelMnV(input_file=f, changepoints=cps, save=True)
    print mnv
