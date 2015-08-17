"""
Building Energy Prediction

This software reads an input file (a required argument) containing 
building energy data in a format similar to example file. 
It then trains a model and estimates the error associated
with predictions using the model.

@author Paul Raftery <p.raftery@berkeley.edu>
@author Tyler Hoyt <thoyt@berkeley.edu>
"""

import os, csv, pickle, pdb
import dateutil.parser
import numpy as np
import pprint
from math import sqrt
from datetime import datetime, timedelta
from sklearn import preprocessing, cross_validation, metrics
import trainers, comparer
from holidays import holidays

class Preprocessor(object):

    HOLIDAY_KEYS = ['USFederal']
    DATETIME_COLUMN_NAME = 'time.LOCAL'
    HISTORICAL_DATA_COLUMN_NAMES = ['dboatF']
    TARGET_COLUMN_NAMES = ['wbelectricitykWh']

    def __init__(self, 
                             input_file, 
                             use_holidays=True,
                             start_frac=0.0,
                             end_frac=1.0,
                             changepoint=None,
                             ):

        self.reader = csv.reader(input_file, delimiter=',')
        self.headers, country = self.process_headers()
        input_file.seek(0) # rewind the file so we don't have to open it again

        self.holidays = set([])
        if country == 'us' and use_holidays:
            for key in self.HOLIDAY_KEYS:
                self.holidays.union(holidays[key])

        input_data = np.genfromtxt(input_file, 
                                   comments='#', 
                                   delimiter=',',
                                   dtype=None, 
                                   skip_header=len(self.headers)-1, 
                                   names=True, 
                                   missing_values='NA')
        dcn = self.DATETIME_COLUMN_NAME.replace(".", "")

        input_data_L = len(input_data)
        start_index = int(start_frac * input_data_L)
        end_index = int(end_frac * input_data_L)
        input_data = input_data[ start_index : end_index ]

        try: 
            datetimes = map(lambda d: datetime.strptime(d, "%m/%d/%y %H:%M"),
                                                             input_data[dcn])
        except ValueError:
            datetimes = map(lambda d: dateutil.parser.parse(d, dayfirst=False),
                                                               input_data[dcn])

        dtypes = input_data.dtype.descr
        dtypes[0] = dtypes[0][0], '|S16' # force S16 datetimes
        for i in range(1,len(dtypes)):
            dtypes[i] = dtypes[i][0], 'f8' # parse other data as float
        input_data = input_data.astype(dtypes)

        input_data, self.datetimes = self.interpolate_datetime(input_data,
                                                               datetimes)
        if changepoint is not None:
            self.changepoint_index = self.get_changepoint_index(changepoint)
        else:
            self.changepoint_index = None
        # TODO: this >360 days test to use month or not applies to the  
        # whole dataset... it should only apply to the baseline training dataset
        self.use_month = True if (self.datetimes[0] - \
                                  self.datetimes[-1]).days > 360 else False
        vectorized_process_datetime = np.vectorize(self.process_datetime)
        d = np.column_stack(vectorized_process_datetime(self.datetimes))
        self.num_dt_features = np.shape(d)[1]
        # minute, hour, weekday, holiday, and (month)

        input_data, target_column_index = self.append_input_features(\
                                                                                                        input_data, d)

        self.X, self.y, self.datetimes = \
                self.clean_missing_data(input_data,
                                        self.datetimes,
                                        target_column_index)

    def clean_missing_data(self, d, datetimes, target_column_index):
        # remove any row with missing data
        # filter the datetimes and data arrays so the match up
        keep_inds = ~np.isnan(d).any(axis=1)
        num_to_del = len(keep_inds[~keep_inds])
        if num_to_del > 0: 
            datetimes = datetimes[keep_inds]
            d = d[keep_inds]

        if (d[:,0] != np.array([dt.minute for dt in datetimes])).any() or \
            (d[:,1] != np.array([dt.hour for dt in datetimes])).any():
            raise Error(" - The datetimes in the datetimes array do not \
                match those in the input features array")

        # split into input and target arrays
        target_data = d[:,target_column_index]
        X = np.hstack((d[:,:target_column_index], d[:,target_column_index+1:]))
        return X, target_data, datetimes

    def append_input_features(self, data, d0, historical_data_points=0):
        column_names = data.dtype.names[1:] # no need to include datetime column
        d = d0
        for s in column_names:
            if s in self.HISTORICAL_DATA_COLUMN_NAMES:
                d = np.column_stack( (d, data[s]) )
                if historical_data_points > 0:
                    # create input features using historical data 
                    # at the intervals defined by n_vals_in_past_day
                    for v in range(1, historical_data_points + 1):
                        past_hours = v * 24 / (historical_data_points + 1)
                        n_vals = past_hours * vals_per_hr
                        past_data = np.roll(data[s], n_vals)
                        # for the first day in the file 
                        # there will be no historical data
                        # use the data from the next day as a rough estimate
                        past_data[0:n_vals] = past_data[ 24 * vals_per_hr: \
                                                 24 * vals_per_hr + n_vals ]
            elif not s in self.TARGET_COLUMN_NAMES:
                # just add the column as an input feature 
                # without historical data
                d = np.column_stack( (d, data[s]) )

        # add the target data
        split = d.shape[1]
        for s in self.TARGET_COLUMN_NAMES:
            if s in column_names:
                d = np.column_stack( (d, data[s]) )
        return d, split

    def interpolate_datetime(self, data, datetimes):
        start = datetimes[0]
        second_val = datetimes[1]
        end = datetimes[-1]

        # calculate the interval between datetimes
        interval = second_val - start
        vals_per_hr = 3600 / interval.seconds
        assert (3600 % interval.seconds) == 0,  \
            'Interval between datetimes must divide evenly into an hour'

        # check to ensure that the timedelta between datetimes is
        # uniform through out the array
        row_length = len(data[0])
        diffs = np.diff(datetimes)
        gaps = np.greater(diffs, interval)
        gap_inds = np.nonzero(gaps)[0] # contains the left indices of the gaps
        NN = 0 # accumulate offset of gap indices as you add entries
        for i in gap_inds:
            gap = diffs[i]
            gap_start = datetimes[i + NN]
            gap_end = datetimes[i + NN + 1]
            N = gap.seconds / interval.seconds - 1 # number of entries to add
            for j in range(1, N+1):
                new_dt = gap_start + j * interval
                new_row = np.array([(new_dt,) + (np.nan,) * (row_length - 1)], 
                                                         dtype=data.dtype)
                #TODO: Logs 
                #print ("-- Missing datetime interval between \
                #         %s and %s" % (gap_start, gap_end))
                data = np.append(data, new_row)
                datetimes = np.append(datetimes, new_dt) 
                datetimes_ind = np.argsort(datetimes) 
                data = data[datetimes_ind]
            datetimes = datetimes[datetimes_ind] # sorts datetimes
            NN += N

        return data, datetimes

    def process_headers(self):
        # reads up to the first 100 lines of a file and returns
        # the headers, and the country in which the building is located
        headers = []
        country = None
        for _ in range(100):
            row = self.reader.next()
            headers.append(row)
            for i, val in enumerate(row):
                if val.lower().strip() == 'country':
                    row = self.reader.next()
                    headers.append(row)
                    country = row[i]
            if row[0] == self.DATETIME_COLUMN_NAME: break
        return headers, country

    def process_datetime(self, dt):
        # takes a datetime and returns a tuple of:
        # minute, hour, weekday, holiday, and (month)
        w = float(dt.weekday())
        rv = float(dt.minute), float(dt.hour), w
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

        if self.use_month: rv += float(dt.month),
        return rv

    def get_changepoint_index(self, changepoint):
        a = np.where( self.datetimes >= changepoint )
        changepoint_index = a[0][0]
        return changepoint_index

class ModelAggregator(object):

    def __init__(self, preprocessor, test_size=0.0):
        X = preprocessor.X
        y = np.ravel(preprocessor.y)

        self.X_standardizer = preprocessing.StandardScaler().fit(X)
        self.y_standardizer = preprocessing.StandardScaler().fit(y)
        self.X_s = self.X_standardizer.transform(X)
        self.y_s = self.y_standardizer.transform(y)

        if preprocessor.changepoint_index is not None:
            # split at changepoint
            self.X_pre_s, self.X_post_s = np.split(self.X_s, 
                                                                                         preprocessor.changepoint_index)
            self.y_pre_s, self.y_post_s = np.split(self.y_s, 
                                                                                         preprocessor.changepoint_index)
            print len(self.X_s), len(self.X_pre_s)
        else:
            self.X_pre_s, self.X_post_s, self.y_pre_s, self.y_post_s = \
                    cross_validation.train_test_split(self.X_s, self.y_s, \
                    test_size=test_size, random_state=0)

        self.models = []
        self.best_model = None
        self.best_score = None

    def train_dummy(self):
        dummy_trainer = trainers.DummyTrainer()
        dummy_trainer.train(self.X_pre_s, 
                            self.y_pre_s,
                            randomized_search=False)
        self.models.append(dummy_trainer.model)
        return dummy_trainer.model

    def train_hour_weekday(self):
        hour_weekday_trainer = trainers.HourWeekdayBinModelTrainer()
        hour_weekday_trainer.train(self.X_pre_s, 
                                   self.y_pre_s, 
                                   randomized_search=False)
        self.models.append(hour_weekday_trainer.model)
        return hour_weekday_trainer.model

    def train_kneighbors(self):
        kneighbors_trainer = trainers.KNeighborsTrainer()
        kneighbors_trainer.train(self.X_pre_s, self.y_pre_s)
        self.models.append(kneighbors_trainer.model)
        return kneighbors_trainer.model

    def train_svr(self):
        svr_trainer = trainers.SVRTrainer(search_iterations=0)
        svr_trainer.train(self.X_pre_s, self.y_pre_s)
        self.models.append(svr_trainer.model)
        return svr_trainer.model

    def train_gradient_boosting(self):
        gradient_boosting_trainer = trainers.GradientBoostingTrainer(\
                                                         search_iterations=2)
        gradient_boosting_trainer.train(self.X_pre_s, self.y_pre_s)
        self.models.append(gradient_boosting_trainer.model)
        return gradient_boosting_trainer.model

    def train_random_forest(self):
        random_forest_trainer = trainers.RandomForestTrainer(\
                                                        search_iterations=20)
        random_forest_trainer.train(self.X_pre_s, self.y_pre_s)
        self.models.append(random_forest_trainer.model)
        return random_forest_trainer.model

    def train_extra_trees(self):
        extra_trees_trainer = trainers.ExtraTreesTrainer(search_iterations=20)
        extra_trees_trainer.train(self.X_pre_s, self.y_pre_s)
        self.models.append(extra_trees_trainer.model)
        return extra_trees_trainer.model 

    def train_all(self):
        self.train_dummy()
        self.train_hour_weekday()
        self.train_kneighbors()
        self.train_random_forest()
        # These take forever and maybe aren't worth it?
        #self.train_svr()
        #self.train_gradient_boosting()

        self.select_model()
        return self.models
    
    def select_model(self):
        for model in self.models:
            if model.best_score_ > self.best_score:
                self.best_score = model.best_score_
                self.best_model = model
        return self.best_model, self.best_score

    def __str__(self):
        rv = "\nBest cross validation score on training data: %s"%\
                                                   self.best_model.best_score_
        rv += "\nBest model:\n%s"%self.best_model.best_estimator_
        try:
            imps = self.best_model.best_estimator_.feature_importances_
            rv +="\nThe relative importances of datetime input features are:"
            rv += "\n  Minute: %s"%imps[0]
            rv += "\n  Hour: %s"%imps[1]
            rv += "\n  Weekday: %s"%imps[2]
            rv += "\n  Holiday: %s"%imps[3]
            if self.use_month: print "\n Month: %s"%imps[4]
        except Exception, e:
            print ""
        return rv

    def score(self):
        baseline = self.y_standardizer.inverse_transform(self.y_post_s)
        prediction = self.y_standardizer.inverse_transform(\
                                         self.best_model.predict(self.X_post_s))
        c = comparer.Comparer(prediction=prediction,baseline=baseline)
        #mse = metrics.mean_squared_error(self.y_post_s, y_out)
        return c

if __name__=='__main__': 

    f = open('data/Ex6.csv', 'Ur')
    changepoint = None #datetime(2012, 1, 29, 13, 15)
    p0 = Preprocessor(f, changepoint=changepoint)

    m = ModelAggregator(p0, test_size=0.25)
    m.train_all()
    print m
    print m.score()
