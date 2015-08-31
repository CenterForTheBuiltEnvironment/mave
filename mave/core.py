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
    DATETIME_COLUMN_NAME = 'LocalDateTime'
    HISTORICAL_DATA_COLUMN_NAMES = ['OutsideDryBulbTemperature']
    TARGET_COLUMN_NAMES = ['EnergyConsumption']
    DISCARD_TAG = -1
    PRE_DATA_TAG = 0
    POST_DATA_TAG = 1

    def __init__(self, 
                 input_file, 
                 use_holidays=True,
                 start_frac=0.0,
                 end_frac=1.0,
                 changepoints=None,
                 test_size=0.25
        ):

        self.reader = csv.reader(input_file, delimiter=',')
        headers, country, named_cols = self.process_headers()
        input_file.seek(0) # rewind the file so we don't have to open it again
        self.holidays = set([])
        if country == 'us' and use_holidays:
            for key in self.HOLIDAY_KEYS:
                self.holidays = self.holidays.union(holidays[key])
        input_data = np.genfromtxt(input_file, 
                                   delimiter=',',
                                   dtype=None, 
                                   skip_header=len(headers)-1, 
                                   usecols=named_cols,
                                   names=True, 
                                   missing_values='NA')
        dcn = self.DATETIME_COLUMN_NAME
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
        self.vals_per_hr = 0
        input_data, self.datetimes = self.interpolate_datetime(input_data,
                                                               datetimes)
        vectorized_process_datetime = np.vectorize(self.process_datetime)
        d = np.column_stack(vectorized_process_datetime(self.datetimes))

        # add other (non datetime related) input features
        input_data, target_column_index = self.append_input_features(\
                                                                input_data, d)
        self.X, self.y, self.datetimes = \
                self.clean_missing_data(input_data,
                                        self.datetimes,
                                        target_column_index)
        self.cps = self.changepoint_feature(changepoints) 
        self.split_dataset(test_size=test_size)

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

    def append_input_features(self, data, d0, historical_data_points=2):
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
                        n_vals = past_hours * self.vals_per_hr
                        past_data = np.roll(data[s], n_vals)
                        # for the first day in the file 
                        # there will be no historical data
                        # use the data from the next day as a rough estimate
                        past_data[0:n_vals] = past_data[24*self.vals_per_hr: \
                                                 24*self.vals_per_hr+n_vals ]
                        d = np.column_stack( (d, past_data) )
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
        self.vals_per_hr = 3600 / interval.seconds
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
        for _ in range(100):
            row = self.reader.next()
            headers.append(row)
            for i, val in enumerate(row):
                if val.lower().strip() == 'country':
                    row = self.reader.next()
                    headers.append(row)
                    country = row[i]
            if len(row)>0: 
                if row[0] == self.DATETIME_COLUMN_NAME: 
                    named_cols = tuple(np.where(np.array(row) !='')[0])
                    break
        return headers, country, named_cols

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

    def changepoint_feature(self, changepoints):
        if changepoints is not None:
            cps = sorted(changepoints)
            feat = np.zeros(len(self.datetimes))
            for (cp_date, tag) in cps:
                ind = np.where(self.datetimes >= cp_date)[0][0] 
                feat[ind:] = tag
        else:
            feat = None
        return feat

    def split_dataset(self, test_size):
        self.X_standardizer = preprocessing.StandardScaler().fit(self.X)
        self.y_standardizer = preprocessing.StandardScaler().fit(self.y)
        self.X_s = self.X_standardizer.transform(self.X)
        self.y_s = self.y_standardizer.transform(self.y)
        if self.cps is not None:
            pre_inds = np.where(self.cps == self.PRE_DATA_TAG)
            post_inds = np.where(self.cps == self.POST_DATA_TAG)
           
            self.X_pre_s, self.X_post_s = self.X_s[pre_inds],self.X_s[post_inds]
            self.y_pre_s, self.y_post_s = self.y_s[pre_inds],self.y_s[post_inds]
            self.datetimes_pre, self.datetimes_post = \
                 self.datetimes[pre_inds], self.datetimes[post_inds]
        else:
            # handle case where no changepoint is given
            # by using a predefined fraction of the dataset
            # to split into pre and post datasets.
            # this is useful for testing the accuracy of the mmodel methods
            # for datasets in which no retrofit is known to have occurred
            self.X_pre_s, self.X_post_s, self.y_pre_s, self.y_post_s = \
                    cross_validation.train_test_split(self.X_s, self.y_s, \
                    test_size=test_size, random_state=0)

class ModelAggregator(object):

    def __init__(self, 
                 preprocessor, 
                 model_type):
        self.p = preprocessor
        self.model_type = model_type
        if self.model_type == "pre-retrofit":
            self.X = preprocessor.X_pre_s
            self.y = np.ravel(self.p.y_pre_s)
        elif self.model_type == "post-retrofit":
            self.X = preprocessor.X_post_s
            self.y = np.ravel(self.p.y_post_s)
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

    def train_dummy(self):
        dummy_trainer = trainers.DummyTrainer()
        dummy_trainer.train(self.X, 
                            self.y,
                            randomized_search=False)
        self.models.append(dummy_trainer.model)
        return dummy_trainer.model

    def train_hour_weekday(self):
        hour_weekday_trainer = trainers.HourWeekdayBinModelTrainer()
        hour_weekday_trainer.train(self.X, 
                                   self.y, 
                                   randomized_search=False)
        self.models.append(hour_weekday_trainer.model)
        return hour_weekday_trainer.model

    def train_kneighbors(self):
        kneighbors_trainer = trainers.KNeighborsTrainer(\
                                     search_iterations=5)
        kneighbors_trainer.train(self.X, self.y)
        self.models.append(kneighbors_trainer.model)
        return kneighbors_trainer.model

    def train_svr(self):
        svr_trainer = trainers.SVRTrainer(\
                                     search_iterations=5)
        svr_trainer.train(self.X, self.y)
        self.models.append(svr_trainer.model)
        return svr_trainer.model

    def train_gradient_boosting(self):
        gradient_boosting_trainer = trainers.GradientBoostingTrainer(\
                                                            search_iterations=5)
        gradient_boosting_trainer.train(self.X, self.y)
        self.models.append(gradient_boosting_trainer.model)
        return gradient_boosting_trainer.model

    def train_random_forest(self):
        random_forest_trainer = trainers.RandomForestTrainer(\
                                                           search_iterations=20)
        random_forest_trainer.train(self.X, self.y)
        self.models.append(random_forest_trainer.model)
        return random_forest_trainer.model

    def train_extra_trees(self):
        extra_trees_trainer = trainers.ExtraTreesTrainer(\
                                                         search_iterations=20)
        extra_trees_trainer.train(self.X, self.y)
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
        self.score()
        return self.models
    
    def select_model(self):
        for model in self.models:
            if model.best_score_ > self.best_score:
                self.best_score = model.best_score_
                self.best_model = model
        return self.best_model, self.best_score

    def score(self):
        baseline = self.p.y_standardizer.inverse_transform(self.y)
        prediction = self.p.y_standardizer.inverse_transform(\
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
        rv += "\n\n=== Fit to the %s data ===" % self.model_type
        rv += "\nThese error metrics represent the match between the"+ \
                 " %s data and the model:"%self.model_type
        rv += str(self.error_metrics)
        return rv

class SingleModelMeasurementAndVerification(object):
    def __init__(self, preprocessor):
        p = preprocessor
        self.m = ModelAggregator(preprocessor = p, model_type="pre-retrofit")
        self.m.train_all()
        measured_post_retrofit = p.y_standardizer.inverse_transform(p.y_post_s)
        predicted_post_retrofit = p.y_standardizer.inverse_transform(\
                                         self.m.best_model.predict(p.X_post_s))
        self.error_metrics = comparer.Comparer(\
                                         prediction=predicted_post_retrofit,
                                         baseline=measured_post_retrofit)

    def __str__(self):
        rv = str(self.m)
        rv += "\n=== Results ==="
        rv += "\nThese error metrics represent the match between the"+ \
                 " measured postretrofit data and the predicted consumption:"
        rv += str(self.error_metrics)
        return rv
 
class DualModelMeasurementAndVerification(object):
    def __init__(self, preprocessor):
        raise NotImplemented

if __name__=='__main__': 
    f = open('data/ex6.csv', 'Ur')
    changepoints = [
                   (datetime(2012, 1, 29, 12, 0), Preprocessor.PRE_DATA_TAG),
                   (datetime(2012, 12, 20, 0, 0), Preprocessor.DISCARD_TAG),
                   (datetime(2013, 1, 5, 0, 0), Preprocessor.PRE_DATA_TAG),
                   (datetime(2013, 3, 1, 0, 0), Preprocessor.POST_DATA_TAG),
                   ]
    p = Preprocessor(f, changepoints=changepoints)
    mnv = SingleModelMeasurementAndVerification(preprocessor=p)
    print mnv
