"""
Misc methods for visualizing/quantifying the error between two arrays
Passed values can be python lists or numpy arrays,
but each must be an the same length/shape

@author Paul Raftery <p.raftery@berkeley.edu>
"""
import math, pickle, os
import numpy as np
import matplotlib.pyplot as plt

def ne(predicted, actual):
    # return an array containing the absolute errors between
    # a predicted and actual value
    predicted = np.array(predicted).astype(np.float)
    actual = np.array(actual).astype(np.float)
    return 100*abs((actual-predicted)/actual)

def nmbe(predicted, actual):
    # return the normalized mean bias error between
    # two arrays: predicted  and actual
    predicted = np.array(predicted).astype(np.float)
    actual = np.array(actual).astype(np.float)
    return 100*(sum(actual-predicted)/np.mean(actual))/len(actual)

def rmse(predicted, actual):
    # return the root mean squared errors between
    # two arrays: predicted  and actual
    predicted = np.array(predicted).astype(np.float)
    actual = np.array(actual).astype(np.float)
    return math.sqrt(sum((actual-predicted)**2)/len(actual))

def cvrmse(predicted, actual):
    # return athe cumulative variation of root mean squared between
    # two arrays: predicted  and actual
    predicted = np.array(predicted).astype(np.float)
    actual = np.array(actual).astype(np.float)
    return 100*(rmse(predicted,actual))/np.mean(actual)

def plot_comparison(predicted, actual):
    n_points = 1000
    a = np.random.randint(len(actual)-n_points)
    b = a+n_points -1
    t = range(len(predicted-actual))
    plt.figure(figsize=(20, 10))
    plt.subplot(211, title="Target value (blue) vs. predicted (red)")
    plt.plot(t[a:b], actual[a:b], 'b-', t[a:b], predicted[a:b], 'ro', a,0, 'b--')
    plt.ylabel('Target value [orig. units]', fontsize=12)
    plt.subplot(212)
    plt.plot(t[a:b], predicted[a:b]/actual[a:b], 'k-', a,0, 'k--')
    plt.xlabel('Training datapoint (ordered, typ. 15 min)', fontsize=10)
    plt.ylabel('Error fraction [-]', fontsize=10)
    plt.suptitle('Prediction Data', fontsize=14, fontweight='bold')
    plt.show()

def print_overview(pred, act, logger):
    ## assess how well the model performs
    logger.info(' NMBE: ' + str(nmbe(pred,act)) + '%')
    logger.info(' CVRMSE: ' + str(cvrmse(pred,act)) + '%')
    norm_errs = ne(pred,act)
    logger.info(' Normalized error,min :' + str(np.min(norm_errs)) + '%')
    logger.info(' Normalized error, 10th percentile :' + str(np.percentile(norm_errs,10)) + '%')
    logger.info(' Normalized error, median :' + str(np.percentile(norm_errs,50)) + '%')
    logger.info(' Normalized error,mean :' + str(np.mean(norm_errs)) + '%')
    logger.info(' Normalized error, 90th percentile :' + str(np.percentile(norm_errs,90)) + '%')
    logger.info(' Normalized error,max :' + str(np.max(norm_errs)) + '%')

def write_model_results(models, op):
    for model in models:
        model_name = str(model.best_estimator_).split('(')[0] # gets the name of the regressor
        model_folder = os.path.join(op,'Models')
        if not os.path.exists(model_folder): os.makedirs(model_folder)
        with open(os.path.join(model_folder,model_name+'_cross_val_results.txt'), 'w') as fo:
            fo.write('------ Best score -----\n')
            fo.write(str(model.best_score_))
            fo.write('\n')
            fo.write('------ Best parameters -----\n')
            fo.write(str(model.best_params_))
            fo.write('\n')
            fo.write('------ Grid scores -----\n')
            for i in range(len(model.grid_scores_)):
                fo.write("\n")
                fo.write(str(model.grid_scores_[i]))
        with open(os.path.join(model_folder,model_name+'.p'), 'w') as fo:
            pickle.dump(model.best_estimator_, fo)
