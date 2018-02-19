import os
import sys

repo_path = os.path.expandvars('${BATVOICE_REPO}')
if repo_path not in sys.path:
 sys.path.append(repo_path)
from setup_bv import setup_bv as setup
repo_path, data_path, libs_path = setup.do_setup()

import models.randomforest_tensorflow as rftf
import time
import sys

# use call() not run() for shutdown, as it works in python 2.7 and 3.6
from subprocess import call

# # # # # # # # # # # # # # # # # # # # # # # #
# DATA FILES

# folder path of data files. Path ust end with a /
FOLDER_PATH = '/data/randomforest/'
# FOLDER_PATH = '/Users/carl/Dropbox/Docs/Batvoice/Entreparticuliers/'

# EDITED PARTIAL LIST (resuming from crashed run)
# lists of train and test data and labels. Related files share the same index across the 4 lists
# FULL LIST
# # lists of train and test data and labels. Related files share the same index across the 4 lists
LIST_TRAIN_DATA = ['full_train_data_filt_over_60_nan-tomedian_zscore-normalization.csv']
LIST_TRAIN_LABELS = ['full_train_class_filt_over_60_nan-tomedian_zscore-normalization.csv']
LIST_TEST_DATA = ['full_test_data_filt_over_60_nan-tomedian_zscore-normalization.csv']
LIST_TEST_LABELS = ['full_test_class_filt_over_60_nan-tomedian_zscore-normalization.csv']

# FULL LIST
# # lists of train and test data and labels. Related files share the same index across the 4 lists
# LIST_TRAIN_DATA = ['full_train_data_filt_over_120_nan-col-eliminated_no-normalization.csv',
# 'full_train_data_filt_over_120_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_data_filt_over_120_nan-keep_no-normalization.csv',
# 'full_train_data_filt_over_120_nan-keep_zscore-normalization.csv',
# 'full_train_data_filt_over_120_nan-tomedian_no-normalization.csv',
# 'full_train_data_filt_over_120_nan-tomedian_zscore-normalization.csv',
# 'full_train_data_filt_over_120_nan-tozero_no-normalization.csv',
# 'full_train_data_filt_over_120_nan-tozero_zscore-normalization.csv',
# 'full_train_data_filt_over_180_nan-col-eliminated_no-normalization.csv',
# 'full_train_data_filt_over_180_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_data_filt_over_180_nan-keep_no-normalization.csv',
# 'full_train_data_filt_over_180_nan-keep_zscore-normalization.csv',
# 'full_train_data_filt_over_180_nan-tomedian_no-normalization.csv',
# 'full_train_data_filt_over_180_nan-tomedian_zscore-normalization.csv',
# 'full_train_data_filt_over_180_nan-tozero_no-normalization.csv',
# 'full_train_data_filt_over_180_nan-tozero_zscore-normalization.csv',
# 'full_train_data_filt_over_240_nan-col-eliminated_no-normalization.csv',
# 'full_train_data_filt_over_240_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_data_filt_over_240_nan-keep_no-normalization.csv',
# 'full_train_data_filt_over_240_nan-keep_zscore-normalization.csv',
# 'full_train_data_filt_over_240_nan-tomedian_no-normalization.csv',
# 'full_train_data_filt_over_240_nan-tomedian_zscore-normalization.csv',
# 'full_train_data_filt_over_240_nan-tozero_no-normalization.csv',
# 'full_train_data_filt_over_240_nan-tozero_zscore-normalization.csv',
# 'full_train_data_filt_over_300_nan-col-eliminated_no-normalization.csv',
# 'full_train_data_filt_over_300_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_data_filt_over_300_nan-keep_no-normalization.csv',
# 'full_train_data_filt_over_300_nan-keep_zscore-normalization.csv',
# 'full_train_data_filt_over_300_nan-tomedian_no-normalization.csv',
# 'full_train_data_filt_over_300_nan-tomedian_zscore-normalization.csv',
# 'full_train_data_filt_over_300_nan-tozero_no-normalization.csv',
# 'full_train_data_filt_over_300_nan-tozero_zscore-normalization.csv',
# 'full_train_data_filt_over_60_nan-col-eliminated_no-normalization.csv',
# 'full_train_data_filt_over_60_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_data_filt_over_60_nan-keep_no-normalization.csv',
# 'full_train_data_filt_over_60_nan-keep_zscore-normalization.csv',
# 'full_train_data_filt_over_60_nan-tomedian_no-normalization.csv',
# 'full_train_data_filt_over_60_nan-tomedian_zscore-normalization.csv',
# 'full_train_data_filt_over_60_nan-tozero_no-normalization.csv',
# 'full_train_data_filt_over_60_nan-tozero_zscore-normalization.csv']
# LIST_TRAIN_LABELS = ['full_train_class_filt_over_120_nan-col-eliminated_no-normalization.csv',
# 'full_train_class_filt_over_120_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_class_filt_over_120_nan-keep_no-normalization.csv',
# 'full_train_class_filt_over_120_nan-keep_zscore-normalization.csv',
# 'full_train_class_filt_over_120_nan-tomedian_no-normalization.csv',
# 'full_train_class_filt_over_120_nan-tomedian_zscore-normalization.csv',
# 'full_train_class_filt_over_120_nan-tozero_no-normalization.csv',
# 'full_train_class_filt_over_120_nan-tozero_zscore-normalization.csv',
# 'full_train_class_filt_over_180_nan-col-eliminated_no-normalization.csv',
# 'full_train_class_filt_over_180_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_class_filt_over_180_nan-keep_no-normalization.csv',
# 'full_train_class_filt_over_180_nan-keep_zscore-normalization.csv',
# 'full_train_class_filt_over_180_nan-tomedian_no-normalization.csv',
# 'full_train_class_filt_over_180_nan-tomedian_zscore-normalization.csv',
# 'full_train_class_filt_over_180_nan-tozero_no-normalization.csv',
# 'full_train_class_filt_over_180_nan-tozero_zscore-normalization.csv',
# 'full_train_class_filt_over_240_nan-col-eliminated_no-normalization.csv',
# 'full_train_class_filt_over_240_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_class_filt_over_240_nan-keep_no-normalization.csv',
# 'full_train_class_filt_over_240_nan-keep_zscore-normalization.csv',
# 'full_train_class_filt_over_240_nan-tomedian_no-normalization.csv',
# 'full_train_class_filt_over_240_nan-tomedian_zscore-normalization.csv',
# 'full_train_class_filt_over_240_nan-tozero_no-normalization.csv',
# 'full_train_class_filt_over_240_nan-tozero_zscore-normalization.csv',
# 'full_train_class_filt_over_300_nan-col-eliminated_no-normalization.csv',
# 'full_train_class_filt_over_300_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_class_filt_over_300_nan-keep_no-normalization.csv',
# 'full_train_class_filt_over_300_nan-keep_zscore-normalization.csv',
# 'full_train_class_filt_over_300_nan-tomedian_no-normalization.csv',
# 'full_train_class_filt_over_300_nan-tomedian_zscore-normalization.csv',
# 'full_train_class_filt_over_300_nan-tozero_no-normalization.csv',
# 'full_train_class_filt_over_300_nan-tozero_zscore-normalization.csv',
# 'full_train_class_filt_over_60_nan-col-eliminated_no-normalization.csv',
# 'full_train_class_filt_over_60_nan-col-eliminated_zscore-normalization.csv',
# 'full_train_class_filt_over_60_nan-keep_no-normalization.csv',
# 'full_train_class_filt_over_60_nan-keep_zscore-normalization.csv',
# 'full_train_class_filt_over_60_nan-tomedian_no-normalization.csv',
# 'full_train_class_filt_over_60_nan-tomedian_zscore-normalization.csv',
# 'full_train_class_filt_over_60_nan-tozero_no-normalization.csv',
# 'full_train_class_filt_over_60_nan-tozero_zscore-normalization.csv']
# LIST_TEST_DATA = ['full_test_data_filt_over_120_nan-col-eliminated_no-normalization.csv',
# 'full_test_data_filt_over_120_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_data_filt_over_120_nan-keep_no-normalization.csv',
# 'full_test_data_filt_over_120_nan-keep_zscore-normalization.csv',
# 'full_test_data_filt_over_120_nan-tomedian_no-normalization.csv',
# 'full_test_data_filt_over_120_nan-tomedian_zscore-normalization.csv',
# 'full_test_data_filt_over_120_nan-tozero_no-normalization.csv',
# 'full_test_data_filt_over_120_nan-tozero_zscore-normalization.csv',
# 'full_test_data_filt_over_180_nan-col-eliminated_no-normalization.csv',
# 'full_test_data_filt_over_180_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_data_filt_over_180_nan-keep_no-normalization.csv',
# 'full_test_data_filt_over_180_nan-keep_zscore-normalization.csv',
# 'full_test_data_filt_over_180_nan-tomedian_no-normalization.csv',
# 'full_test_data_filt_over_180_nan-tomedian_zscore-normalization.csv',
# 'full_test_data_filt_over_180_nan-tozero_no-normalization.csv',
# 'full_test_data_filt_over_180_nan-tozero_zscore-normalization.csv',
# 'full_test_data_filt_over_240_nan-col-eliminated_no-normalization.csv',
# 'full_test_data_filt_over_240_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_data_filt_over_240_nan-keep_no-normalization.csv',
# 'full_test_data_filt_over_240_nan-keep_zscore-normalization.csv',
# 'full_test_data_filt_over_240_nan-tomedian_no-normalization.csv',
# 'full_test_data_filt_over_240_nan-tomedian_zscore-normalization.csv',
# 'full_test_data_filt_over_240_nan-tozero_no-normalization.csv',
# 'full_test_data_filt_over_240_nan-tozero_zscore-normalization.csv',
# 'full_test_data_filt_over_300_nan-col-eliminated_no-normalization.csv',
# 'full_test_data_filt_over_300_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_data_filt_over_300_nan-keep_no-normalization.csv',
# 'full_test_data_filt_over_300_nan-keep_zscore-normalization.csv',
# 'full_test_data_filt_over_300_nan-tomedian_no-normalization.csv',
# 'full_test_data_filt_over_300_nan-tomedian_zscore-normalization.csv',
# 'full_test_data_filt_over_300_nan-tozero_no-normalization.csv',
# 'full_test_data_filt_over_300_nan-tozero_zscore-normalization.csv',
# 'full_test_data_filt_over_60_nan-col-eliminated_no-normalization.csv',
# 'full_test_data_filt_over_60_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_data_filt_over_60_nan-keep_no-normalization.csv',
# 'full_test_data_filt_over_60_nan-keep_zscore-normalization.csv',
# 'full_test_data_filt_over_60_nan-tomedian_no-normalization.csv',
# 'full_test_data_filt_over_60_nan-tomedian_zscore-normalization.csv',
# 'full_test_data_filt_over_60_nan-tozero_no-normalization.csv',
# 'full_test_data_filt_over_60_nan-tozero_zscore-normalization.csv']
# LIST_TEST_LABELS = ['full_test_class_filt_over_120_nan-col-eliminated_no-normalization.csv',
# 'full_test_class_filt_over_120_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_class_filt_over_120_nan-keep_no-normalization.csv',
# 'full_test_class_filt_over_120_nan-keep_zscore-normalization.csv',
# 'full_test_class_filt_over_120_nan-tomedian_no-normalization.csv',
# 'full_test_class_filt_over_120_nan-tomedian_zscore-normalization.csv',
# 'full_test_class_filt_over_120_nan-tozero_no-normalization.csv',
# 'full_test_class_filt_over_120_nan-tozero_zscore-normalization.csv',
# 'full_test_class_filt_over_180_nan-col-eliminated_no-normalization.csv',
# 'full_test_class_filt_over_180_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_class_filt_over_180_nan-keep_no-normalization.csv',
# 'full_test_class_filt_over_180_nan-keep_zscore-normalization.csv',
# 'full_test_class_filt_over_180_nan-tomedian_no-normalization.csv',
# 'full_test_class_filt_over_180_nan-tomedian_zscore-normalization.csv',
# 'full_test_class_filt_over_180_nan-tozero_no-normalization.csv',
# 'full_test_class_filt_over_180_nan-tozero_zscore-normalization.csv',
# 'full_test_class_filt_over_240_nan-col-eliminated_no-normalization.csv',
# 'full_test_class_filt_over_240_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_class_filt_over_240_nan-keep_no-normalization.csv',
# 'full_test_class_filt_over_240_nan-keep_zscore-normalization.csv',
# 'full_test_class_filt_over_240_nan-tomedian_no-normalization.csv',
# 'full_test_class_filt_over_240_nan-tomedian_zscore-normalization.csv',
# 'full_test_class_filt_over_240_nan-tozero_no-normalization.csv',
# 'full_test_class_filt_over_240_nan-tozero_zscore-normalization.csv',
# 'full_test_class_filt_over_300_nan-col-eliminated_no-normalization.csv',
# 'full_test_class_filt_over_300_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_class_filt_over_300_nan-keep_no-normalization.csv',
# 'full_test_class_filt_over_300_nan-keep_zscore-normalization.csv',
# 'full_test_class_filt_over_300_nan-tomedian_no-normalization.csv',
# 'full_test_class_filt_over_300_nan-tomedian_zscore-normalization.csv',
# 'full_test_class_filt_over_300_nan-tozero_no-normalization.csv',
# 'full_test_class_filt_over_300_nan-tozero_zscore-normalization.csv',
# 'full_test_class_filt_over_60_nan-col-eliminated_no-normalization.csv',
# 'full_test_class_filt_over_60_nan-col-eliminated_zscore-normalization.csv',
# 'full_test_class_filt_over_60_nan-keep_no-normalization.csv',
# 'full_test_class_filt_over_60_nan-keep_zscore-normalization.csv',
# 'full_test_class_filt_over_60_nan-tomedian_no-normalization.csv',
# 'full_test_class_filt_over_60_nan-tomedian_zscore-normalization.csv',
# 'full_test_class_filt_over_60_nan-tozero_no-normalization.csv',
# 'full_test_class_filt_over_60_nan-tozero_zscore-normalization.csv']
#
# LIST_TRAIN_DATA = ['x_full_train_nonan_filt.csv', 'x_full_train_nonan_filt2.csv']
# LIST_TRAIN_LABELS = ['x_full_train_class_filt.csv', 'x_full_train_class_filt2.csv']
# LIST_TEST_DATA = ['x_full_test_nonan_filt.csv', 'x_full_test_nonan_filt2.csv']
# LIST_TEST_LABELS = ['x_full_test_class_filt.csv', 'x_full_test_class_filt2.csv']

# END OF DATA FILES
# # # # # # # # # # # # # # # # # # # # # # #

# create timestamp for logfile and conffile filenames (only one timestamp per harness run, for file grouping)
timestr = time.strftime("%Y%m%d_%H%M%S")

# for loop to iterate through sets of data files defined in the LISTS
for train_data, train_labels, test_data, test_labels in zip(LIST_TRAIN_DATA, LIST_TRAIN_LABELS,
                                                            LIST_TEST_DATA, LIST_TEST_LABELS):

  # try/except block ensures that test harness does not terminate if one input file has a problem
  try:

    # # # # # # # # # # # # # # # # # # # # # # # #
    # VARIABLES

    # define value of arguments, including start value of variable to be incremented.
    num_classes = 2
    num_features = 760  # randomforest_tensorflow.py will detect automatically from training data file
    num_trees = 150 # optimal 150
    max_nodes = 1000
    train_steps = 1
    batch_size = 200
    bagging_fraction = 0.4 #optimal 0.4 or 0.9 (with 60s data)
    feature_bagging_fraction = 1.0  # optimal 1.0 (with 60s data)
    data_dir = '/tmp/data/'  # must end with /
    model_dir = ''  # will default to OS tmp directory if not specified
    use_training_loss = False

    # sole variable to change (must be a string)
    # This file only adjusts one variable at a time in the for loop, to hone in on the right combo
    variable = 'bagging_fraction'

    # set step increment for chosen variable
    step = 0.1

    # set number of times to increment chosen variable by step
    iterations = 10

    # if flag set to True, models will be deleted to save storage space and avoid out of memory
    delete_models = True

    # if shutdown flag is set to True, OS will shutdown and the instance will terminate to save money
    shutdown = False

    # END OF VARIABLES
    # # # # # # # # # # # # # # # # # # # # # # # #

    # create output files with timestamp in filename. A pair of files for each training data file.
    conffilepath = FOLDER_PATH + 'logfile_' + timestr + '_' + train_data + '_' + variable + '_conf.txt'
    logfilepath = FOLDER_PATH + 'logfile_' + timestr + '_' + train_data + '_' + variable + '.txt'

    # set attribute values for conf dictionary object, to pass into train_and_eval method of
    # randomforest_tensorflow
    conf = {
      'train_data': FOLDER_PATH + train_data,
      'train_labels': FOLDER_PATH + train_labels,
      'test_data': FOLDER_PATH + test_data,
      'test_labels': FOLDER_PATH + test_labels,
      'num_classes': num_classes,
      'num_features': num_features,
      'num_trees': num_trees,
      'max_nodes': max_nodes,
      'train_steps': train_steps,
      'batch_size': batch_size,
      'bagging_fraction': bagging_fraction,
      'feature_bagging_fraction': feature_bagging_fraction,
      'model_dir': model_dir,
      'delete_models': delete_models,
      'data_dir': data_dir,
      'use_training_loss': use_training_loss
    }

    # append hyperparameters so we know what each of the results mean
    sys.stdout = open(conffilepath, 'a')
    print(conf)
    print(timestr)

    # append csv formatted output to logfile
    sys.stdout = open(logfilepath, 'a')

    # print header line to logfile
    print('iteration,' + variable + ',accuracy,global_step,loss')

    # for loop to call randomforest_tensorflow.py with increasing variable values (use step)
    for _ in range(1,iterations+1):

      # print loop iteration to logfile - syntax needs python 3
      print(str(_) + ',', end="")
      print(str(conf[variable]) + ',', end="")

      # call train_and_eval method of randomforest_tensorflow, passing in the conf object as arguments
      # append results to the logfile
      rftf.train_and_eval(conf=conf)

      # throw exception
      # raise ValueError('A very specific bad thing happened')

      # flush stdout to write to file
      sys.stdout.flush()

      # increment variable by step
      conf[variable] = conf[variable] + step

  except Exception as e:
    print('\n' + str(e))
    pass

# if shutdown flag is set to True, OS will shutdown and the instance will terminate to save money
# run locally on mac i get 'sudo: no tty present and no askpass program specified' but on server it works
# use if __name__ == "__main__": to avoid pydoc and other scripts running the code
if __name__ == "__main__":
    if shutdown:
        call(['sudo', 'shutdown', '-h', 'now'])
