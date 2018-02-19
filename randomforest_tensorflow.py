   # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Random forest in Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np
import csv

# use call() not run() for rm, as it works in python 2.7 and 3.6
from subprocess import call

# pylint: disable=g-backslash-continuation
from tensorflow.contrib.learn.python.learn\
        import metric_spec
from tensorflow.contrib.learn.python.learn.estimators\
        import estimator
from tensorflow.contrib.tensor_forest.client\
        import eval_metrics
from tensorflow.contrib.tensor_forest.client\
        import random_forest
from tensorflow.contrib.tensor_forest.python\
        import tensor_forest
from tensorflow.python.platform import app

# initialise config object to be empty
config = None


class objectview(object):
  def __init__(self, d):
    self.__dict__ = d

# Helper method to load data from CSV file into a numpy array
def loc_genfromtxt(datapath: object) -> object:
  alldata=[]
  with open(datapath) as csvfile:
    csvread = csv.reader(csvfile, delimiter=',')
    for row in csvread:
      alldata.append(np.array(row))
  return np.array(alldata)

# Build an estimator
def build_estimator(model_dir):
  params = tensor_forest.ForestHParams(
      num_classes=config.num_classes, num_features=config.num_features,
      num_trees=config.num_trees, max_nodes=config.max_nodes,
      bagging_fraction=config.bagging_fraction, feature_bagging_fraction=config.feature_bagging_fraction)
  graph_builder_class = tensor_forest.RandomForestGraphs
  if config.use_training_loss:
    graph_builder_class = tensor_forest.TrainingLossForest
  # Use the SKCompat wrapper, which gives us a convenient way to split
  # in-memory data like MNIST into batches.
  return estimator.SKCompat(random_forest.TensorForestEstimator(
      params, graph_builder_class=graph_builder_class,
      model_dir=model_dir))

# Train and evaluate the model
def train_and_eval(conf=None):
  global config

  # if an argument is provided, set config to this value - used for calling the method from outside of the file.
  # if no argument passed, then the arguments passed on the command line, as interpreted by the parser, are used.
  if conf:
    config = conf
  else:
    config = {
      'train_data': train_data,
      'train_labels': train_labels,
      'test_data': test_data,
      'test_labels': test_labels,
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

  # convert config dict into an object, for acceptance in the following lines
  config = objectview(config)

  # if a specific directory to store the generated model is specified in the arguments, use that
  # otherwise, use a temporary directory
  model_dir = tempfile.mkdtemp() if not config.model_dir else config.model_dir

  # load the training data and cast it to float32
  if not config.train_data:
    sys.exit('Usage: --train_data <csv file>')
  train_data = loc_genfromtxt(config.train_data)
  train_data = train_data.astype(np.float32)

  if not config.train_labels:
    sys.exit('Usage: --train_labels <csv file>')
  train_labels = loc_genfromtxt(config.train_labels)
  train_labels = train_labels.astype(np.float32)

  # auto-detect number of features in training data
  # print('train_data has number of features/columns = ' + str(train_data.shape[1]))
  config.num_features = train_data.shape[1]

  # get a random forest estimator object
  est = build_estimator(model_dir)

  # fit the random forest model using the training data
  est.fit(x=train_data, y=train_labels, batch_size=config.batch_size)

  # load the test data and cast it to float32
  if not config.test_data:
    sys.exit('Usage: --test_data <csv file>')
  test_data = loc_genfromtxt(config.test_data)
  test_data = test_data.astype(np.float32)

  if not config.test_labels:
    sys.exit('Usage: --test_labels <csv file>')
  test_labels = loc_genfromtxt(config.test_labels)
  test_labels = test_labels.astype(np.float32)

  # define the metric to be 'accuracy'
  metric_name = 'accuracy'
  metric = {metric_name:
            metric_spec.MetricSpec(
                eval_metrics.get_metric(metric_name),
                prediction_key=eval_metrics.get_prediction_key(metric_name))}

  # calculate the score using the test
  results = est.score(x=test_data, y=test_labels,
                      batch_size=config.batch_size,
                      metrics=metric)

  # print each value with comma and space after it, except last value, which has line feed only
  i = 1
  length = len(sorted(results))
  for key in sorted(results):
    if i == length:
      print(str(results[key]))
    else:
      print(str(results[key]) + ',', end="")
    i = i + 1

  # if flag set, delete model dir in order to free up space / avoid out of memory
  if config.delete_models:
    call(['rm', '-r', model_dir])


def main(_):
  train_and_eval()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data',
      type=str,
      default='/bvdata/randomforest/full_train_nonan_filt.csv',
      help='Path to csv file containing data to train the model.'
  )
  parser.add_argument(
      '--train_labels',
      type=str,
      default='/bvdata/randomforest/full_train_class_filt.csv',
      help='Path to csv file containing training data labels.'
  )
  parser.add_argument(
      '--test_data',
      type=str,
      default='/bvdata/randomforest/full_test_nonan_filt.csv',
      help='Path to csv file containing data to test the model.'
  )
  parser.add_argument(
      '--test_labels',
      type=str,
      default='/bvdata/randomforest/full_test_class_filt.csv',
      help='Path to csv file containing test data labels.'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='Base directory for output models.'
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/data/',
      help='Directory for storing data'
  )
  parser.add_argument(
      '--num_classes',
      type=int,
      default=2,
      help='Number of unique classes in the data labels.'
  )
  parser.add_argument(
      '--num_features',
      type=int,
      default=760,
      help='Number of features in the training data.'
  )
  parser.add_argument(
      '--train_steps',
      type=int,
      default=1,
      help='Number of training steps.'
  )
  parser.add_argument(
      '--batch_size',
      type=str,
      default=200,
      help='Number of examples in a training batch.'
  )
  parser.add_argument(
      '--num_trees',
      type=int,
      default=100,
      help='Number of trees in the forest.'
  )
  parser.add_argument(
      '--max_nodes',
      type=int,
      default=1000,
      help='Max total nodes in a single tree.'
  )
  parser.add_argument(
      '--bagging_fraction',
      type=float,
      default=1.0,
      help='Fraction of training data randomly sampled WITHOUT replacement to train each tree (default = 1.0, no random selection).'
  )
  parser.add_argument(
      '--feature_bagging_fraction',
      type=float,
      default=1.0,
      help='Fraction of total features randomly selected for each tree (default = 1.0, no random selection).'
  )
  parser.add_argument(
      '--use_training_loss',
      type=bool,
      default=False,
      help='If true, use training loss as termination criteria.'
  )
  parser.add_argument(
      '--delete_models',
      type=bool,
      default=False,
      help='If true, delete models one by one to save space'
  )
  config, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
