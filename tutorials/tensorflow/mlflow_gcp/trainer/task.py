# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains a Keras model to predict income bracket from other Census data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import tempfile
import os
import shutil

from builtins import int
from mlflow import pyfunc
from tensorflow.python.saved_model import tag_constants
from time import time

from . import model
from . import utils
from . import model_deployment

import mlflow
import mlflow.tensorflow
import tensorflow as tf


# mlflow.tensorflow.autolog()


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        default='gs://cloud-samples-data/ml-engine/census/data/adult.data.csv')
    parser.add_argument(
        '--eval-files',
        help='GCS file or local paths to evaluation data',
        default='gs://cloud-samples-data/ml-engine/census/data/adult.test.csv')
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='Local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for, at each checkpoint',
        default=1,
        type=int)
    parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""
          Flag to decide if the model checkpoint should be re-used from the 
          job-dir.
          If set to False then the job-dir will be deleted.
          """)
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    parser.add_argument(
        '--deploy-gcp',
        action='store_true',
        default=False,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--project-id',
        type=str,
        help='AI Platform project id')
    #
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default='mlflow tracking URI',
        help='MLFlow tracking URI')
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default='mlflow_gcp',
        help='AI Platform GCS bucket')
    parser.add_argument(
        '--model-name',
        type=str,
        default='mlflow',
        help='AI Platform model')
    parser.add_argument(
        '--run-time-version',
        type=str,
        default='1.14',
        help='AI Platform Run time version')
    args, _ = parser.parse_known_args()
    return args


def _mlflow_log_metrics(metrics, metric_name):
    """Record metric value during each epoch using the step parameter in
    mlflow.log_metric.

    :param metrics:
    :param metric_name:
    :return:
    """
    for epoch, metric in enumerate(metrics[metric_name], 1): mlflow.log_metric(
        metric_name, metric,
        step=epoch)


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in utils.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    History objects returns:
        {'loss': [0.5699903990809373, 0.3629718415849791],
         'acc': [0.78604823, 0.8331693],
         'val_loss': [0.3966572880744934, 0.3477487564086914],
         'val_acc': [0.8278044, 0.8281116],
         'lr': [0.02, 0.015]}
    Args:
      args: dictionary of arguments - see get_args() for details
    """

    logging.info('Resume training: {}'.format(args.reuse_job_dir))
    if not args.reuse_job_dir:
        if tf.io.gfile.exists(args.job_dir):
            tf.io.gfile.rmtree(args.job_dir)
            logging.info(
                'Deleted job_dir {} to avoid re-use'.format(args.job_dir))
    else:
        logging.info('Reusing job_dir {} if it exists'.format(args.job_dir))

    train_x, train_y, eval_x, eval_y = utils.load_data(args.train_files,
                                                       args.eval_files)
    # dimensions
    num_train_examples, input_dim = train_x.shape
    num_eval_examples = eval_x.shape[0]

    # Create the Keras Model
    keras_model = model.create_keras_model(
        input_dim=input_dim, learning_rate=args.learning_rate)

    # Pass a numpy array by passing DataFrame.values
    training_dataset = model.input_fn(
        features=train_x.values,
        labels=train_y,
        shuffle=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)

    # Pass a numpy array by passing DataFrame.values
    validation_dataset = model.input_fn(
        features=eval_x.values,
        labels=eval_y,
        shuffle=False,
        num_epochs=args.num_epochs,
        batch_size=num_eval_examples)

    start_time = time()
    # Set tracking URI
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    # Train model
    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id

        class MlflowCallback(tf.keras.callbacks.Callback):
            # This function will be called after training completes.
            def on_train_end(self, logs=None):
                mlflow.log_param('num_layers', len(self.model.layers))
                mlflow.log_param('optimizer_name',
                                 type(self.model.optimizer).__name__)

        mlflow_callback = MlflowCallback()
        # Setup Learning Rate decay.
        lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
            verbose=False)
        # Setup TensorBoard callback.
        tensorboard_path = os.path.join(args.job_dir, run_id, 'tensorboard')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            tensorboard_path,
            histogram_freq=1)
        history = keras_model.fit(
            training_dataset,
            steps_per_epoch=int(num_train_examples / args.batch_size),
            epochs=args.num_epochs,
            validation_data=validation_dataset,
            validation_steps=args.eval_steps,
            verbose=1,
            callbacks=[lr_decay_callback, tensorboard_callback,
                       mlflow_callback])
        metrics = history.history
        logging.info(metrics)
        keras_model.summary()
        mlflow.log_param('train_files', args.train_files)
        mlflow.log_param('eval_files', args.eval_files)
        mlflow.log_param('num_epochs', args.num_epochs)
        mlflow.log_param('batch_size', args.batch_size)
        mlflow.log_param('learning_rate', args.learning_rate)
        mlflow.log_param('train_samples', num_train_examples)
        mlflow.log_param('eval_samples', num_eval_examples)
        mlflow.log_param('eval_steps', args.eval_steps)
        mlflow.log_param('steps_per_epoch',
                         int(num_train_examples / args.batch_size))
        # Add metrics
        _mlflow_log_metrics(metrics, 'loss')
        _mlflow_log_metrics(metrics, 'acc')
        _mlflow_log_metrics(metrics, 'val_loss')
        _mlflow_log_metrics(metrics, 'val_acc')
        _mlflow_log_metrics(metrics, 'lr')
        # Export SavedModel
        model_local_path = os.path.join(args.job_dir, run_id, 'model')
        tf.keras.experimental.export_saved_model(keras_model, model_local_path)
        # Define artifacts.
        logging.info('Model exported to: {}'.format(model_local_path))
        # MLflow workaround since is unable to read GCS path.
        # https://github.com/mlflow/mlflow/issues/1765
        if model_local_path.startswith('gs://'):
            logging.info('Creating temp folder')
            temp = tempfile.mkdtemp()
            model_deployment.copy_artifacts(model_local_path, temp)
            model_local_path = os.path.join(temp, 'model')

        if tensorboard_path.startswith('gs://'):
            logging.info('Creating temp folder')
            temp = tempfile.mkdtemp()
            model_deployment.copy_artifacts(tensorboard_path, temp)
            tensorboard_path = temp

        mlflow.tensorflow.log_model(tf_saved_model_dir=model_local_path,
                                    tf_meta_graph_tags=[tag_constants.SERVING],
                                    tf_signature_def_key='serving_default',
                                    artifact_path='model')
        # Reloading the model
        pyfunc_model = mlflow.pyfunc.load_model(
            mlflow.get_artifact_uri('model'))
        logging.info('Uploading TensorFlow events as a run artifact.')
        mlflow.log_artifacts(tensorboard_path)
        logging.info(
            'Launch TensorBoard with:\n\ntensorboard --logdir=%s' %
            tensorboard_path)
        duration = time() - start_time
        mlflow.log_metric('duration', duration)
        mlflow.end_run()
        if model_local_path.startswith('gs://') and tensorboard_path.startswith(
            'gs://'):
            shutil.rmtree(model_local_path)
            shutil.rmtree(tensorboard_path)

    # Deploy to AI Platform.
    if args.deploy_gcp:
        # Create AI Platform helper instance.
        model_helper = model_deployment.AIPlatformModel(
            project_id=args.project_id)
        # Copy local model to GCS for deployment.
        if not model_local_path.startswith('gs://'):
            model_gcs_path = os.path.join('gs://', args.gcs_bucket, run_id,
                                          'model')
            model_deployment.copy_artifacts(model_local_path, model_gcs_path)
        # Create model
        model_helper.create_model(args.model_name)
        # Create model version
        model_helper.deploy_model(model_gcs_path, args.model_name, run_id,
                                  args.run_time_version)
        logging.info('Model deployment in GCP completed')
    logging.info(
        'This model took: {} seconds to train and test.'.format(duration))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
