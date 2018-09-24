# Copyright 2018 Google LLC
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


import argparse
import tensorflow as tf


def model_fn(features, labels, mode):
    global_step = tf.train.get_or_create_global_step()
    hidden = tf.layers.dense(features, 10, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, 1)

    predictions = outputs
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.nn.l2_loss(outputs - labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )



def train_input_fn():
    fake_features = tf.constant([[1], [2]], dtype=tf.float32)
    fake_labels = tf.constant([[3], [4]], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(
        (fake_features, fake_labels)
    )
    return ds.repeat().batch(10)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        default='/tmp/cmle'
    )
    parser.add_argument(
        '--steps',
        default=300)

    args, unused_args = parser.parse_known_args()
    
    return args



def train(args):
    estimator = tf.estimator.Estimator(model_fn, model_dir=args.job_dir)
    estimator.train(train_input_fn, steps=args.steps)
    
    return estimator



if __name__ == '__main__':
    args = parse_args()

    estimator = train(args)
