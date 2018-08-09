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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        default='/tmp/cmle'
    )
    parser.add_argument(
        '--steps',
        default=300)

    args, unused_args = parser.parse_known_args()

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.job_dir)
    estimator.train(train_input_fn, steps=args.steps)


