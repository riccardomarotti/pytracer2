import tensorflow as tf


def hit(xs):
    mask = tf.greater(xs, 0)
    return tf.reduce_min(tf.boolean_mask(xs, mask))
