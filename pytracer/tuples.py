import tensorflow as tf
import numpy as np


def point(x, y, z):
    return tf.constant(np.array([x, y, z, 1.0]))

def vector(x, y, z):
    return tf.constant(np.array([x, y, z, 0.0]))
