import tensorflow as tf
import numpy as np
from pytracer.tuples import point, vector


def identity_matrix():
    return tf.constant(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]),  dtype=tf.float32)


def translation(x, y, z):
    T = tf.constant(np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ]),  dtype=tf.float32)
    return lambda p: tf.tensordot(T, p, axes=1)


def scaling(x, y, z):
    T = tf.constant(np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ]),  dtype=tf.float32)
    return lambda p: tf.tensordot(T, p, axes = 1)

def invert(t):
    T=t(identity_matrix())
    Tinv=tf.linalg.inv(T)
    return lambda p: tf.tensordot(Tinv, p, axes=1)
