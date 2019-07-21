import tensorflow as tf
import numpy as np


def point(x, y, z):
    return tf.constant(np.array([x, y, z, 1.0]))


def vector(x, y, z):
    return tf.constant(np.array([x, y, z, 0.0]))


def normalize(v):
    return v / tf.norm(v)


def dot(v1, v2):
    return tf.tensordot(v1, v2, axes=1)


def cross(v1, v2):
    cross = tf.cross(v1[:3], v2[:3])
    zero = tf.constant(np.array([0.]))
    return tf.concat([cross, zero], -1)
