import tensorflow as tf
import numpy as np


def color(r, g, b):
    return tf.constant(np.array([r, g, b]))
