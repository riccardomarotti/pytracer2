from pytracer.tuples import point, vector
import tensorflow as tf
import numpy as np


def test_a_point_is_an_array_with_w_set_to_1():
    actual_point = point(4.3, -4.2, 3.1)
    expected_point = tf.constant(np.array([4.3, -4.2, 3.1, 1.0]))

    with tf.Session() as sess:
        fetches = [actual_point, expected_point]
        result = sess.run(fetches)

    assert((result[0] == result[1]).all())


def test_a_vector_is_an_array_with_w_set_to_0():
    actual_vector = vector(4.3, -4.2, 3.1)
    expected_vector = tf.constant(np.array([4.3, -4.2, 3.1, 0]))

    with tf.Session() as sess:
        fetches = [actual_vector, expected_vector]
        result = sess.run(fetches)

    assert((result[0] == result[1]).all())
