from pytracer.tuples import point, vector
import tensorflow as tf
import numpy as np


def test_a_point_is_an_array_with_w_set_to_1():
    actual_point = point(4.3, -4.2, 3.1)
    expected_point = tf.constant(np.array([4.3, -4.2, 3.1, 1.0]))

    with tf.Session() as sess:
        result = sess.run([actual_point, expected_point])

    assert((result[0] == result[1]).all())


def test_a_vector_is_an_array_with_w_set_to_0():
    actual_vector = vector(4.3, -4.2, 3.1)
    expected_vector = tf.constant(np.array([4.3, -4.2, 3.1, 0]))

    with tf.Session() as sess:
        result = sess.run([actual_vector, expected_vector])

    assert((result[0] == result[1]).all())


def test_sum_of_two_vectors_is_a_vector():
    v1 = vector(3, -2, 5)
    v2 = vector(-2, 3, 1)
    expected_vector = vector(1, 1, 6)

    with tf.Session() as sess:
        result = sess.run([v1 + v2, expected_vector])

    assert((result[0] == result[1]).all())


def test_sum_of_vector_and_point_is_a_point():
    v = vector(3, -2, 5)
    p = point(-2, 3, 1)
    expected_point = point(1, 1, 6)

    with tf.Session() as sess:
        result = sess.run([v + p, expected_point])

    assert((result[0] == result[1]).all())


def test_difference_of_two_point_is_a_vector():
    p1 = point(3, 2, 1)
    p2 = point(5, 6, 7)
    expected_vector = vector(-2, -4, -6)

    with tf.Session() as sess:
        result = sess.run([p1-p2, expected_vector])

    assert((result[0] == result[1]).all())


def test_difference_of_a_vector_and_a_point_is_a_point():
    p = point(3, 2, 1)
    v = vector(5, 6, 7)
    expected_point = point(-2, -4, -6)

    with tf.Session() as sess:
        result = sess.run([p-v, expected_point])

    assert((result[0] == result[1]).all())


def test_difference_of_two_vectors_is_a_vector():
    v1 = vector(3, 2, 1)
    v2 = vector(5, 6, 7)
    expected_vector = vector(-2, -4, -6)

    with tf.Session() as sess:
        result = sess.run([v1-v2, expected_vector])

    assert((result[0] == result[1]).all())
