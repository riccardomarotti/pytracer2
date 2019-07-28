from pytracer.tuples import point, vector, normalize, dot, cross
import tensorflow as tf
import numpy as np
import math


def test_a_point_is_an_array_with_w_set_to_1():
    actual_point = point(4.3, -4.2, 3.1)
    expected_point = tf.constant(
        np.array([4.3, -4.2, 3.1, 1.0]),  dtype=tf.float32)

    with tf.Session() as sess:
        actual, expected = sess.run([actual_point, expected_point])

    assert((actual == expected).all())


def test_a_vector_is_an_array_with_w_set_to_0():
    actual_vector = vector(4.3, -4.2, 3.1)
    expected_vector = tf.constant(
        np.array([4.3, -4.2, 3.1, 0]),  dtype=tf.float32)

    with tf.Session() as sess:
        actual, expected = sess.run([actual_vector, expected_vector])

    assert((actual == expected).all())


def test_sum_of_two_vectors_is_a_vector():
    v1 = vector(3, -2, 5)
    v2 = vector(-2, 3, 1)
    expected_vector = vector(1, 1, 6)

    with tf.Session() as sess:
        actual, expected = sess.run([v1 + v2, expected_vector])

    assert((actual == expected).all())


def test_sum_of_vector_and_point_is_a_point():
    v = vector(3, -2, 5)
    p = point(-2, 3, 1)
    expected_point = point(1, 1, 6)

    with tf.Session() as sess:
        actual, expected = sess.run([v + p, expected_point])

    assert((actual == expected).all())


def test_difference_of_two_point_is_a_vector():
    p1 = point(3, 2, 1)
    p2 = point(5, 6, 7)
    expected_vector = vector(-2, -4, -6)

    with tf.Session() as sess:
        actual, expected = sess.run([p1-p2, expected_vector])

    assert((actual == expected).all())


def test_difference_of_a_vector_and_a_point_is_a_point():
    p = point(3, 2, 1)
    v = vector(5, 6, 7)
    expected_point = point(-2, -4, -6)

    with tf.Session() as sess:
        actual, expected = sess.run([p-v, expected_point])

    assert((actual == expected).all())


def test_difference_of_two_vectors_is_a_vector():
    v1 = vector(3, 2, 1)
    v2 = vector(5, 6, 7)
    expected_vector = vector(-2, -4, -6)

    with tf.Session() as sess:
        actual, expected = sess.run([v1-v2, expected_vector])

    assert((actual == expected).all())


def test_multiply_a_vector_by_a_scalar():
    v = vector(1, -2, 3)
    scalar = 3.5
    expected_vector = vector(3.5, -7, 10.5)

    with tf.Session() as sess:
        actual, expected = sess.run([v * scalar, expected_vector])

    assert((actual == expected).all())


def test_multiply_a_vector_by_a_fraction():
    v = vector(1, -2, 3)
    scalar = 0.5
    expected_vector = vector(0.5, -1, 1.5)

    with tf.Session() as sess:
        actual, expected = sess.run([v * scalar, expected_vector])

    assert((actual == expected).all())


def test_dividing_a_vector_by_a_scalar():
    v = vector(1, -2, 3)
    scalar = 2
    expected_vector = vector(0.5, -1, 1.5)

    with tf.Session() as sess:
        actual, expected = sess.run([v / scalar, expected_vector])

    assert((actual == expected).all())


def test_computing_the_magnitude_of_vector():
    actual_magnitude = tf.norm(vector(-1, -2, -3))
    expected_magnitude = tf.constant(math.sqrt(14),  dtype=tf.float32)

    with tf.Session() as sess:
        actual, expected = sess.run([actual_magnitude, expected_magnitude])

    np.testing.assert_almost_equal(actual, expected, 7)


def test_normalize_vector():
    v = vector(1, 2, 3)
    expected_vector = vector(1. / math.sqrt(14),
                             2. / math.sqrt(14),
                             3. / math.sqrt(14))
    actual_vector = normalize(v)

    with tf.Session() as sess:
        actual, expected = sess.run([actual_vector, expected_vector])

    np.testing.assert_almost_equal(actual, expected, 7)


def test_dot_product_of_vectors():
    v1 = vector(1, 2, 3)
    v2 = vector(2, 3, 4)
    actual_dot_product = dot(v1, v2)

    with tf.Session() as sess:
        actual = sess.run(actual_dot_product)

    assert(actual == 20)


def test_cross_product_of_vectors():
    v1 = vector(1, 2, 3)
    v2 = vector(2, 3, 4)
    actual_cross_product = cross(v1, v2)
    expected_cross_product = vector(-1, 2, -1)

    with tf.Session() as sess:
        actual, expected = sess.run(
            [actual_cross_product, expected_cross_product])

    assert((actual == expected).all())
