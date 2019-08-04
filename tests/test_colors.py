from tftracer.colors import color
import tensorflow as tf
import numpy as np
import math


def test_adding_colors():
    c1 = color(0.9, 0.6, 0.75)
    c2 = color(0.7, 0.1, 0.25)

    expected_sum = color(1.6, 0.7, 1.0)
    actual_sum = c1 + c2

    with tf.Session() as sess:
        actual, expected = sess.run([actual_sum, expected_sum])

    np.testing.assert_almost_equal(actual, expected)


def test_subtracting_colors():
    c1 = color(0.9, 0.6, 0.75)
    c2 = color(0.7, 0.1, 0.25)

    expected_diff = color(0.2, 0.5, 0.5)
    actual_diff = c1 - c2

    with tf.Session() as sess:
        actual, expected = sess.run([actual_diff, expected_diff])

    np.testing.assert_almost_equal(actual, expected, 3)


def test_multiplying_a_color_by_a_scalar():
    c = color(0.2, 0.3, 0.4)

    expected_color = color(0.4, 0.6, 0.8)
    actual_color = c*2

    with tf.Session() as sess:
        actual, expected = sess.run([actual_color, expected_color])

    np.testing.assert_almost_equal(actual, expected, 3)


def test_multiplying_colors():
    c1 = color(1.0, 0.2, 0.4)
    c2 = color(0.9, 1.0, 0.1)

    expected_mul = color(0.9, 0.2, 0.04)
    actual_mul = c1 * c2

    with tf.Session() as sess:
        actual, expected = sess.run([actual_mul, expected_mul])

    np.testing.assert_almost_equal(actual, expected, 3)
