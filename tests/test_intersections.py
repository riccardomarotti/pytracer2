from pytracer.intersections import hit
from pytracer.spheres import Sphere

import tensorflow as tf
import numpy as np


def test_hit_when_all_intersections_are_positive():
    xs = tf.constant(np.array([1., 2.]), dtype=tf.float32)
    i = hit(xs)

    with tf.Session() as sess:
        actual_hit = sess.run(i)

    assert(actual_hit == 1.)


def test_hit_when_some_intersections_are_negative():
    xs = tf.constant(np.array([-1., 1.]), dtype=tf.float32)
    i = hit(xs)

    with tf.Session() as sess:
        actual_hit = sess.run(i)

    assert(actual_hit == 1.)


def test_hit_when_all_intersections_are_negative():
    xs = tf.constant(np.array([-2., -1.]), dtype=tf.float32)
    i = hit(xs)

    with tf.Session() as sess:
        actual_hit = sess.run(i)

    assert(actual_hit == np.inf)


def test_hit_is_always_the_lowest_nonnegative_intersection():
    xs = tf.constant(np.array([5., 7., -3., 2.]), dtype=tf.float32)
    i = hit(xs)

    with tf.Session() as sess:
        actual_hit = sess.run(i)

    assert(actual_hit == 2.)
