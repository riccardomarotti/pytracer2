import tensorflow as tf
import numpy as np
from pytracer import transformations
from pytracer.tuples import point, vector


def test_multiplying__by_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    p = point(-3, 4, 5)

    expected_point = point(2, 1, 7)

    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_point, transform(p)])

    assert((expected == actual).all())


def test_multiplying_by_the_inverse_of_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    transform = transformations.invert(transform)
    p = point(-3, 4, 5)
    expected_point = point(-8, 7, 3)

    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_point, transform(p)])

    assert((expected == actual).all())


def test_translation_does_not_affect_vectors():
    transform = transformations.translation(5, -3, 2)
    expected_vector = vector(-3, 4, 5)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [expected_vector, transform(expected_vector)])

    assert((expected == actual).all())
