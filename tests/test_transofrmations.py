import tensorflow as tf
import numpy as np
from tftracer import transformations
from tftracer.tuples import point, vector
import math


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


def test_a_scaling_matrix_applied_to_a_point():
    transform = transformations.scaling(2, 3, 4)
    p = point(-4, 6, 8)
    expected_point = point(-8, 18, 32)

    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_point, transform(p)])

    assert((expected == actual).all())


def test_a_scaling_matrix_applied_to_a_vecor():
    transform = transformations.scaling(2, 3, 4)
    v = vector(-4, 6, 8)
    expected_vector = vector(-8, 18, 32)

    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_vector, transform(v)])

    assert((expected == actual).all())


def test_multiplying_the_inverse_of_a_scaling_matrix():
    transform = transformations.scaling(2, 3, 4)
    inverse = transformations.invert(transform)
    v = vector(-4, 6, 8)
    expected_vector = vector(-2, 2, 2)

    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_vector, inverse(v)])

    assert((expected == actual).all())


def test_rotating_a_point_around_the_x_axis():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_x(math.pi/4)
    full_quarter = transformations.rotation_x(math.pi/2)

    expected_half_quarter = point(0, math.sqrt(2)/2, math.sqrt(2)/2)
    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_half_quarter, half_quarter(p)])

    assert(np.allclose(expected, actual))

    expected_full_quarter = point(0, 0, 1)
    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_full_quarter, full_quarter(p)])

    assert(np.allclose(expected, actual))


def test_the_inverse_of_an_x_rotation_rotates_in_the_opposite_direction():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_x(math.pi/4)
    inverse = transformations.invert(half_quarter)
    expected_rotated = point(0, math.sqrt(2)/2, -math.sqrt(2)/2)

    with tf.Session() as sess:
        [expected, actual] = sess.run([expected_rotated, inverse(p)])

    assert(np.allclose(expected, actual))


def test_rotating_a_point_around_the_y_axis():
    p = point(0, 0, 1)
    half_quarter = transformations.rotation_y(math.pi/4)
    full_quarter = transformations.rotation_y(math.pi/2)
    expected_half_quarted_rotated = point(math.sqrt(2)/2, 0, math.sqrt(2)/2)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [expected_half_quarted_rotated, half_quarter(p)])

    assert(np.allclose(expected, actual))

    expected_full_quarter_rotated = point(1, 0, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [expected_full_quarter_rotated, full_quarter(p)])

    assert(np.allclose(expected, actual))


def test_rotating_a_point_around_the_z_axis():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_z(math.pi/4)
    full_quarter = transformations.rotation_z(math.pi/2)

    expected_half_quarted_rotated = point(-math.sqrt(2)/2, math.sqrt(2)/2, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [expected_half_quarted_rotated, half_quarter(p)])

    assert(np.allclose(expected, actual))

    expected_full_quarter_rotated = point(-1, 0, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [expected_full_quarter_rotated, full_quarter(p)])
    assert(np.allclose(expected, actual))


def test_a_shearing_transformation_moves_x_in_proportion_to_y():
    transform = transformations.shearing(1, 0, 0, 0, 0, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [point(5, 3, 4), transform(point(2, 3, 4))])

    assert((expected == actual).all())


def test_a_shearing_transformation_moves_x_in_proportion_to_z():
    transform = transformations.shearing(0, 1, 0, 0, 0, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [point(6, 3, 4), transform(point(2, 3, 4))])

    assert((expected == actual).all())


def test_a_shearing_transformation_moves_y_in_proportion_to_x():
    transform = transformations.shearing(0, 0, 1, 0, 0, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [point(2, 5, 4), transform(point(2, 3, 4))])

    assert((expected == actual).all())


def test_a_shearing_transformation_moves_y_in_proportion_to_z():
    transform = transformations.shearing(0, 0, 0, 1, 0, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [point(2, 7, 4), transform(point(2, 3, 4))])

    assert((expected == actual).all())


def test_a_shearing_transformation_moves_z_in_proportion_to_x():
    transform = transformations.shearing(0, 0, 0, 0, 1, 0)

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [point(2, 3, 6), transform(point(2, 3, 4))])

    assert((expected == actual).all())


def test_a_shearing_transformation_moves_z_in_proportion_to_y():
    transform = transformations.shearing(0, 0, 0, 0, 0, 1)
    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [point(2, 3, 7), transform(point(2, 3, 4))])

    assert((expected == actual).all())


def test_individual_transformations_are_applied_in_sequence():
    p = point(1, 0, 1)
    A = transformations.rotation_x(math.pi/2)
    B = transformations.scaling(5, 5, 5)
    C = transformations.translation(10, 5, 7)

    with tf.Session() as sess:
        [expected_p2, expected_p3, expected_p4, actual_p2, actual_p3, actual_p4] = sess.run(
            [point(1, -1, 0), point(5, -5, 0), point(15, 0, 7), A(p), B(A(p)), C(B(A(p)))])

    assert(np.allclose(expected_p2, actual_p2))
    assert(np.allclose(expected_p3, actual_p3))
    assert(np.allclose(expected_p4, actual_p4))


def test_chained_transofrmations_must_be_applied_in_reverse_order():
    p = point(1, 0, 1)
    A = transformations.rotation_x(math.pi/2)
    B = transformations.scaling(5, 5, 5)
    C = transformations.translation(10, 5, 7)

    CBA = transformations.concat(C, B, A)

    with tf.Session() as sess:
        [expected, actual] = sess.run([point(15, 0, 7), CBA(p)])

    assert(np.allclose(expected, actual))
