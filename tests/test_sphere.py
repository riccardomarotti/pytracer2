from tftracer.spheres import Sphere
from tftracer import transformations
from tftracer.rays import Ray
from tftracer.tuples import point, vector
import numpy as np
import tensorflow as tf
import math


def test_identity_is_a_sphere_default_transformation():
    sphere = Sphere()
    expected_transformation = transformations.identity_matrix
    actual_transformation = sphere.transformation

    with tf.Session() as sess:
        [expected, actual] = sess.run(
            [expected_transformation(), actual_transformation()])

    assert(np.array_equal(expected, actual))


def test_a_ray_intersects_a_sphere_at_two_points():
    xs = Sphere().intersect(Ray(point(0, 0, -5), vector(0, 0, 1)))

    with tf.Session() as sess:
        [actual_xs] = sess.run([xs])

    assert(len(actual_xs) == 2)
    assert(actual_xs[0] == 4.0)
    assert(actual_xs[1] == 6.0)


def test_a_ray_intersects_a_sphere_at_a_tangent():
    xs = Sphere().intersect(Ray(point(0, 1, -5), vector(0, 0, 1)))

    with tf.Session() as sess:
        [actual_xs] = sess.run([xs])

    assert(len(actual_xs) == 2)
    assert(actual_xs[0] == 5.0)
    assert(actual_xs[1] == 5.0)


def test_a_ray_misses_a_sphere():
    xs = Sphere().intersect(Ray(point(0, 2, -5), vector(0, 0, 1)))

    with tf.Session() as sess:
        [actual_xs] = sess.run([xs])

    assert(len(actual_xs) == 2)
    assert(math.isnan(actual_xs[0]))
    assert(math.isnan(actual_xs[1]))


def test_a_ray_originates_inside_a_sphere():
    xs = Sphere().intersect(Ray(point(0, 0, 0), vector(0, 0, 1)))

    with tf.Session() as sess:
        [actual_xs] = sess.run([xs])

    assert(len(actual_xs) == 2)
    assert(actual_xs[0] == -1.0)
    assert(actual_xs[1] == 1.0)


def test_a_sphere_behind_a_ary():
    xs = Sphere().intersect(Ray(point(0, 0, 5), vector(0, 0, 1)))

    with tf.Session() as sess:
        [actual_xs] = sess.run([xs])

    assert(len(actual_xs) == 2)
    assert(actual_xs[0] == -6.0)
    assert(actual_xs[1] == -4.0)


def test_intersecting_a_scaled_sphere_with_a_ray():
    r = Ray(point(0., 0., -5.), vector(0., 0., 1.))
    s = Sphere(transformations.scaling(2, 2, 2))

    xs = s.intersect(r)

    with tf.Session() as sess:
        actual_xs = sess.run(xs)

    assert(len(actual_xs) == 2)
    assert(actual_xs[0] == 3)
    assert(actual_xs[1] == 7)