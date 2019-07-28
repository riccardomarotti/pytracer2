from pytracer.spheres import Sphere
from pytracer import transformations
from pytracer.rays import Ray
from pytracer.tuples import point, vector
import numpy as np
import tensorflow as tf


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
