from pytracer.spheres import Sphere
from pytracer import transformations
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
