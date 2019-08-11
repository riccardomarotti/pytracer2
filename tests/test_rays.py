import tensorflow as tf
import numpy as np

from tftracer.tuples import point, vector
from tftracer.rays import Ray
from tftracer.transformations import translation
import tftracer.transformations


def test_computing_a_point_from_a_distance():
    origin = point(2, 3, 4)
    direction = vector(1, 0, 0)
    r = Ray(origin, direction)

    with tf.Session() as sess:
        [expected1, expected2, expected3, expected4, actual1, actual2, actual3, actual4] = sess.run(
            [point(2, 3, 4), point(3, 3, 4), point(1, 3, 4), point(4.5, 3, 4), r.position(0), r.position(1), r.position(-1), r.position(2.5)])

    assert((expected1 == actual1).all())
    assert((expected2 == actual2).all())
    assert((expected3 == actual3).all())
    assert((expected4 == actual4).all())


def test_translating_a_ray():
    origin = point(1, 2, 3)
    direction = vector(0, 1, 0)
    r = Ray(origin, direction)
    expected_origin = point(4, 6, 8)
    expected_direction = vector(0, 1, 0)

    m = tftracer.transformations.translation(3, 4, 5)

    with tf.Session() as sess:
        [actual_origin, actual_direction, expected_origin,
            expected_direction] = sess.run([r.transform(m).origin, r.transform(m).direction, expected_origin, expected_direction])

    assert(np.allclose(actual_origin, expected_origin))
    assert(np.allclose(actual_direction, expected_direction))
