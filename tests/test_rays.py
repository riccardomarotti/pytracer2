import tensorflow as tf
from tftracer.tuples import point, vector
from tftracer.rays import Ray


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
