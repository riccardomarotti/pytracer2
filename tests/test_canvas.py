from pytracer.canvas import Canvas
from pytracer.colors import color
import tensorflow as tf


def test_creating_canvas():
    c = Canvas(10, 20)

    assert(c.width == 10)
    assert(c.height == 20)

    expected_colors = tf.zeros((10, 20, 3))

    with tf.Session() as sess:
        actual, expected = sess.run([c.pixels_tensor(), expected_colors])

    assert((actual == expected).all())


def test_writing_pixels_to_a_canvas():
    c = Canvas(10, 20)
    red = color(1, 0, 0)

    c.write_pixel(2, 3, red)

    with tf.Session() as sess:
        actual_colors = sess.run(c.pixels_tensor())

    assert((actual_colors[2, 3] == (1, 0, 0)).all())
