from pytracer.canvas import Canvas
from pytracer.colors import color, red
import tensorflow as tf


def test_creating_canvas():
    c = Canvas(10, 20)

    assert(c.width == 10)
    assert(c.height == 20)

    expected_colors = tf.zeros((10, 20, 3))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        actual, expected = sess.run([c.pixels, expected_colors])

    assert((actual == expected).all())


def test_writing_pixels_to_a_canvas():
    c = Canvas(10, 20)
    c.add_color_to_pixel(2, 3, red())

    with tf.Session() as sess:
        actual_colors = sess.run(c.pixels)

    assert((actual_colors[2, 3] == (1, 0, 0)).all())
