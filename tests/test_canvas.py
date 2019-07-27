from pytracer.canvas import Canvas
from pytracer.colors import color, red
import tensorflow as tf
import numpy as np


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


def test_constructing_the_PPM_header():
    actualPPM = Canvas(5, 3).to_PPM()

    expectedPPM = """P3
5 3
255"""

    assert(actualPPM.startswith(expectedPPM))


def test_constructing_the_PPM_pixel_data():
    c = Canvas(5, 3)
    c1 = color(1.5, 0., 0.)
    c2 = color(0., 0.5, 0.)
    c3 = color(-0.5, 0., 1.)

    c.add_color_to_pixel(0, 0, c1)
    c.add_color_to_pixel(2, 1, c2)
    c.add_color_to_pixel(4, 2, c3)

    actualPPM = c.to_PPM()
    expectedPPM = """P3
5 3
255
255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 127 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"""

    assert(expectedPPM == actualPPM)


def test_splitting_long_lines_in_PPM_files():
    canvas = Canvas(10, 2, np.ones((10, 2, 3))*[1, 0.8, 0.6])
    actualPPM = canvas.to_PPM()

    expectedPPM = """255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153"""

    assert(len(actualPPM.split("\n")) == 8)
    actualPPM = "\n".join(actualPPM.split("\n")[3:7])

    assert(expectedPPM == actualPPM)
