import tensorflow as tf
import numpy as np


class Canvas:
    def __init__(self, width, height, pixels=None):
        self._width = width
        self._height = height
        if pixels is None:
            self._pixels = tf.zeros((width, height, 3))
        else:
            self._pixels = pixels

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def pixels(self):
        return self._pixels

    def write_pixel(self, x, y, color):
        with tf.Session() as sess:
            colors, color = sess.run([self._pixels, color])

        colors[x, y] = color
        self._pixels = tf.constant(colors)
