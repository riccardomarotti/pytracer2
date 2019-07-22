import tensorflow as tf
import numpy as np


class Canvas:
    def __init__(self, width, height, pixels=None):
        self._width = width
        self._height = height
        if pixels is None:
            self._pixels = np.zeros((width, height, 3))
        else:
            self._pixels = pixels

    def pixels_tensor(self):
        return tf.constant(self._pixels)

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
            color = sess.run(color)
        self.pixels[x, y] = color
