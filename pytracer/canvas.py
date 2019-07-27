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

    def add_color_to_pixel(self, x, y, color):
        indices = tf.constant(np.array([[x, y, 0], [x, y, 1], [x, y, 2]]))
        update = tf.SparseTensor(
            indices, [color[0], color[1], color[2]], self.pixels.shape)
        self._pixels = self.pixels + tf.sparse_tensor_to_dense(update)
