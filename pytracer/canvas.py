import tensorflow as tf
import numpy as np


class Canvas:
    def __init__(self, width, height, pixels=None):
        self._width = width
        self._height = height
        if pixels is None:
            self._pixels = tf.zeros((width, height, 3))
        else:
            self._pixels = tf.constant(pixels)

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

    def to_PPM(self):
        ppm_string = "P3\n{} {}\n255\n".format(self.width, self.height)

        with tf.Session() as sess:
            actualCanvas = sess.run(self.pixels)

        for row_id in range(self.height):
            current_line = ""
            colors = actualCanvas[:, row_id]
            clamped_colors = clamp(colors.flatten())
            current_line = " ".join(clamped_colors.astype(str))
            ppm_string += truncate(current_line) + "\n"

        return ppm_string


@np.vectorize
def clamp(color):
    return int(max(0, min(255, color * 255)))


def truncate(s):
    MAX = 70
    x = len(s)
    if(x) < MAX:
        return s

    last_space_index = s[:MAX+1].rfind(' ')
    if last_space_index != -1:
        return s[:last_space_index] + "\n" + s[last_space_index+1:]
