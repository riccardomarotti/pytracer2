from tftracer.transformations import identity_matrix
from tftracer.transformations import invert
from tftracer.tuples import point, dot
import numpy as np
import tensorflow as tf


class Sphere:
    def __init__(self, transformation=identity_matrix):
        self._transformation = transformation

    @property
    def transformation(self):
        return self._transformation

    def intersect(self, ray):
        transformation = invert(self.transformation)
        transformed_ray = ray.transform(transformation)
        return perform_intersection(transformed_ray.origin, transformed_ray.direction)


def perform_intersection(origin, direction):
    sphere_to_ray = origin - point(0, 0, 0)

    a = dot(direction, direction)
    b = 2*dot(direction, sphere_to_ray)
    c = dot(sphere_to_ray, sphere_to_ray) - 1

    delta = b**2 - 4*a*c

    t1 = (-b - tf.sqrt(delta)) / tf.scalar_mul(2, a)
    t2 = (-b + tf.sqrt(delta)) / tf.scalar_mul(2, a)

    return tf.stack([t1, t2])
