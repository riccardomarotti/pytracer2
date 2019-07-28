from pytracer.transformations import identity_matrix


class Sphere:
    def __init__(self, transformation=identity_matrix):
        self._transformation = transformation

    @property
    def transformation(self):
        return self._transformation
