class Ray:
    def __init__(self, origin, direction):
        self._origin = origin
        self._direction = direction

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    def position(self, distance):
        return self.origin + self.direction*distance

    def transform(self, transform):
        return Ray(transform(self.origin), transform(self.direction))
