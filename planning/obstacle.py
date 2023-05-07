

class Obstacle:
    def __init__(self, center, side_length, height, altitude_start=0):

        self.vertices = [
            (center[0] - side_length / 2, center[1] - side_length / 2, altitude_start + height),
            (center[0] + side_length / 2, center[1] - side_length / 2, altitude_start + height),
            (center[0] + side_length / 2, center[1] + side_length / 2, altitude_start + height),
            (center[0] - side_length / 2, center[1] + side_length / 2, altitude_start + height),
            (center[0] - side_length / 2, center[1] - side_length / 2, altitude_start),
            (center[0] + side_length / 2, center[1] - side_length / 2, altitude_start),
            (center[0] + side_length / 2, center[1] + side_length / 2, altitude_start),
            (center[0] - side_length / 2, center[1] + side_length / 2, altitude_start)
        ]

        self.edges = [
            (self.vertices[0], self.vertices[1]),
            (self.vertices[1], self.vertices[2]),
            (self.vertices[2], self.vertices[3]),
            (self.vertices[3], self.vertices[0]),
            (self.vertices[4], self.vertices[5]),
            (self.vertices[5], self.vertices[6]),
            (self.vertices[6], self.vertices[7]),
            (self.vertices[7], self.vertices[4]),
            (self.vertices[0], self.vertices[4]),
            (self.vertices[1], self.vertices[5]),
            (self.vertices[2], self.vertices[6]),
            (self.vertices[3], self.vertices[7])
        ]



if __name__ == "__main__":
    pass
