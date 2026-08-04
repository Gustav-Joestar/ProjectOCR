from dataclasses import dataclass
import math


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length(self):
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def horizontal(self):
        return abs(self.y1 - self.y2) <= 3

    @property
    def vertical(self):
        return abs(self.x1 - self.x2) <= 3


class FrameDetector:

    def __init__(self, lines):

        self.lines = [Line(*l) for l in lines]

        self.horizontal = []
        self.vertical = []

    def split_lines(self):

        self.horizontal.clear()
        self.vertical.clear()

        for line in self.lines:

            if line.horizontal:
                self.horizontal.append(line)

            elif line.vertical:
                self.vertical.append(line)

        self.horizontal.sort(
            key=lambda l: l.length,
            reverse=True
        )

        self.vertical.sort(
            key=lambda l: l.length,
            reverse=True
        )


    def run(self):

        self.split_lines()

        return self.horizontal, self.vertical