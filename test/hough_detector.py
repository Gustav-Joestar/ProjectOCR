import cv2
import numpy as np
import math


class HoughDetector:

    def __init__(self, image_path):

        self.image_path = image_path

        self.image = None
        self.gray = None
        self.binary = None

        self.lines = []

    def load(self):

        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise FileNotFoundError(self.image_path)

        self.gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

    def preprocess(self):

        self.binary = cv2.adaptiveThreshold(
            self.gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10
        )

    def detect_lines(self):

        self.lines = []

        result = cv2.HoughLinesP(
            self.binary,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=80,
            maxLineGap=20
        )

        if result is None:
            return

        for line in result:

            x1, y1, x2, y2 = line[0]

            self.lines.append(
                (x1, y1, x2, y2)
            )

    def keep_horizontal_vertical(self, angle_tol=5):

        filtered = []

        for x1, y1, x2, y2 in self.lines:

            angle = abs(
                math.degrees(
                    math.atan2(
                        y2 - y1,
                        x2 - x1
                    )
                )
            )

            if angle > 90:
                angle = 180 - angle

            if angle <= angle_tol or angle >= 90 - angle_tol:

                filtered.append(
                    (x1, y1, x2, y2)
                )

        self.lines = filtered

    def save_debug(self, path):

        debug = self.image.copy()

        for x1, y1, x2, y2 in self.lines:

            cv2.line(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        cv2.imwrite(str(path), debug)