from pathlib import Path
import math
import cv2


class LineDetector:

    def __init__(self, image_path):
        self.image_path = Path(image_path)

        self.image = None
        self.gray = None
        self.binary = None

        self.lines = []

    def load(self):
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(self.image_path)

    def preprocess(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.binary = cv2.adaptiveThreshold(
            self.gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )

    def detect_lines(self):
        detector = cv2.createLineSegmentDetector()
        result = detector.detect(self.binary)

        self.lines = []

        if result is None or result[0] is None:
            return

        for line in result[0]:
            x1, y1, x2, y2 = line[0]
            self.lines.append(
                (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )
            )

    def filter_short_lines(self, min_length=30):
        filtered = []

        for x1, y1, x2, y2 in self.lines:
            if math.hypot(x2 - x1, y2 - y1) >= min_length:
                filtered.append((x1, y1, x2, y2))

        self.lines = filtered

    def keep_horizontal_vertical(self, angle_threshold=5):
        filtered = []

        for x1, y1, x2, y2 in self.lines:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            horizontal = angle <= angle_threshold or angle >= 180 - angle_threshold
            vertical = abs(angle - 90) <= angle_threshold

            if horizontal or vertical:
                filtered.append((x1, y1, x2, y2))

        self.lines = filtered

    def merge_lines(self, y_tol=4, x_tol=20):
        """
        Merge horizontal and vertical line segments.

        y_tol - max deviation from one horizontal level
        x_tol - max allowed gap between segments
        """

        horizontal = []
        vertical = []

        # -----------------------------------
        # split
        # -----------------------------------

        for x1, y1, x2, y2 in self.lines:

            if abs(y1 - y2) <= y_tol:

                if x1 > x2:
                    x1, x2 = x2, x1

                horizontal.append((x1, y1, x2, y2))

            elif abs(x1 - x2) <= y_tol:

                if y1 > y2:
                    y1, y2 = y2, y1

                vertical.append((x1, y1, x2, y2))

        horizontal = self._merge_horizontal(horizontal, y_tol, x_tol)
        vertical = self._merge_vertical(vertical, y_tol, x_tol)

        self.lines = horizontal + vertical

    def _merge_horizontal(self, lines, y_tol, gap):

        if not lines:
            return []

        lines.sort(key=lambda l: (l[1], l[0]))

        merged = []

        current = list(lines[0])

        for line in lines[1:]:

            x1, y1, x2, y2 = line

            # другая строка
            if abs(y1 - current[1]) > y_tol:

                merged.append(tuple(current))
                current = list(line)
                continue

            # продолжается
            if x1 <= current[2] + gap:

                current[2] = max(current[2], x2)
                current[1] = int((current[1] + y1) / 2)
                current[3] = current[1]

            else:

                merged.append(tuple(current))
                current = list(line)

        merged.append(tuple(current))

        return merged

    def _merge_vertical(self, lines, x_tol, gap):

        if not lines:
            return []

        lines.sort(key=lambda l: (l[0], l[1]))

        merged = []

        current = list(lines[0])

        for line in lines[1:]:

            x1, y1, x2, y2 = line

            if abs(x1 - current[0]) > x_tol:

                merged.append(tuple(current))
                current = list(line)
                continue

            if y1 <= current[3] + gap:

                current[3] = max(current[3], y2)
                current[0] = int((current[0] + x1) / 2)
                current[2] = current[0]

            else:

                merged.append(tuple(current))
                current = list(line)

        merged.append(tuple(current))

        return merged

    def filter_merged_lines(self, min_length=300):
        """
        Remove merged lines shorter than min_length.
        """

        filtered = []

        for x1, y1, x2, y2 in self.lines:

            length = math.hypot(x2 - x1, y2 - y1)

            if length >= min_length:
                filtered.append((x1, y1, x2, y2))

        self.lines = filtered

    def save_debug(self, output_path):
        debug = self.image.copy()

        for x1, y1, x2, y2 in self.lines:
            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), debug)

    def run(self):
        self.load()
        self.preprocess()
        self.detect_lines()
        self.filter_short_lines(30)
        self.keep_horizontal_vertical(5)
        self.merge_lines()
        self.filter_merged_lines(300)

        return self.lines