from pathlib import Path

import cv2
import numpy as np


class StampSegmenter:
    def __init__(self, image_source):
        self.image_source = image_source

        self.image = None
        self.binary = None
        self.horizontal_mask = None
        self.vertical_mask = None
        self.grid_mask = None

    def load(self):
        """Загружает изображение из файла или принимает ndarray из памяти."""

        if isinstance(self.image_source, (str, Path)):
            self.image = cv2.imread(
                str(self.image_source)
            )

            if self.image is None:
                raise FileNotFoundError(
                    f"Не удалось загрузить изображение: {self.image_source}"
                )

        elif isinstance(self.image_source, np.ndarray):
            if self.image_source.size == 0:
                raise ValueError(
                    "Передано пустое изображение штампа."
                )

            self.image = self.image_source.copy()

        else:
            raise TypeError(
                "image_source должен быть путём "
                "к изображению или numpy.ndarray."
            )

        return self.image

    def preprocess(self):
        """Создаёт бинарное изображение."""

        if self.image is None:
            raise RuntimeError("Сначала необходимо вызвать load().")

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        self.binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        return self.binary

    def detect_grid(self):
        """Выделяет горизонтальные и вертикальные линии сетки."""

        if self.binary is None:
            raise RuntimeError(
                "Сначала необходимо вызвать preprocess()."
            )

        height, width = self.binary.shape

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(10, width // 30), 1)
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(10, height // 20))
        )

        self.horizontal_mask = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            horizontal_kernel
        )

        self.vertical_mask = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            vertical_kernel
        )

        self.grid_mask = cv2.bitwise_or(
            self.horizontal_mask,
            self.vertical_mask
        )

        return self.grid_mask

    def _detect_segments(self, mask, orientation):
        """Находит длинные горизонтальные или вертикальные отрезки."""

        if mask is None:
            raise RuntimeError(
                "Сначала необходимо вызвать detect_grid()."
            )

        height, width = mask.shape

        if orientation == "horizontal":
            threshold = 50
            min_length = int(width * 0.04)

        elif orientation == "vertical":
            threshold = 30
            min_length = int(height * 0.08)

        else:
            raise ValueError(
                "orientation должен быть "
                "'horizontal' или 'vertical'."
            )

        lines = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=10
        )

        if lines is None:
            return []

        segments = []

        for x1, y1, x2, y2 in lines[:, 0]:
            x1, y1, x2, y2 = map(
                int,
                (x1, y1, x2, y2)
            )

            if orientation == "horizontal":
                if abs(y2 - y1) > 3:
                    continue

                y = round((y1 + y2) / 2)

                segments.append(
                    (
                        min(x1, x2),
                        y,
                        max(x1, x2),
                        y
                    )
                )

            else:
                if abs(x2 - x1) > 3:
                    continue

                x = round((x1 + x2) / 2)

                segments.append(
                    (
                        x,
                        min(y1, y2),
                        x,
                        max(y1, y2)
                    )
                )

        return segments

    def merge_segments(
        self,
        segments,
        orientation,
        coord_tolerance=8,
        gap_tolerance=20
    ):
        """Объединяет близкие отрезки одной физической линии."""

        if not segments:
            return []

        if orientation == "vertical":
            rotated = [
                (y1, x, y2, x)
                for x, y1, _, y2 in segments
            ]

            merged = self.merge_segments(
                rotated,
                "horizontal",
                coord_tolerance,
                gap_tolerance
            )

            return [
                (y, x1, y, x2)
                for x1, y, x2, _ in merged
            ]

        if orientation != "horizontal":
            raise ValueError(
                "orientation должен быть "
                "'horizontal' или 'vertical'."
            )

        segments = sorted(
            segments,
            key=lambda segment: (
                segment[1],
                segment[0]
            )
        )

        groups = []

        for segment in segments:
            y = segment[1]

            for group in groups:
                group_y = round(
                    sum(item[1] for item in group)
                    / len(group)
                )

                if abs(y - group_y) <= coord_tolerance:
                    group.append(segment)
                    break
            else:
                groups.append([segment])

        merged = []

        for group in groups:
            y = round(
                sum(item[1] for item in group)
                / len(group)
            )

            ranges = sorted(
                (item[0], item[2])
                for item in group
            )

            start, end = ranges[0]

            for x1, x2 in ranges[1:]:
                if x1 <= end + gap_tolerance:
                    end = max(end, x2)
                else:
                    merged.append(
                        (start, y, end, y)
                    )
                    start, end = x1, x2

            merged.append(
                (start, y, end, y)
            )

        return merged

    def find_horizontal_segments(self):
        """Возвращает объединённые горизонтальные отрезки."""

        segments = self._detect_segments(
            self.horizontal_mask,
            "horizontal"
        )

        return self.merge_segments(
            segments,
            "horizontal"
        )

    def find_vertical_segments(self):
        """Возвращает объединённые вертикальные отрезки."""

        segments = self._detect_segments(
            self.vertical_mask,
            "vertical"
        )

        return self.merge_segments(
            segments,
            "vertical"
        )

    def find_intersections(self, tolerance=8):
        """Находит реальные пересечения сегментов сетки."""

        horizontal = self.find_horizontal_segments()
        vertical = self.find_vertical_segments()

        intersections = []

        for x1, y, x2, _ in horizontal:
            for x, y1, _, y2 in vertical:
                if (
                    x1 - tolerance <= x <= x2 + tolerance
                    and
                    y1 - tolerance <= y <= y2 + tolerance
                ):
                    intersections.append(
                        (x, y)
                    )

        return sorted(
            set(intersections),
            key=lambda point: (
                point[1],
                point[0]
            )
        )

    def find_cells(self, tolerance=8):
        """Находит минимальные замкнутые ячейки сетки."""

        horizontal = self.find_horizontal_segments()
        vertical = self.find_vertical_segments()
        intersections = self.find_intersections(
            tolerance
        )

        points = set(intersections)
        cells = []

        def has_horizontal(y, x1, x2):
            return any(
                abs(line_y - y) <= tolerance
                and line_x1 <= x1 + tolerance
                and line_x2 >= x2 - tolerance
                for line_x1, line_y, line_x2, _ in horizontal
            )

        def has_vertical(x, y1, y2):
            return any(
                abs(line_x - x) <= tolerance
                and line_y1 <= y1 + tolerance
                and line_y2 >= y2 - tolerance
                for line_x, line_y1, _, line_y2 in vertical
            )

        for left, top in intersections:
            rights = sorted(
                x
                for x, y in points
                if y == top and x > left
            )

            bottoms = sorted(
                y
                for x, y in points
                if x == left and y > top
            )

            for right in rights:
                for bottom in bottoms:
                    if (right, bottom) not in points:
                        continue

                    if not has_horizontal(
                        top,
                        left,
                        right
                    ):
                        continue

                    if not has_horizontal(
                        bottom,
                        left,
                        right
                    ):
                        continue

                    if not has_vertical(
                        left,
                        top,
                        bottom
                    ):
                        continue

                    if not has_vertical(
                        right,
                        top,
                        bottom
                    ):
                        continue

                    inner_vertical = any(
                        left < x < right
                        and has_vertical(
                            x,
                            top,
                            bottom
                        )
                        for x, _, _, _ in vertical
                    )

                    inner_horizontal = any(
                        top < y < bottom
                        and has_horizontal(
                            y,
                            left,
                            right
                        )
                        for _, y, _, _ in horizontal
                    )

                    if inner_vertical or inner_horizontal:
                        continue

                    cells.append(
                        (
                            left,
                            top,
                            right,
                            bottom
                        )
                    )

        return sorted(
            set(cells),
            key=lambda cell: (
                cell[1],
                cell[0]
            )
        )