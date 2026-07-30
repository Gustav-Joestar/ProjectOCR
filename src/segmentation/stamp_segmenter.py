from pathlib import Path

import cv2
import numpy as np


class StampSegmenter:
    def __init__(self, image_path):
        self.image_path = Path(image_path)

        self.image = None
        self.binary = None
        self.horizontal_mask = None
        self.vertical_mask = None
        self.grid_mask = None

    def load(self):
        """Загружает изображение штампа."""

        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise FileNotFoundError(
                f"Не удалось загрузить изображение: {self.image_path}"
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

    def extract_cells(self, padding=3):
        """
        Вырезает найденные ячейки из исходного изображения.

        Возвращает список словарей:
        {
            "index": номер ячейки,
            "bbox": (x1, y1, x2, y2),
            "image": изображение ячейки
        }
        """

        if self.image is None:
            raise RuntimeError(
                "Сначала необходимо вызвать load()."
            )

        cells = self.find_cells()

        height, width = self.image.shape[:2]

        result = []

        for index, (x1, y1, x2, y2) in enumerate(
            cells,
            start=1
        ):
            # Чуть отступаем внутрь от линий сетки.
            crop_x1 = min(width, x1 + padding)
            crop_y1 = min(height, y1 + padding)

            crop_x2 = max(0, x2 - padding)
            crop_y2 = max(0, y2 - padding)

            if (
                crop_x2 <= crop_x1
                or crop_y2 <= crop_y1
            ):
                continue

            crop = self.image[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ].copy()

            result.append({
                "index": index,
                "bbox": (x1, y1, x2, y2),
                "image": crop,
            })

        return result

    def save_debug(self, output_dir):
        """Сохраняет маски этапов обработки."""

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        images = {
            "01_binary.png": self.binary,
            "02_horizontal.png": self.horizontal_mask,
            "03_vertical.png": self.vertical_mask,
            "04_grid.png": self.grid_mask,
        }

        for name, image in images.items():
            if image is not None:
                cv2.imwrite(
                    str(output_dir / name),
                    image
                )

    def save_segments_debug(self, output_path):
        """Рисует найденные сегменты сетки."""

        if self.image is None:
            raise RuntimeError(
                "Сначала необходимо вызвать load()."
            )

        horizontal = self.find_horizontal_segments()
        vertical = self.find_vertical_segments()

        debug = self.image.copy()

        for x1, y1, x2, y2 in horizontal:
            cv2.line(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

        for x1, y1, x2, y2 in vertical:
            cv2.line(
                debug,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            debug
        )

        return horizontal, vertical

    def save_intersections_debug(self, output_path):
        """Рисует найденные пересечения сетки."""

        if self.image is None:
            raise RuntimeError(
                "Сначала необходимо вызвать load()."
            )

        intersections = self.find_intersections()
        debug = self.image.copy()

        for x, y in intersections:
            cv2.circle(
                debug,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            debug
        )

        return intersections

    def save_cells_debug(self, output_path):
        """Рисует найденные ячейки и их номера."""

        if self.image is None:
            raise RuntimeError(
                "Сначала необходимо вызвать load()."
            )

        cells = self.find_cells()
        debug = self.image.copy()

        for index, (x1, y1, x2, y2) in enumerate(
            cells,
            start=1
        ):
            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                debug,
                str(index),
                (x1 + 8, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            debug
        )

        return cells