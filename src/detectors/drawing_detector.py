from pathlib import Path

import cv2
import numpy as np


class DrawingDetector:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)

        self.image = None
        self.gray = None
        self.binary = None

        self.horizontal = None
        self.vertical = None
        self.lines = None
        self.connected_lines = None

        self.drawings = []

    def load(self):
        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.image_path}"
            )

        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise RuntimeError(
                f"Не удалось открыть изображение: {self.image_path}"
            )

    def preprocess(self):
        if self.image is None:
            raise RuntimeError(
                "Сначала необходимо загрузить изображение."
            )

        self.gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        self.binary = cv2.adaptiveThreshold(
            self.gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

    def extract_horizontal_lines(self):
        if self.binary is None:
            raise RuntimeError(
                "Сначала необходимо выполнить предобработку."
            )

        _, width = self.binary.shape

        kernel_length = max(
            50,
            width // 40
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_length, 1)
        )

        self.horizontal = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            kernel
        )

    def extract_vertical_lines(self):
        if self.binary is None:
            raise RuntimeError(
                "Сначала необходимо выполнить предобработку."
            )

        height, _ = self.binary.shape

        kernel_length = max(
            50,
            height // 40
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_length)
        )

        self.vertical = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            kernel
        )

    def combine_lines(self):
        if self.horizontal is None:
            raise RuntimeError(
                "Сначала необходимо выделить горизонтальные линии."
            )

        if self.vertical is None:
            raise RuntimeError(
                "Сначала необходимо выделить вертикальные линии."
            )

        self.lines = cv2.bitwise_or(
            self.horizontal,
            self.vertical
        )

    def connect_lines(self):
        if self.lines is None:
            raise RuntimeError(
                "Сначала необходимо объединить линии."
            )

        height, width = self.lines.shape

        kernel_size = max(
            3,
            round(min(width, height) * 0.001)
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )

        self.connected_lines = cv2.morphologyEx(
            self.lines,
            cv2.MORPH_CLOSE,
            kernel
        )

    def _border_coverage(self, rect):
        """
        Оценивает покрытие четырёх сторон кандидата линиями.

        Значения, близкие к 1, означают, что соответствующая
        сторона почти полностью представлена в маске линий.
        """
        x, y, w, h = rect

        mask = self.connected_lines
        image_height, image_width = mask.shape

        margin = max(
            3,
            round(min(image_width, image_height) * 0.001)
        )

        x1 = max(0, x - margin)
        x2 = min(image_width, x + w + margin)

        y1 = max(0, y - margin)
        y2 = min(image_height, y + h + margin)

        top_y1 = max(0, y - margin)
        top_y2 = min(image_height, y + margin + 1)

        bottom = y + h - 1
        bottom_y1 = max(0, bottom - margin)
        bottom_y2 = min(image_height, bottom + margin + 1)

        left_x1 = max(0, x - margin)
        left_x2 = min(image_width, x + margin + 1)

        right = x + w - 1
        right_x1 = max(0, right - margin)
        right_x2 = min(image_width, right + margin + 1)

        top_region = mask[
            top_y1:top_y2,
            x1:x2
        ]

        bottom_region = mask[
            bottom_y1:bottom_y2,
            x1:x2
        ]

        left_region = mask[
            y1:y2,
            left_x1:left_x2
        ]

        right_region = mask[
            y1:y2,
            right_x1:right_x2
        ]

        def horizontal_coverage(region):
            if region.size == 0:
                return 0.0

            occupied = (region > 0).any(axis=0)
            return occupied.mean()

        def vertical_coverage(region):
            if region.size == 0:
                return 0.0

            occupied = (region > 0).any(axis=1)
            return occupied.mean()

        return (
            horizontal_coverage(top_region),
            horizontal_coverage(bottom_region),
            vertical_coverage(left_region),
            vertical_coverage(right_region),
        )

    def _looks_like_frame(self, rect):
        top, bottom, left, right = self._border_coverage(rect)

        coverages = [
            top,
            bottom,
            left,
            right
        ]

        strong_sides = sum(
            coverage >= 0.70
            for coverage in coverages
        )

        weak_sides = sum(
            coverage >= 0.40
            for coverage in coverages
        )

        return (
            strong_sides >= 3
            or (
                strong_sides >= 2
                and weak_sides == 4
            )
        )

    @staticmethod
    def _count_line_groups(projection, threshold):
        """
        Считает группы соседних строк или столбцов,
        образующих отдельные линии.
        """
        active = projection >= threshold

        groups = 0
        inside_group = False

        for value in active:
            if value and not inside_group:
                groups += 1
                inside_group = True
            elif not value:
                inside_group = False

        return groups

    def _stamp_score(self, rect):
        """
        Оценивает наличие табличной структуры штампа
        в нижней части кандидата.

        0.0 — признаки практически отсутствуют.
        1.0 — выраженная табличная структура.
        """
        if self.horizontal is None or self.vertical is None:
            return 0.0

        x, y, w, h = rect

        image_height, image_width = self.horizontal.shape

        stamp_top = y + int(h * 0.68)
        stamp_bottom = y + h

        margin_x = max(3, int(w * 0.015))
        margin_y = max(3, int(h * 0.01))

        x1 = max(0, x + margin_x)
        x2 = min(image_width, x + w - margin_x)

        y1 = max(0, stamp_top)
        y2 = min(image_height, stamp_bottom - margin_y)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        horizontal_roi = (
            self.horizontal[y1:y2, x1:x2] > 0
        )

        vertical_roi = (
            self.vertical[y1:y2, x1:x2] > 0
        )

        roi_height, roi_width = horizontal_roi.shape

        if roi_width < 10 or roi_height < 10:
            return 0.0

        horizontal_projection = horizontal_roi.sum(axis=1)
        vertical_projection = vertical_roi.sum(axis=0)

        horizontal_groups = self._count_line_groups(
            horizontal_projection,
            max(10, int(roi_width * 0.08))
        )

        vertical_groups = self._count_line_groups(
            vertical_projection,
            max(10, int(roi_height * 0.08))
        )

        intersections = np.logical_and(
            horizontal_roi,
            vertical_roi
        )

        intersection_density = intersections.mean()

        score = 0.0

        score += min(
            horizontal_groups / 6.0,
            1.0
        ) * 0.40

        score += min(
            vertical_groups / 6.0,
            1.0
        ) * 0.40

        score += min(
            intersection_density / 0.003,
            1.0
        ) * 0.20

        if horizontal_groups < 2 or vertical_groups < 2:
            score *= 0.25

        return min(score, 1.0)

    def _frame_score(self, rect):
        """
        Оценивает качество внешней рамки в диапазоне 0..1.
        """
        coverages = self._border_coverage(rect)
        return sum(coverages) / len(coverages)

    @staticmethod
    def _intersection(rect_a, rect_b):
        ax, ay, aw, ah = rect_a
        bx, by, bw, bh = rect_b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        return width * height

    @classmethod
    def _overlap_smaller(cls, rect_a, rect_b):
        """
        Возвращает долю меньшего прямоугольника,
        перекрываемую вторым прямоугольником.
        """
        intersection = cls._intersection(
            rect_a,
            rect_b
        )

        area_a = rect_a[2] * rect_a[3]
        area_b = rect_b[2] * rect_b[3]

        smaller_area = min(area_a, area_b)

        if smaller_area == 0:
            return 0.0

        return intersection / smaller_area

    @staticmethod
    def _contains(outer, inner):
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner

        return (
            ix >= ox
            and iy >= oy
            and ix + iw <= ox + ow
            and iy + ih <= oy + oh
        )

    def _candidate_score(self, rect):
        """
        Рассчитывает итоговую оценку кандидата.

        Основной вес имеет наличие штампа,
        дополнительный — качество внешней рамки.
        """
        frame = self._frame_score(rect)
        stamp = self._stamp_score(rect)

        score = (
            stamp * 0.75
            + frame * 0.25
        )

        return score, frame, stamp

    def find_drawings(self):
        if self.connected_lines is None:
            raise RuntimeError(
                "Сначала необходимо соединить разрывы линий."
            )

        contours, hierarchy = cv2.findContours(
            self.connected_lines,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            self.drawings = []
            return self.drawings

        image_height, image_width = self.connected_lines.shape
        image_area = image_width * image_height

        candidates = []

        # Собираем геометрически подходящие рамки.
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            width_ratio = w / image_width
            height_ratio = h / image_height
            area_ratio = (w * h) / image_area

            # Исключаем рамку всей страницы.
            if (
                width_ratio >= 0.90
                and height_ratio >= 0.90
            ):
                continue

            # Исключаем слишком маленькие области.
            if width_ratio < 0.10:
                continue

            if height_ratio < 0.10:
                continue

            if area_ratio < 0.018:
                continue

            # Исключаем слишком крупные контейнеры.
            if width_ratio > 0.80:
                continue

            if height_ratio > 0.80:
                continue

            rect = (x, y, w, h)

            if not self._looks_like_frame(rect):
                continue

            score, frame_score, stamp_score = (
                self._candidate_score(rect)
            )

            candidates.append(
                {
                    "rect": rect,
                    "area": w * h,
                    "score": score,
                    "frame": frame_score,
                    "stamp": stamp_score,
                }
            )

        # Исключаем кандидатов практически без признаков штампа.
        candidates = [
            candidate
            for candidate in candidates
            if candidate["stamp"] >= 0.20
        ]

        # Более качественные кандидаты рассматриваем первыми.
        candidates.sort(
            key=lambda candidate: candidate["score"],
            reverse=True
        )

        selected = []

        # Устраняем конкурирующие варианты одной рамки.
        for candidate in candidates:
            rect = candidate["rect"]

            conflict_index = None

            for index, existing in enumerate(selected):
                overlap = self._overlap_smaller(
                    rect,
                    existing["rect"]
                )

                if overlap >= 0.88:
                    conflict_index = index
                    break

            if conflict_index is None:
                selected.append(candidate)
                continue

            existing = selected[conflict_index]

            candidate_stamp = candidate["stamp"]
            existing_stamp = existing["stamp"]

            if candidate_stamp > existing_stamp + 0.08:
                selected[conflict_index] = candidate
                continue

            if existing_stamp > candidate_stamp + 0.08:
                continue

            candidate_frame = candidate["frame"]
            existing_frame = existing["frame"]

            if candidate_frame > existing_frame + 0.08:
                selected[conflict_index] = candidate
                continue

            if existing_frame > candidate_frame + 0.08:
                continue

            # При близких оценках выбираем более полную рамку.
            if candidate["area"] > existing["area"]:
                selected[conflict_index] = candidate

        filtered = []

        # Устраняем крупные контейнеры, содержащие несколько
        # самостоятельных кандидатов с выраженными штампами.
        for candidate in selected:
            rect = candidate["rect"]
            area = candidate["area"]

            inner_drawings = []

            for other in selected:
                if other is candidate:
                    continue

                if not self._contains(
                    rect,
                    other["rect"]
                ):
                    continue

                if other["area"] >= area * 0.80:
                    continue

                if other["stamp"] < 0.75:
                    continue

                inner_drawings.append(other)

            if len(inner_drawings) >= 2:
                continue

            filtered.append(candidate)

        filtered.sort(
            key=lambda candidate: (
                candidate["rect"][1],
                candidate["rect"][0]
            )
        )

        self.drawings = [
            candidate["rect"]
            for candidate in filtered
        ]

        return self.drawings

    def save_detection_preview(self, output_path: str):
        if self.image is None:
            raise RuntimeError(
                "Исходное изображение отсутствует."
            )

        preview = self.image.copy()

        for index, (x, y, w, h) in enumerate(
            self.drawings,
            start=1
        ):
            cv2.rectangle(
                preview,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                10
            )

            cv2.putText(
                preview,
                str(index),
                (x + 30, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 0, 255),
                8
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(
                f"Не удалось сохранить: {output_path}"
            )