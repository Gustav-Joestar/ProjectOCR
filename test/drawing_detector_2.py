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

        self.DEBUG = True
        self.debug_dir = self.image_path.parent / "_debug" / self.image_path.stem

    def _debug_save(self, name, image):
        if not self.DEBUG or image is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        from pathlib import Path
        cv2.imwrite(str(self.debug_dir / f"{name}.png"), image)

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
        print("[DEBUG] preprocess")
        self._debug_save("01_gray", self.gray)
        self._debug_save("02_binary", self.binary)

    def extract_horizontal_lines(self):
        if self.binary is None:
            raise RuntimeError(
                "Сначала необходимо выполнить предобработку."
            )

        _, width = self.binary.shape

        kernel_length = max(
            40,
            width // 160
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_length, 1)
        )

        horizontal = cv2.erode(
            self.binary,
            kernel,
            iterations=1
        )

        self.horizontal = cv2.dilate(
            horizontal,
            kernel,
            iterations=1
        )

        print("[DEBUG] horizontal")
        self._debug_save("03_horizontal", self.horizontal)

    def extract_vertical_lines(self):
        if self.binary is None:
            raise RuntimeError(
                "Сначала необходимо выполнить предобработку."
            )

        height, _ = self.binary.shape

        kernel_length = max(
            40,
            height // 160
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_length)
        )

        vertical = cv2.erode(
            self.binary,
            kernel,
            iterations=1
        )

        self.vertical = cv2.dilate(
            vertical,
            kernel,
            iterations=1
        )

        print("[DEBUG] vertical")
        self._debug_save("04_vertical", self.vertical)

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
        print("[DEBUG] lines")
        self._debug_save("05_lines", self.lines)

    def connect_lines(self):
        if self.lines is None:
            raise RuntimeError(
                "Сначала необходимо объединить линии."
            )

        height, width = self.lines.shape

        kernel_size = 7

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )

        self.connected_lines = cv2.morphologyEx(
            self.lines,
            cv2.MORPH_CLOSE,
            kernel
        )
        print("[DEBUG] connected")
        self._debug_save("06_connected_lines", self.connected_lines)

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
            coverage >= 0.55
            for coverage in coverages
        )

        weak_sides = sum(
            coverage >= 0.30
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
        
        height, width = self.connected_lines.shape

        min_line = int(min(width, height) * 0.08)

        lines = cv2.HoughLinesP(
            self.connected_lines,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line,
            maxLineGap=25,
        )

        if lines is None:
            self.drawings = []
            return self.drawings

        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx >= dy * 5:
                horizontal.append(
                    (
                        min(x1, x2),
                        y1,
                        max(x1, x2),
                        y2,
                    )
                )

            elif dy >= dx * 5:
                vertical.append(
                    (
                        x1,
                        min(y1, y2),
                        x2,
                        max(y1, y2),
                    )
                )

        debug = self.image.copy()

        for x1, y1, x2, y2 in horizontal:
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)

        for x1, y1, x2, y2 in vertical:
            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)

        self._debug_save("07_hough_lines", debug)

        print(f"H: {len(horizontal)}")
        print(f"V: {len(vertical)}")

        return []

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

    def save_drawings(self, output_dir: str):
        if self.image is None:
            raise RuntimeError(
                "Исходное изображение отсутствует."
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        saved_paths = []

        for index, (x, y, w, h) in enumerate(
            self.drawings,
            start=1
        ):
            drawing = self.image[
                y:y + h,
                x:x + w
            ]

            output_path = (
                output_dir
                / f"{self.image_path.stem}_drawing_{index:02d}.png"
            )

            if not cv2.imwrite(str(output_path), drawing):
                raise RuntimeError(
                    f"Не удалось сохранить: {output_path}"
                )

            saved_paths.append(output_path)

        return saved_paths

    def save_drawings(
        self,
        output_dir: str | Path,
        page_number: int
    ) -> list[Path]:
        """
        Вырезает найденные чертежи из исходного изображения
        и сохраняет каждый чертёж отдельным PNG-файлом.
        """
        if self.image is None:
            raise RuntimeError(
                "Исходное изображение отсутствует."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        saved_drawings = []

        for drawing_number, (x, y, w, h) in enumerate(
            self.drawings,
            start=1
        ):
            crop = self.image[
                y:y + h,
                x:x + w
            ]

            if crop.size == 0:
                continue

            output_path = (
                output_dir
                / (
                    f"page_{page_number:03d}"
                    f"_drawing_{drawing_number:03d}.png"
                )
            )

            if not cv2.imwrite(str(output_path), crop):
                raise RuntimeError(
                    f"Не удалось сохранить чертёж: {output_path}"
                )

            saved_drawings.append(output_path)

        return saved_drawings