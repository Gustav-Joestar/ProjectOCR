from pathlib import Path
from collections import Counter

import cv2
import numpy as np


class StampDetector:
    """
    Детектор основной надписи чертежа.
    """

    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path)

        self.image: np.ndarray | None = None
        self.binary: np.ndarray | None = None

        self.horizontal_mask: np.ndarray | None = None
        self.vertical_mask: np.ndarray | None = None
        self.lines_mask: np.ndarray | None = None

    def load(self):
        """Загружает изображение."""

        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise FileNotFoundError(
                f"Не удалось загрузить изображение: {self.image_path}"
            )

        return self.image

    def preprocess(self):
        """Подготавливает бинарное изображение."""

        if self.image is None:
            raise RuntimeError("Сначала необходимо вызвать load().")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        return self.binary

    def detect_lines(self):
        """Выделяет горизонтальные и вертикальные линии."""

        if self.binary is None:
            raise RuntimeError("Сначала необходимо вызвать preprocess().")

        height, width = self.binary.shape

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(20, width // 30), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(20, height // 30))
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
        self.lines_mask = cv2.bitwise_or(
            self.horizontal_mask,
            self.vertical_mask
        )

        return self.lines_mask

    def detect_intersections(self):
        """Находит пересечения горизонтальных и вертикальных линий."""

        if self.horizontal_mask is None or self.vertical_mask is None:
            raise RuntimeError("Сначала необходимо вызвать detect_lines().")

        return cv2.bitwise_and(
            self.horizontal_mask,
            self.vertical_mask
        )

    def find_intersection_rows(self):
        """
        Находит горизонтальные ряды, содержащие
        несколько H/V-пересечений.

        Возвращает:
            [(y, x_positions), ...]
        """

        intersections = self.detect_intersections()

        height, width = intersections.shape

        # Немного расширяем пересечения только по вертикали,
        # чтобы пиксели одного пересечения было проще объединить.
        kernel_height = max(3, height // 1000)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_height)
        )

        mask = cv2.dilate(
            intersections,
            kernel,
            iterations=1
        )

        rows = []

        # Минимальное количество пересечений,
        # чтобы строка была похожа на строку таблицы.
        min_intersections = 4

        for y in range(height):
            row = mask[y]

            # Находим отдельные белые участки по X.
            binary_row = (row > 0).astype(np.uint8)

            num_labels, _, stats, centers = (
                cv2.connectedComponentsWithStats(
                    binary_row.reshape(1, -1),
                    connectivity=8
                )
            )

            x_positions = []

            for i in range(1, num_labels):
                x = int(centers[i][0])
                x_positions.append(x)

            if len(x_positions) >= min_intersections:
                rows.append(
                    (y, x_positions)
                )

        return rows

    def group_intersection_rows(self):
        """Объединяет соседние Y-строки в физические ряды таблицы."""

        rows = self.find_intersection_rows()

        if not rows:
            return []

        groups = [[rows[0]]]

        for row in rows[1:]:
            if row[0] - groups[-1][-1][0] <= 3:
                groups[-1].append(row)
            else:
                groups.append([row])

        result = []

        for group in groups:
            best_row = max(group, key=lambda row: len(row[1]))
            y = int(sum(row[0] for row in group) / len(group))

            result.append((y, best_row[1]))

        return result

    def group_table_rows(self):
        """
        Объединяет ряды пересечений в группы,
        используя расстояния между самими рядами.
        """

        rows = self.group_intersection_rows()

        if not rows:
            return []

        if len(rows) == 1:
            return [rows]

        # Расстояния между соседними рядами.
        gaps = []

        for i in range(1, len(rows)):
            gap = rows[i][0] - rows[i - 1][0]

            if gap > 0:
                gaps.append(gap)

        if not gaps:
            return [rows]

        # Нас интересуют небольшие интервалы.
        # Огромные промежутки между элементами чертежа
        # не должны влиять на оценку шага таблицы.
        sorted_gaps = sorted(gaps)

        lower_half = sorted_gaps[
            :max(1, len(sorted_gaps) // 2)
        ]

        typical_gap = float(np.median(lower_half))

        # Даём достаточно большой запас:
        # строки основной надписи могут иметь
        # неодинаковую высоту.
        max_gap = max(
            10,
            int(typical_gap * 1.8)
        )

        groups = []
        current_group = [rows[0]]

        for row in rows[1:]:
            previous_y = current_group[-1][0]
            current_y = row[0]

            gap = current_y - previous_y

            if gap <= max_gap:
                current_group.append(row)
            else:
                groups.append(current_group)
                current_group = [row]

        groups.append(current_group)

        return groups

    def group_x_intersections(self, x_counter, tolerance=5):
        """Объединяет близкие X-координаты в одну вертикаль."""

        groups = []

        for x in sorted(x_counter):
            if groups and x - groups[-1][-1] <= tolerance:
                groups[-1].append(x)
            else:
                groups.append([x])

        return groups

    def find_stamp_group(self):
        """Выбирает наиболее высокую группу табличных рядов."""

        groups = [
            group
            for group in self.group_table_rows()
            if len(group) >= 2
        ]

        if not groups:
            return None

        return max(
            groups,
            key=lambda group: group[-1][0] - group[0][0]
        )

    def find_stable_x_lines(self, stamp_group, tolerance=5):
        """Находит устойчивые вертикальные линии штампа."""

        if not stamp_group:
            return []

        x_counter = Counter(
            x
            for _, x_positions in stamp_group
            for x in x_positions
        )

        x_groups = self.group_x_intersections(
            x_counter,
            tolerance
        )

        min_hits = max(2, len(stamp_group) // 2)
        stable_x = []

        for group in x_groups:
            count = sum(x_counter[x] for x in group)

            if count >= min_hits:
                x_center = round(
                    sum(x * x_counter[x] for x in group) / count
                )
                stable_x.append((x_center, count))

        return stable_x

    def find_stamp_bounds(self):
        """Возвращает границы штампа: (x_left, y_top, x_right, y_bottom)."""

        stamp_group = self.find_stamp_group()

        if not stamp_group:
            return None

        x_bounds = self.find_stamp_x_bounds(stamp_group)

        if x_bounds is None:
            return None

        x_left, x_right = x_bounds
        y_top = stamp_group[0][0]
        y_bottom = stamp_group[-1][0]

        return x_left, y_top, x_right, y_bottom

    def find_stamp_x_bounds(self, stamp_group):
        """Определяет левую и правую границы штампа."""

        stable_x = self.find_stable_x_lines(stamp_group)

        if not stable_x:
            return None

        return stable_x[0][0], stable_x[-1][0]

    def save_intersection_rows_debug(
        self,
        output_path: str | Path
    ):
        """
        Рисует найденные ряды пересечений.
        """

        if self.image is None:
            raise RuntimeError(
                "Изображение не загружено."
            )

        rows = self.group_intersection_rows()

        debug = self.image.copy()

        height, width = debug.shape[:2]

        thickness = max(
            2,
            width // 700
        )

        for index, (y, x_positions) in enumerate(
            rows,
            start=1
        ):
            # Рисуем диапазон только между
            # крайними пересечениями ряда.
            x_left = min(x_positions)
            x_right = max(x_positions)

            cv2.line(
                debug,
                (x_left, y),
                (x_right, y),
                (0, 0, 255),
                thickness
            )

            cv2.putText(
                debug,
                f"{index} [{len(x_positions)}]",
                (x_left, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
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

        return rows

    def save_table_groups_debug(
        self,
        output_path: str | Path
    ):
        """
        Рисует группы рядов пересечений.
        """

        if self.image is None:
            raise RuntimeError(
                "Изображение не загружено."
            )

        groups = self.group_table_rows()

        debug = self.image.copy()

        height, width = debug.shape[:2]

        thickness = max(
            2,
            width // 700
        )

        for index, group in enumerate(
            groups,
            start=1
        ):
            all_x = []

            for _, x_positions in group:
                all_x.extend(x_positions)

            x_left = min(all_x)
            x_right = max(all_x)

            y_top = group[0][0]
            y_bottom = group[-1][0]

            # Небольшой запас вокруг крайних рядов.
            padding_y = max(
                5,
                int(height * 0.005)
            )

            y_top = max(
                0,
                y_top - padding_y
            )

            y_bottom = min(
                height - 1,
                y_bottom + padding_y
            )

            cv2.rectangle(
                debug,
                (x_left, y_top),
                (x_right, y_bottom),
                (0, 0, 255),
                thickness
            )

            cv2.putText(
                debug,
                f"{index} [{len(group)} rows]",
                (x_left, max(30, y_top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
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

        return groups

    def crop_stamp(self):
        """Находит и вырезает штамп из исходного изображения."""

        if self.image is None:
            raise RuntimeError("Изображение не загружено.")

        bounds = self.find_stamp_bounds()

        if bounds is None:
            return None

        x_left, y_top, x_right, y_bottom = bounds

        return self.image[
            y_top:y_bottom + 1,
            x_left:x_right + 1
        ]

    def save_debug(self, output_dir: str | Path):
        """
        Сохраняет отладочные изображения.
        """

        if self.lines_mask is None:
            raise RuntimeError(
                "Сначала необходимо вызвать detect_lines()."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(output_dir / "01_binary.png"),
            self.binary
        )

        cv2.imwrite(
            str(output_dir / "02_horizontal.png"),
            self.horizontal_mask
        )

        cv2.imwrite(
            str(output_dir / "03_vertical.png"),
            self.vertical_mask
        )

        cv2.imwrite(
            str(output_dir / "04_lines.png"),
            self.lines_mask
        )

        # Пересечения H/V
        intersections = self.detect_intersections()

        cv2.imwrite(
            str(output_dir / "05_intersections.png"),
            intersections
        )