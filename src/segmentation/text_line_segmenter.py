import cv2
import numpy as np


def _merge_intervals(intervals, gap):
    """
    Объединяет близкие вертикальные интервалы.
    """
    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [list(intervals[0])]

    for y1, y2 in intervals[1:]:
        prev_y1, prev_y2 = merged[-1]

        if y1 - prev_y2 <= gap:
            merged[-1][1] = max(prev_y2, y2)
        else:
            merged.append([y1, y2])

    return [tuple(x) for x in merged]


def segment_text_lines(image):
    """
    Разбивает изображение ячейки штампа на отдельные строки текста.

    Возвращает список изображений строк сверху вниз.
    """

    if image is None or image.size == 0:
        return []

    # Paddle Recognition ожидает 3 канала,
    # но для анализа нам нужен grayscale.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Небольшое размытие.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Чёрный текст -> белые объекты.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ---------------------------------------------------------
    # Удаляем рамку ячейки.
    # ---------------------------------------------------------

    margin_x = max(2, int(w * 0.01))
    margin_y = max(2, int(h * 0.02))

    binary[:margin_y, :] = 0
    binary[h - margin_y:, :] = 0
    binary[:, :margin_x] = 0
    binary[:, w - margin_x:] = 0

    # ---------------------------------------------------------
    # Удаляем длинные горизонтальные и вертикальные линии.
    # ---------------------------------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, w // 5), 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, h // 3))
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    clean = cv2.subtract(binary, horizontal)
    clean = cv2.subtract(clean, vertical)

    # ---------------------------------------------------------
    # Connected Components.
    # ---------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            clean,
            connectivity=8
        )
    )

    components = []

    min_area = max(5, int(h * w * 0.00001))

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Шум.
        if area < min_area:
            continue

        # Остатки огромных линий.
        if cw > w * 0.8 and ch < h * 0.08:
            continue

        if ch > h * 0.9 and cw < w * 0.05:
            continue

        # Очень мелкие компоненты.
        if ch < max(2, h * 0.015):
            continue

        components.append((x, y, cw, ch))

    if not components:
        return [image]

    # ---------------------------------------------------------
    # Каждый символ задаёт вертикальный диапазон.
    #
    # Буквы одной строки пересекаются по Y или находятся
    # очень близко друг к другу.
    # ---------------------------------------------------------

    intervals = []

    for x, y, cw, ch in components:
        intervals.append((y, y + ch))

    gap = max(3, int(h * 0.04))

    lines = _merge_intervals(intervals, gap)

    # ---------------------------------------------------------
    # Фильтруем странные результаты.
    # ---------------------------------------------------------

    filtered_lines = []

    for y1, y2 in lines:

        line_height = y2 - y1

        if line_height < h * 0.08:
            continue

        filtered_lines.append((y1, y2))

    # Если ничего нормального не нашли.
    if not filtered_lines:
        return [image]

    # ---------------------------------------------------------
    # Вырезаем строки из ОРИГИНАЛЬНОГО изображения.
    # ---------------------------------------------------------

    result = []

    padding_y = max(3, int(h * 0.04))
    padding_x = max(3, int(w * 0.01))

    for y1, y2 in filtered_lines:

        yy1 = max(0, y1 - padding_y)
        yy2 = min(h, y2 + padding_y)

        # Находим компоненты, относящиеся к этой строке,
        # чтобы не оставлять 1500 px пустого пространства.
        line_components = []

        for x, y, cw, ch in components:

            cy = y + ch / 2

            if y1 <= cy <= y2:
                line_components.append((x, y, cw, ch))

        if line_components:

            x1 = min(x for x, y, cw, ch in line_components)
            x2 = max(x + cw for x, y, cw, ch in line_components)

            xx1 = max(0, x1 - padding_x)
            xx2 = min(w, x2 + padding_x)

        else:
            xx1 = 0
            xx2 = w

        crop = image[yy1:yy2, xx1:xx2]

        if crop.size > 0:
            result.append(crop)

    return result