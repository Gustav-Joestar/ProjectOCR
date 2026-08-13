import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    """
    Приводит изображение к grayscale.
    """

    if image is None or image.size == 0:
        return np.empty((0, 0), dtype=np.uint8)

    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def crop_cell_borders(
    image: np.ndarray,
    margin_ratio: float = 0.08
) -> np.ndarray:
    """
    Убирает края ячейки, где чаще всего остаются
    линии рамки таблицы.
    """

    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]

    margin_x = max(2, int(w * margin_ratio))
    margin_y = max(2, int(h * margin_ratio))

    # Не позволяем обрезать слишком большую часть
    # маленькой ячейки.
    margin_x = min(margin_x, max(1, w // 4))
    margin_y = min(margin_y, max(1, h // 4))

    if margin_x * 2 >= w or margin_y * 2 >= h:
        return image.copy()

    return image[
        margin_y:h - margin_y,
        margin_x:w - margin_x
    ].copy()


def make_binary(image: np.ndarray) -> np.ndarray:
    """
    Создаёт бинарное изображение:
    белое = содержимое,
    чёрное = фон.
    """

    gray = _to_gray(image)

    if gray.size == 0:
        return gray

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary


def remove_long_lines(
    binary: np.ndarray,
    horizontal_ratio: float = 0.70,
    vertical_ratio: float = 0.70
) -> np.ndarray:
    """
    Удаляет очень длинные горизонтальные и вертикальные
    элементы, которые с высокой вероятностью являются
    линиями таблицы.

    Короткие штрихи символов сохраняются.
    """

    if binary is None or binary.size == 0:
        return binary

    h, w = binary.shape[:2]

    horizontal_length = max(
        10,
        int(w * horizontal_ratio)
    )

    vertical_length = max(
        10,
        int(h * vertical_ratio)
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    lines = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines
    )

    cleaned = cv2.bitwise_and(
        binary,
        cv2.bitwise_not(lines)
    )

    return cleaned


def remove_small_noise(
    binary: np.ndarray,
    min_area: int = 8
) -> np.ndarray:
    """
    Удаляет совсем маленькие компоненты:
    одиночные точки и мелкий шум.
    """

    if binary is None or binary.size == 0:
        return binary

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary)

    for label in range(1, count):
        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def get_component_stats(
    binary: np.ndarray
) -> tuple[int, int]:
    """
    Возвращает:
    - количество компонентов;
    - площадь крупнейшего компонента.
    """

    if binary is None or binary.size == 0:
        return 0, 0

    count, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    component_count = 0
    largest_area = 0

    for label in range(1, count):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area <= 0:
            continue

        component_count += 1
        largest_area = max(
            largest_area,
            area
        )

    return component_count, largest_area


def analyze_cell(
    image: np.ndarray,
    margin_ratio: float = 0.08,
    empty_threshold: float = 0.002,
    min_component_area: int = 8
) -> dict:
    """
    Определяет наличие реального содержимого в ячейке.

    Сначала:
    1. убирает края;
    2. бинаризует;
    3. удаляет длинные линии таблицы;
    4. удаляет мелкий шум;
    5. анализирует оставшиеся компоненты.
    """

    if image is None or image.size == 0:
        return {
            "is_empty": True,
            "dark_ratio": 0.0,
            "component_count": 0,
            "largest_component_area": 0,
        }

    inner = crop_cell_borders(
        image,
        margin_ratio=margin_ratio
    )

    binary = make_binary(inner)

    cleaned = remove_long_lines(binary)

    cleaned = remove_small_noise(
        cleaned,
        min_area=min_component_area
    )

    total_pixels = cleaned.size

    if total_pixels == 0:
        dark_ratio = 0.0
    else:
        dark_ratio = (
            cv2.countNonZero(cleaned)
            / total_pixels
        )

    component_count, largest_area = (
        get_component_stats(cleaned)
    )

    # Ячейка считается непустой, если после удаления
    # сетки осталось достаточно содержимого.
    has_content = (
        dark_ratio >= empty_threshold
        and component_count > 0
        and largest_area >= min_component_area
    )

    return {
        "is_empty": not has_content,
        "dark_ratio": dark_ratio,
        "component_count": component_count,
        "largest_component_area": largest_area,
    }


def is_empty_cell(
    image: np.ndarray,
    margin_ratio: float = 0.08,
    empty_threshold: float = 0.002,
    min_component_area: int = 8
) -> bool:
    """
    True  — ячейка пустая.
    False — есть содержимое.
    """

    return analyze_cell(
        image,
        margin_ratio=margin_ratio,
        empty_threshold=empty_threshold,
        min_component_area=min_component_area
    )["is_empty"]