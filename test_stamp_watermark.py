from pathlib import Path

import cv2

from src.detectors.stamp_detector import StampDetector


DRAWING_PATH = Path(
    "output/drawing/album_dukmasova/page_043_drawing_001.png"
)

OUTPUT_DIR = Path(
    "output/test/stamp_watermark"
)

BOUNDS_PATH = OUTPUT_DIR / "page_043_stamp_bounds.png"
STAMP_PATH = OUTPUT_DIR / "page_043_stamp.png"


def main():
    print("=" * 70)
    print("STAMP FALLBACK TEST")
    print("=" * 70)

    if not DRAWING_PATH.exists():
        raise FileNotFoundError(
            f"Чертёж не найден: {DRAWING_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # StampDetector
    # ---------------------------------------------------------

    detector = StampDetector(DRAWING_PATH)

    detector.load()
    detector.preprocess()
    detector.detect_lines()

    # Здесь сначала работает старый алгоритм,
    # а при его неудаче — наш fallback.
    bounds = detector.find_stamp_bounds()

    print(f"Файл:  {DRAWING_PATH}")
    print(f"Штамп: {bounds}")
    print()

    if bounds is None:
        print("✗ Штамп не найден")
        return

    # ---------------------------------------------------------
    # Отрисовка найденных границ
    # ---------------------------------------------------------

    image = cv2.imread(
        str(DRAWING_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Не удалось открыть изображение: {DRAWING_PATH}"
        )

    x1, y1, x2, y2 = bounds

    preview = image.copy()

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        8,
    )

    cv2.imwrite(
        str(BOUNDS_PATH),
        preview,
    )

    # ---------------------------------------------------------
    # Сохраняем найденный штамп отдельно
    # ---------------------------------------------------------

    stamp = image[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    if stamp.size > 0:
        cv2.imwrite(
            str(STAMP_PATH),
            stamp,
        )

    # ---------------------------------------------------------
    # Результат
    # ---------------------------------------------------------

    print("✓ Штамп найден")
    print(
        f"  bounds: ({x1}, {y1}, {x2}, {y2})"
    )
    print(
        f"  size:   {x2 - x1} x {y2 - y1}"
    )

    print()
    print(f"Границы: {BOUNDS_PATH}")
    print(f"Штамп:   {STAMP_PATH}")


if __name__ == "__main__":
    main()