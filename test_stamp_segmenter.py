from pathlib import Path

from src.segmentation.stamp_segmenter import StampSegmenter


IMAGE_PATH = Path(
    "output/stamp_batch/page_001_drawing_001_stamp.png"
)

DEBUG_DIR = Path(
    "output/stamp_segmentation_debug"
)


def main():
    print("\n🧩 Тест сегментации штампа")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    segmenter = StampSegmenter(IMAGE_PATH)

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()
    segmenter.save_debug(DEBUG_DIR)
    horizontal, vertical = segmenter.save_segments_debug(
        DEBUG_DIR / "05_detected_lines.png"
    )

    intersections = segmenter.save_intersections_debug(
    DEBUG_DIR / "06_intersections.png"
    )

    print(f"➖ Горизонтальных отрезков: {len(horizontal)}")
    print(f"┃ Вертикальных отрезков: {len(vertical)}")

    height, width = segmenter.image.shape[:2]

    print("\n✚ Пересечения:")

    for i, (x, y) in enumerate(intersections, start=1):
        print(f"   [{i:2}] x={x:4} | y={y:4}")

    cells = segmenter.save_cells_debug(
        DEBUG_DIR / "07_cells.png"
    )

    print(f"▦ Ячеек: {len(cells)}")

    print("\n▦ Найденные ячейки:")

    for i, (x1, y1, x2, y2) in enumerate(cells, start=1):
        print(
            f"   [{i:2}] "
            f"({x1}, {y1}) → ({x2}, {y2}) | "
            f"{x2 - x1}×{y2 - y1}"
        )

    print(f"✚ Пересечений: {len(intersections)}")
    print(f"🖼️ Штамп: {IMAGE_PATH.name}")
    print(f"📐 Размер: {width} × {height}")
    print(f"💾 Отладка: {DEBUG_DIR}")
    print("✅ Готово")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()