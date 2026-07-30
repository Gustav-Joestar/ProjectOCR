from pathlib import Path
import time

import cv2

from src.segmentation.stamp_segmenter import StampSegmenter


INPUT_DIR = Path("output/stamp_batch")
OUTPUT_DIR = Path("output/stamp_segmentation_batch")


def main():
    stamp_paths = sorted(INPUT_DIR.glob("*_stamp.png"))

    if not stamp_paths:
        print(f"❌ Штампы не найдены: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failed = []
    start_time = time.time()

    print("\n🧩 Массовый тест сегментации штампов")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🖼️ Найдено штампов: {len(stamp_paths)}\n")

    for index, image_path in enumerate(stamp_paths, start=1):
        try:
            segmenter = StampSegmenter(image_path)

            segmenter.load()
            segmenter.preprocess()
            segmenter.detect_grid()

            horizontal = segmenter.find_horizontal_segments()
            vertical = segmenter.find_vertical_segments()
            intersections = segmenter.find_intersections()
            cells = segmenter.find_cells()

            height, width = segmenter.image.shape[:2]

            debug = segmenter.image.copy()

            for cell_index, (x1, y1, x2, y2) in enumerate(
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
                    str(cell_index),
                    (x1 + 8, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            output_path = OUTPUT_DIR / (
                f"{image_path.stem}_cells.png"
            )

            cv2.imwrite(
                str(output_path),
                debug
            )

            results.append({
                "name": image_path.name,
                "width": width,
                "height": height,
                "horizontal": len(horizontal),
                "vertical": len(vertical),
                "intersections": len(intersections),
                "cells": len(cells),
            })

            print(
                f"[{index:02}/{len(stamp_paths)}] "
                f"✅ {image_path.name} | "
                f"{width}×{height} | "
                f"H={len(horizontal)} "
                f"V={len(vertical)} "
                f"X={len(intersections)} "
                f"C={len(cells)}"
            )

        except Exception as error:
            failed.append(
                (image_path.name, str(error))
            )

            print(
                f"[{index:02}/{len(stamp_paths)}] "
                f"💥 {image_path.name} | "
                f"{error}"
            )

    elapsed = time.time() - start_time

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 Сводка\n")

    for result in results:
        print(
            f"{result['name']:<40} "
            f"{result['width']:>4}×{result['height']:<4} | "
            f"H={result['horizontal']:<3} "
            f"V={result['vertical']:<3} "
            f"X={result['intersections']:<3} "
            f"C={result['cells']:<3}"
        )

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🖼️ Обработано: {len(results)}/{len(stamp_paths)}")
    print(f"💥 Ошибок: {len(failed)}")
    print(f"⏱️ Время: {elapsed:.1f} сек.")
    print(f"📁 Результаты: {OUTPUT_DIR}")

    if failed:
        print("\n⚠️ Ошибки:")

        for name, error in failed:
            print(f"   • {name}: {error}")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()