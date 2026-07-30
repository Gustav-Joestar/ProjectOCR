import csv
import shutil
from pathlib import Path

import cv2

from src.segmentation.stamp_segmenter import StampSegmenter
from src.segmentation.cell_filter import analyze_cell


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "stamp_batch"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "cell_filter_batch"
)

EMPTY_DIR = OUTPUT_DIR / "empty"
NON_EMPTY_DIR = OUTPUT_DIR / "non_empty"

CSV_PATH = OUTPUT_DIR / "filter_results.csv"


def prepare_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    EMPTY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    NON_EMPTY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def find_stamp_images():
    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if (
            path.is_file()
            and path.name.lower().endswith("_stamp.png")
        )
    )


def main():
    print("=" * 80)
    print("BATCH CELL FILTER")
    print("=" * 80)
    print()

    if not INPUT_DIR.exists():
        print("Не найдена папка:")
        print(INPUT_DIR)
        return

    stamps = find_stamp_images()

    if not stamps:
        print("В папке не найдено изображений:")
        print(INPUT_DIR)
        return

    print(f"Найдено штампов: {len(stamps)}")
    print()

    prepare_output()

    rows = []

    total_cells = 0
    total_empty = 0
    total_non_empty = 0
    total_errors = 0

    for stamp_number, stamp_path in enumerate(
        stamps,
        start=1
    ):
        print("-" * 80)
        print(
            f"[{stamp_number}/{len(stamps)}] "
            f"{stamp_path.name}"
        )

        try:
            segmenter = StampSegmenter(stamp_path)

            segmenter.load()
            segmenter.preprocess()
            segmenter.detect_grid()

            cells = segmenter.extract_cells()

        except Exception as error:
            print(f"ОШИБКА СЕГМЕНТАЦИИ: {error}")

            total_errors += 1

            rows.append({
                "stamp": stamp_path.name,
                "cell": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "dark_ratio": "",
                "classification": "SEGMENTATION_ERROR",
            })

            continue

        print(f"Ячеек: {len(cells)}")

        total_cells += len(cells)

        stamp_empty = 0
        stamp_non_empty = 0

        for cell in cells:
            index = cell["index"]
            image = cell["image"]

            x1, y1, x2, y2 = cell["bbox"]

            analysis = analyze_cell(image)

            dark_ratio = analysis["dark_ratio"]
            is_empty = analysis["is_empty"]

            # Имя штампа включаем в имя файла.
            # Поэтому cell_001 разных штампов
            # друг друга не перезапишут.
            output_name = (
                f"{stamp_path.stem}"
                f"__cell_{index:03d}.png"
            )

            if is_empty:
                destination = (
                    EMPTY_DIR
                    / output_name
                )

                classification = "EMPTY"

                stamp_empty += 1
                total_empty += 1

            else:
                destination = (
                    NON_EMPTY_DIR
                    / output_name
                )

                classification = "NON_EMPTY"

                stamp_non_empty += 1
                total_non_empty += 1

            cv2.imwrite(
                str(destination),
                image
            )

            rows.append({
                "stamp": stamp_path.name,
                "cell": index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "dark_ratio": f"{dark_ratio:.6f}",
                "classification": classification,
            })

        print(
            f"EMPTY: {stamp_empty} | "
            f"NON_EMPTY: {stamp_non_empty}"
        )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "stamp",
                "cell",
                "x1",
                "y1",
                "x2",
                "y2",
                "dark_ratio",
                "classification",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    print(f"Штампов:     {len(stamps)}")
    print(f"Ячеек:       {total_cells}")
    print(f"EMPTY:       {total_empty}")
    print(f"NON_EMPTY:   {total_non_empty}")
    print(f"Ошибок:      {total_errors}")

    print()
    print(f"Результаты: {OUTPUT_DIR}")
    print(f"CSV:        {CSV_PATH}")


if __name__ == "__main__":
    main()