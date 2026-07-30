import csv
import shutil
from pathlib import Path

import cv2

from src.segmentation.cell_filter import analyze_cell


PROJECT_ROOT = Path(__file__).resolve().parent

# Папка, где лежат ВСЕ ячейки штампа:
# и пустые, и непустые.
INPUT_DIR = PROJECT_ROOT / "output" / "stamp_cells"

# Debug-результаты.
DEBUG_DIR = PROJECT_ROOT / "output" / "cell_filter_debug"
EMPTY_DIR = DEBUG_DIR / "empty"
NON_EMPTY_DIR = DEBUG_DIR / "non_empty"

CSV_PATH = DEBUG_DIR / "filter_results.csv"


def prepare_output_dirs():
    """
    Подготавливает debug-папки.

    Старые результаты удаляются, чтобы картинки
    от предыдущего запуска не смешивались с новыми.
    """

    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)

    EMPTY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    NON_EMPTY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def get_cell_files():
    """
    Возвращает список PNG-файлов с ячейками.
    """

    return sorted(
        INPUT_DIR.glob("cell_*.png")
    )


def main():

    print("=" * 80)
    print("CELL FILTER")
    print("=" * 80)
    print()

    if not INPUT_DIR.exists():
        print("ОШИБКА:")
        print("Не найдена папка с исходными ячейками:")
        print(INPUT_DIR)
        print()
        print(
            "INPUT_DIR должен указывать на папку, "
            "где находятся ВСЕ ячейки штампа."
        )
        return

    cell_files = get_cell_files()

    if not cell_files:
        print("ОШИБКА:")
        print("В папке не найдено cell_*.png:")
        print(INPUT_DIR)
        return

    print(f"Исходная папка: {INPUT_DIR}")
    print(f"Найдено ячеек: {len(cell_files)}")
    print()

    prepare_output_dirs()

    rows = []

    empty_count = 0
    non_empty_count = 0
    error_count = 0

    for path in cell_files:

        image = cv2.imread(str(path))

        if image is None:
            print(
                f"{path.name:<20} | ERROR: "
                f"не удалось открыть"
            )

            error_count += 1

            rows.append({
                "file": path.name,
                "dark_ratio": "",
                "classification": "ERROR",
            })

            continue

        result = analyze_cell(image)

        dark_ratio = result["dark_ratio"]
        is_empty = result["is_empty"]

        if is_empty:
            classification = "EMPTY"
            destination = EMPTY_DIR / path.name
            empty_count += 1
        else:
            classification = "NON_EMPTY"
            destination = NON_EMPTY_DIR / path.name
            non_empty_count += 1

        # Копируем исходную картинку.
        # Оригинал остаётся на месте.
        shutil.copy2(
            path,
            destination
        )

        rows.append({
            "file": path.name,
            "dark_ratio": f"{dark_ratio:.6f}",
            "classification": classification,
        })

        print(
            f"{path.name:<20} | "
            f"{dark_ratio:8.5f} | "
            f"{classification}"
        )

    # Сохраняем CSV.
    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "file",
                "dark_ratio",
                "classification",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("РЕЗУЛЬТАТ")
    print("=" * 80)

    print(f"Всего:       {len(cell_files)}")
    print(f"EMPTY:       {empty_count}")
    print(f"NON_EMPTY:   {non_empty_count}")
    print(f"ERROR:       {error_count}")

    print()
    print("Пустые:")
    print(EMPTY_DIR)

    print()
    print("Непустые:")
    print(NON_EMPTY_DIR)

    print()
    print("CSV:")
    print(CSV_PATH)

    print()
    print("ГОТОВО")


if __name__ == "__main__":
    main()