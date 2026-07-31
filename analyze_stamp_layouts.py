import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.ocr.stamp_field_mapper import StampFieldMapper
from src.segmentation.stamp_segmenter import StampSegmenter


PROJECT_ROOT = Path(__file__).resolve().parent

STAMP_DIR = PROJECT_ROOT / "output" / "stamp_batch"

OCR_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "ocr_results_corrected.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "stamp_layout_analysis.csv"
)


# Шаблонные подписи основной надписи.
# Сейчас они нужны как геометрические якоря.
ANCHORS = {
    "Изм.",
    "Лист",
    "№ докум.",
    "Подп.",
    "Дата",
    "Разраб.",
    "Пров.",
    "Т. контр.",
    "Н. контр.",
    "Утв.",
    "Лит.",
    "Масса",
    "Масштаб",
}


CELL_PATTERN = re.compile(
    r"^(?P<stamp>.+_stamp)__cell_(?P<index>\d+)\.png$"
)


def load_ocr_results():
    """
    Загружает corrected OCR один раз и группирует результаты:

        stamp_stem -> cell_index -> OCR data
    """

    grouped = defaultdict(dict)

    with OCR_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row.get("file", "")

            match = CELL_PATTERN.match(filename)

            if not match:
                continue

            stamp_stem = match.group("stamp")
            index = int(match.group("index"))

            text = (
                row.get("corrected_text")
                or row.get("raw_text")
                or ""
            ).strip()

            raw_text = (
                row.get("raw_text")
                or ""
            ).strip()

            try:
                confidence = float(
                    row.get("confidence") or 0.0
                )
            except (TypeError, ValueError):
                confidence = 0.0

            grouped[stamp_stem][index] = {
                "text": text,
                "raw_text": raw_text,
                "confidence": confidence,
            }

    return grouped


def process_stamp(stamp_path, ocr_by_stamp):
    """
    Сегментирует один штамп и возвращает найденные
    шаблонные anchor-поля с нормализованной геометрией.
    """

    segmenter = StampSegmenter(stamp_path)

    image = segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    bounds = segmenter.find_cells()

    stamp_height, stamp_width = image.shape[:2]

    mapper = StampFieldMapper(
        stamp_width=stamp_width,
        stamp_height=stamp_height,
    )

    stamp_ocr = ocr_by_stamp.get(
        stamp_path.stem,
        {},
    )

    rows = []

    for index, cell_bounds in enumerate(
        bounds,
        start=1,
    ):
        ocr = stamp_ocr.get(index)

        if ocr is None:
            continue

        text = ocr["text"]

        if text not in ANCHORS:
            continue

        cell = mapper.prepare_cell({
            "index": index,
            "bounds": cell_bounds,
            "text": text,
            "confidence": ocr["confidence"],
        })

        rows.append({
            "stamp": stamp_path.name,
            "cell_index": cell.index,

            "text": cell.text,
            "raw_text": ocr["raw_text"],
            "confidence": cell.confidence,

            "stamp_width": stamp_width,
            "stamp_height": stamp_height,

            "nx1": cell.nx1,
            "ny1": cell.ny1,
            "nx2": cell.nx2,
            "ny2": cell.ny2,

            "center_x": cell.normalized_center_x,
            "center_y": cell.normalized_center_y,

            "width": cell.normalized_width,
            "height": cell.normalized_height,
        })

    return rows, len(bounds)


def save_csv(rows):
    """
    Сохраняет каждое обнаружение anchor-поля.
    """

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "stamp",
        "cell_index",
        "text",
        "raw_text",
        "confidence",

        "stamp_width",
        "stamp_height",

        "nx1",
        "ny1",
        "nx2",
        "ny2",

        "center_x",
        "center_y",

        "width",
        "height",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_metric(name, values):
    """
    Печатает min / avg / max для набора значений.
    """

    if not values:
        return

    print(
        f"{name:<10} "
        f"min={min(values):.4f}  "
        f"avg={mean(values):.4f}  "
        f"max={max(values):.4f}  "
        f"range={max(values) - min(values):.4f}"
    )


def print_statistics(rows, total_stamps):
    """
    Статистика геометрии каждого anchor-поля.
    """

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["text"]].append(row)

    print()
    print("=" * 100)
    print("СТАТИСТИКА ANCHOR-ПОЛЕЙ")
    print("=" * 100)

    # Сортируем прежде всего по количеству обнаружений.
    anchors_sorted = sorted(
        ANCHORS,
        key=lambda anchor: len(grouped.get(anchor, [])),
        reverse=True,
    )

    for anchor in anchors_sorted:
        items = grouped.get(anchor, [])

        print()
        print("-" * 100)
        print(f"ANCHOR: {anchor!r}")
        print("-" * 100)

        if not items:
            print(
                f"Найдено: 0 / {total_stamps}"
            )
            continue

        stamps = {
            item["stamp"]
            for item in items
        }

        print(
            f"Найдено: {len(items)} раз "
            f"в {len(stamps)} / {total_stamps} штампах"
        )

        print_metric(
            "center_x",
            [item["center_x"] for item in items],
        )

        print_metric(
            "center_y",
            [item["center_y"] for item in items],
        )

        print_metric(
            "width",
            [item["width"] for item in items],
        )

        print_metric(
            "height",
            [item["height"] for item in items],
        )


def print_layout_signatures(rows):
    """
    Показывает, какие наборы anchor-полей встречаются
    в штампах.

    Это поможет понять:
    один у нас layout или несколько вариантов.
    """

    by_stamp = defaultdict(set)

    for row in rows:
        by_stamp[row["stamp"]].add(
            row["text"]
        )

    signatures = defaultdict(list)

    for stamp, anchors in by_stamp.items():
        signature = tuple(
            sorted(anchors)
        )

        signatures[signature].append(stamp)

    sorted_signatures = sorted(
        signatures.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    print()
    print("=" * 100)
    print("НАБОРЫ ОБНАРУЖЕННЫХ ANCHOR-ПОЛЕЙ")
    print("=" * 100)

    for number, (signature, stamps) in enumerate(
        sorted_signatures,
        start=1,
    ):
        print()
        print(
            f"LAYOUT #{number}: "
            f"{len(stamps)} штампов"
        )

        print(
            "Anchors: "
            + ", ".join(signature)
        )

        examples = stamps[:5]

        print(
            "Примеры: "
            + ", ".join(examples)
        )


def main():
    print("=" * 100)
    print("STAMP LAYOUT ANALYSIS")
    print("=" * 100)

    if not STAMP_DIR.exists():
        print()
        print("Папка со штампами не найдена:")
        print(STAMP_DIR)
        return

    if not OCR_CSV.exists():
        print()
        print("Corrected OCR CSV не найден:")
        print(OCR_CSV)
        return

    stamp_paths = sorted(
        STAMP_DIR.glob("*_stamp.png")
    )

    if not stamp_paths:
        print()
        print("Штампы не найдены:")
        print(STAMP_DIR)
        return

    print()
    print(f"Штампов найдено: {len(stamp_paths)}")
    print(f"OCR: {OCR_CSV}")

    print()
    print("Загрузка OCR...")

    ocr_by_stamp = load_ocr_results()

    print(
        f"Штампов с OCR: {len(ocr_by_stamp)}"
    )

    all_rows = []

    errors = []
    cell_counts = []

    print()
    print("Анализ геометрии...")

    for number, stamp_path in enumerate(
        stamp_paths,
        start=1,
    ):
        try:
            rows, cell_count = process_stamp(
                stamp_path,
                ocr_by_stamp,
            )

            all_rows.extend(rows)
            cell_counts.append(cell_count)

            print(
                f"[{number:03d}/{len(stamp_paths):03d}] "
                f"{stamp_path.name} | "
                f"cells={cell_count:3d} | "
                f"anchors={len(rows):2d}"
            )

        except Exception as error:
            errors.append(
                (stamp_path.name, str(error))
            )

            print(
                f"[{number:03d}/{len(stamp_paths):03d}] "
                f"ERROR | "
                f"{stamp_path.name} | "
                f"{error}"
            )

    save_csv(all_rows)

    print_statistics(
        all_rows,
        total_stamps=len(stamp_paths),
    )

    print_layout_signatures(all_rows)

    print()
    print("=" * 100)
    print("ИТОГ")
    print("=" * 100)

    print(
        f"Всего штампов:       {len(stamp_paths)}"
    )

    print(
        f"Успешно обработано:  "
        f"{len(stamp_paths) - len(errors)}"
    )

    print(
        f"Ошибок:              {len(errors)}"
    )

    print(
        f"Anchor-находок:      {len(all_rows)}"
    )

    if cell_counts:
        print(
            f"Ячеек на штамп:      "
            f"min={min(cell_counts)}, "
            f"avg={mean(cell_counts):.1f}, "
            f"max={max(cell_counts)}"
        )

    print()
    print("CSV:")
    print(OUTPUT_CSV)

    if errors:
        print()
        print("=" * 100)
        print("ОШИБКИ")
        print("=" * 100)

        for stamp, error in errors:
            print()
            print(stamp)
            print(error)


if __name__ == "__main__":
    main()