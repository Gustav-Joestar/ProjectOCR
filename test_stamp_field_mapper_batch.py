import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

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
    / "stamp_field_mapper_results.csv"
)


FIELDS = (
    "designation",
    "name",
    "material",
    "scale",
    "sheet_count",
)


CELL_PATTERN = re.compile(
    r"^(?P<stamp>.+_stamp)__cell_(?P<index>\d+)\.png$"
)


def load_ocr_results():
    """
    Загружает corrected OCR и группирует его:

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


def process_stamp(
    stamp_path: Path,
    ocr_by_stamp: dict,
):
    """
    Сегментирует один штамп, соединяет геометрию
    с уже готовым OCR и запускает mapper.
    """

    segmenter = StampSegmenter(stamp_path)

    image = segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    bounds = segmenter.find_cells()

    stamp_height, stamp_width = image.shape[:2]

    stamp_ocr = ocr_by_stamp.get(
        stamp_path.stem,
        {},
    )

    cells = []

    for index, cell_bounds in enumerate(
        bounds,
        start=1,
    ):
        ocr = stamp_ocr.get(index)

        if ocr is None:
            text = ""
            confidence = 0.0
        else:
            text = ocr["text"]
            confidence = ocr["confidence"]

        cells.append({
            "index": index,
            "bounds": cell_bounds,
            "text": text,
            "confidence": confidence,
        })

    mapper = StampFieldMapper(
        stamp_width=stamp_width,
        stamp_height=stamp_height,
    )

    result = mapper.map(cells)

    return result, len(bounds), len(stamp_ocr)


def save_results(rows):
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "stamp",
        "cell_count",
        "ocr_count",
        *FIELDS,
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


def print_coverage(rows):
    print()
    print("=" * 100)
    print("FIELD COVERAGE")
    print("=" * 100)
    print()

    total = len(rows)

    for field in FIELDS:
        found = sum(
            1
            for row in rows
            if row[field] != "-"
        )

        missing = total - found

        percentage = (
            found / total * 100
            if total
            else 0.0
        )

        print(
            f"{field:<16} "
            f"{found:>3} / {total:<3} "
            f"({percentage:6.2f}%) "
            f"missing={missing}"
        )


def print_common_values(rows):
    """
    Показывает самые частые найденные значения.

    Полезно для обнаружения очевидной фигни:
    если name внезапно массово становится 'Разраб.'
    или designation ловит не те строки, это будет видно.
    """

    print()
    print("=" * 100)
    print("MOST COMMON VALUES")
    print("=" * 100)

    for field in FIELDS:
        counter = Counter(
            row[field]
            for row in rows
            if row[field] != "-"
        )

        print()
        print(f"{field.upper()}")
        print("-" * 100)

        if not counter:
            print("Нет найденных значений.")
            continue

        for value, count in counter.most_common(15):
            print(
                f"{count:>4}x  {value!r}"
            )


def print_missing_examples(rows):
    """
    Показывает примеры штампов, где mapper
    не нашёл конкретное поле.
    """

    print()
    print("=" * 100)
    print("MISSING FIELD EXAMPLES")
    print("=" * 100)

    for field in FIELDS:
        missing = [
            row["stamp"]
            for row in rows
            if row[field] == "-"
        ]

        print()
        print(
            f"{field}: {len(missing)} missing"
        )

        for stamp in missing[:15]:
            print(f"  {stamp}")

        if len(missing) > 15:
            print(
                f"  ... ещё {len(missing) - 15}"
            )


def main():
    print("=" * 100)
    print("STAMP FIELD MAPPER BATCH TEST")
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

    rows = []
    errors = []

    print()
    print("Запуск mapper...")

    for number, stamp_path in enumerate(
        stamp_paths,
        start=1,
    ):
        try:
            result, cell_count, ocr_count = process_stamp(
                stamp_path,
                ocr_by_stamp,
            )

            data = result.to_dict()

            row = {
                "stamp": stamp_path.name,
                "cell_count": cell_count,
                "ocr_count": ocr_count,
            }

            for field in FIELDS:
                row[field] = data[field]

            rows.append(row)

            found_count = sum(
                data[field] != "-"
                for field in FIELDS
            )

            print(
                f"[{number:03d}/{len(stamp_paths):03d}] "
                f"{stamp_path.name} | "
                f"cells={cell_count:3d} | "
                f"ocr={ocr_count:2d} | "
                f"mapped={found_count}/5"
            )

        except Exception as error:
            errors.append(
                (
                    stamp_path.name,
                    str(error),
                )
            )

            print(
                f"[{number:03d}/{len(stamp_paths):03d}] "
                f"ERROR | "
                f"{stamp_path.name} | "
                f"{error}"
            )

    save_results(rows)

    print_coverage(rows)
    print_common_values(rows)
    print_missing_examples(rows)

    print()
    print("=" * 100)
    print("ИТОГ")
    print("=" * 100)

    print(
        f"Всего штампов:       {len(stamp_paths)}"
    )

    print(
        f"Успешно обработано:  {len(rows)}"
    )

    print(
        f"Ошибок:              {len(errors)}"
    )

    print()
    print("Результат:")
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