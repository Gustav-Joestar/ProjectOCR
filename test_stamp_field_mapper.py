import csv
import re
from pathlib import Path

from src.ocr.stamp_field_mapper import StampFieldMapper
from src.segmentation.stamp_segmenter import StampSegmenter


PROJECT_ROOT = Path(__file__).resolve().parent

STAMP_PATH = (
    PROJECT_ROOT
    / "output"
    / "stamp_batch"
    / "page_001_drawing_001_stamp.png"
)

OCR_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "ocr_results_corrected.csv"
)


def load_ocr_for_stamp(stamp_stem: str) -> dict[int, dict]:
    """
    Загружает OCR только для ячеек указанного штампа.

    Ожидаемые имена:
    page_001_drawing_001_stamp__cell_024.png
    """

    results = {}

    pattern = re.compile(
        rf"^{re.escape(stamp_stem)}__cell_(\d+)\.png$"
    )

    with OCR_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row.get("file", "")

            match = pattern.match(filename)

            if not match:
                continue

            index = int(match.group(1))

            text = (
                row.get("corrected_text")
                or row.get("raw_text")
                or ""
            ).strip()

            confidence_raw = row.get("confidence", "")

            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0

            results[index] = {
                "text": text,
                "confidence": confidence,
                "raw_text": row.get("raw_text", "").strip(),
            }

    return results


def main():
    print("=" * 100)
    print("REAL STAMP FIELD MAPPER TEST")
    print("=" * 100)

    if not STAMP_PATH.exists():
        print()
        print("Штамп не найден:")
        print(STAMP_PATH)
        return

    if not OCR_CSV.exists():
        print()
        print("OCR CSV не найден:")
        print(OCR_CSV)
        return

    # ---------------------------------------------------------
    # 1. Получаем реальную геометрию ячеек
    # ---------------------------------------------------------

    segmenter = StampSegmenter(STAMP_PATH)

    image = segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    bounds = segmenter.find_cells()

    stamp_height, stamp_width = image.shape[:2]

    print()
    print(f"Штамп: {STAMP_PATH.name}")
    print(f"Размер: {stamp_width} x {stamp_height}")
    print(f"Ячеек сегментатора: {len(bounds)}")

    # ---------------------------------------------------------
    # 2. Загружаем уже готовый OCR
    # ---------------------------------------------------------

    ocr_results = load_ocr_for_stamp(
        STAMP_PATH.stem
    )

    print(f"Ячеек в OCR CSV:    {len(ocr_results)}")

    # ---------------------------------------------------------
    # 3. Соединяем OCR и геометрию
    # ---------------------------------------------------------

    cells = []

    for index, cell_bounds in enumerate(
        bounds,
        start=1,
    ):
        ocr = ocr_results.get(index)

        if ocr is None:
            text = ""
            confidence = 0.0
            raw_text = ""
        else:
            text = ocr["text"]
            confidence = ocr["confidence"]
            raw_text = ocr["raw_text"]

        cells.append({
            "index": index,
            "bounds": cell_bounds,
            "text": text,
            "confidence": confidence,
            "raw_text": raw_text,
        })

    # ---------------------------------------------------------
    # 4. StampFieldMapper
    # ---------------------------------------------------------

    mapper = StampFieldMapper(
        stamp_width=stamp_width,
        stamp_height=stamp_height,
    )

    prepared = mapper.prepare_cells(
        cells,
        include_empty=False,
    )

    result = mapper.map(prepared)

    print()
    print("=" * 100)
    print("MAPPED STAMP DATA")
    print("=" * 100)
    print()

    for field, value in result.to_dict().items():
        print(
            f"{field:<24} = {value!r}"
        )    

    # ---------------------------------------------------------
    # 5. Вывод
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("РАСПОЗНАННЫЕ ЯЧЕЙКИ")
    print("=" * 100)
    print()

    for cell in prepared:
        source = cells[cell.index - 1]
        raw_text = source["raw_text"]

        print(
            f"#{cell.index:03d} | "
            f"{cell.text!r}"
        )

        if raw_text and raw_text != cell.text:
            print(
                f"      raw:        {raw_text!r}"
            )

        print(
            f"      confidence: {cell.confidence:.3f}"
        )

        print(
            "      bounds:     "
            f"({cell.x1}, {cell.y1}, "
            f"{cell.x2}, {cell.y2})"
        )

        print(
            "      normalized: "
            f"({cell.nx1:.4f}, "
            f"{cell.ny1:.4f}, "
            f"{cell.nx2:.4f}, "
            f"{cell.ny2:.4f})"
        )

        print(
            "      center:     "
            f"({cell.normalized_center_x:.4f}, "
            f"{cell.normalized_center_y:.4f})"
        )

        print()

    # ---------------------------------------------------------
    # 6. Отдельно интересующие нас значения/якоря
    # ---------------------------------------------------------

    interesting = {
        "Лит.",
        "Масса",
        "Масштаб",
        "Листов",
        "Листов 1",
        "Разраб.",
        "Пров.",
        "Т. контр.",
        "Н. контр.",
        "Утв.",
        "№ докум.",
        "Опора",
        "5:1",
        "Сталь У8А ГОСТ 1435-99",
        "00-000.06.01.01.07",
    }

    print("=" * 100)
    print("ИНТЕРЕСУЮЩИЕ ПОЛЯ")
    print("=" * 100)
    print()

    found = 0

    for cell in prepared:
        if cell.text not in interesting:
            continue

        found += 1

        print(
            f"#{cell.index:03d} "
            f"{cell.text!r:<32} "
            f"center=("
            f"{cell.normalized_center_x:.4f}, "
            f"{cell.normalized_center_y:.4f}) "
            f"size=("
            f"{cell.normalized_width:.4f}, "
            f"{cell.normalized_height:.4f})"
        )

    print()
    print(f"Найдено интересующих ячеек: {found}")


if __name__ == "__main__":
    main()