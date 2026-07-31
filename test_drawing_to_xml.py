from pathlib import Path

import cv2

from src.detectors.stamp_detector import StampDetector
from src.segmentation.stamp_segmenter import StampSegmenter
from src.ocr.stamp_ocr import StampOCR
from src.ocr.stamp_field_mapper import StampFieldMapper
from src.exporters.xml_exporter import XMLExporter


DRAWING_PATH = Path(
    "output/drawing/album_dukmasova/page_001_drawing_001.png"
)

TEST_DIR = Path(
    "output/test/drawing_to_xml"
)

STAMP_PATH = TEST_DIR / "stamp.png"

XML_PATH = TEST_DIR / "page_001_drawing_001.xml"


def main():
    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DRAWING -> XML TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Drawing -> Stamp
    # ---------------------------------------------------------

    print("\n[1/4] Поиск основной надписи")

    detector = StampDetector(
        DRAWING_PATH
    )

    detector.load()
    detector.preprocess()
    detector.detect_lines()

    bounds = detector.find_stamp_bounds()

    if bounds is None:
        raise RuntimeError(
            "Основная надпись не найдена."
        )

    print(f"Границы штампа: {bounds}")

    stamp_image = detector.crop_stamp()

    if stamp_image is None or stamp_image.size == 0:
        raise RuntimeError(
            "Не удалось вырезать основную надпись."
        )

    if not cv2.imwrite(
        str(STAMP_PATH),
        stamp_image,
    ):
        raise RuntimeError(
            f"Не удалось сохранить штамп: {STAMP_PATH}"
        )

    stamp_height, stamp_width = (
        stamp_image.shape[:2]
    )

    print(
        f"Размер штампа: "
        f"{stamp_width} x {stamp_height}"
    )

    # ---------------------------------------------------------
    # 2. Stamp -> Cells
    # ---------------------------------------------------------

    print("\n[2/4] Сегментация штампа")

    segmenter = StampSegmenter(
        STAMP_PATH
    )

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    cells = segmenter.find_cells()

    print(
        f"Найдено ячеек: {len(cells)}"
    )

    # ---------------------------------------------------------
    # 3. Cells -> OCR -> StampData
    # ---------------------------------------------------------

    print("\n[3/4] OCR и mapping")

    ocr = StampOCR(
        segmenter.image,
        cells,
    )

    try:
        ocr_results = ocr.recognize()
    finally:
        ocr.close()

    print(
        f"Распознано непустых ячеек: "
        f"{len(ocr_results)}"
    )

    height, width = (
        segmenter.image.shape[:2]
    )

    mapper = StampFieldMapper(
        stamp_width=width,
        stamp_height=height,
    )

    stamp_data = mapper.map(
        ocr_results
    )

    print()
    print("STAMP DATA")
    print("-" * 70)

    for field, value in (
        stamp_data.to_dict().items()
    ):
        print(
            f"{field:<24} = {value!r}"
        )

    # ---------------------------------------------------------
    # 4. StampData -> XML
    # ---------------------------------------------------------

    print("\n[4/4] XML export")

    exporter = XMLExporter()

    xml_path = exporter.export(
        stamp_data=stamp_data,
        output_path=XML_PATH,
    )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)

    print(f"Drawing: {DRAWING_PATH}")
    print(f"Stamp:   {STAMP_PATH}")
    print(f"XML:     {xml_path}")

    print()
    print(
        xml_path.read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()