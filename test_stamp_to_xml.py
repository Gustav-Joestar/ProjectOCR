from pathlib import Path

from src.segmentation.stamp_segmenter import StampSegmenter
from src.ocr.stamp_ocr import StampOCR
from src.ocr.stamp_field_mapper import StampFieldMapper
from src.exporters.xml_exporter import XMLExporter


STAMP_PATH = Path(
    "output/stamp_batch/page_001_drawing_001_stamp.png"
)

OUTPUT_PATH = Path(
    "output/test/page_001_drawing_001.xml"
)


def main():
    print("=" * 70)
    print("STAMP -> XML TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Сегментация штампа
    # ---------------------------------------------------------

    segmenter = StampSegmenter(STAMP_PATH)

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    cells = segmenter.find_cells()

    print(f"Ячеек сегментатора: {len(cells)}")

    # ---------------------------------------------------------
    # 2. OCR + фильтр пустых ячеек + corrections
    # ---------------------------------------------------------

    ocr = StampOCR(
        segmenter.image,
        cells,
    )

    try:
        ocr_results = ocr.recognize()
    finally:
        ocr.close()

    print(f"Распознано ячеек:   {len(ocr_results)}")

    # ---------------------------------------------------------
    # 3. OCR results -> StampData
    # ---------------------------------------------------------

    height, width = segmenter.image.shape[:2]

    mapper = StampFieldMapper(
        stamp_width=width,
        stamp_height=height,
    )

    stamp_data = mapper.map(
        ocr_results
    )

    print()
    print("=" * 70)
    print("STAMP DATA")
    print("=" * 70)

    for field, value in stamp_data.to_dict().items():
        print(
            f"{field:<24} = {value!r}"
        )

    # ---------------------------------------------------------
    # 4. StampData -> XML
    # ---------------------------------------------------------

    exporter = XMLExporter()

    xml_path = exporter.export(
        stamp_data=stamp_data,
        output_path=OUTPUT_PATH,
    )

    print()
    print("=" * 70)
    print("XML")
    print("=" * 70)
    print(f"Создан: {xml_path}")
    print()
    print(
        xml_path.read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()