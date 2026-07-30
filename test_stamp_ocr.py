from pathlib import Path

from src.segmentation.stamp_segmenter import StampSegmenter
from src.ocr.stamp_ocr import StampOCR


IMAGE_PATH = Path(
    "output/stamp_batch/page_001_drawing_001_stamp.png"
)

OUTPUT_DIR = Path(
    "output/stamp_ocr_debug"
)


def main():
    print("\n🔤 Тест подготовки штампа к OCR")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    segmenter = StampSegmenter(IMAGE_PATH)

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    cells = segmenter.find_cells()

    print(f"▦ Найдено ячеек: {len(cells)}")

    ocr = StampOCR(
        segmenter.image,
        cells
    )

    crops = ocr.save_debug(
        OUTPUT_DIR
    )

    print(f"✂️ Подготовлено ячеек: {len(crops)}")
    print(f"📁 Результаты: {OUTPUT_DIR}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()