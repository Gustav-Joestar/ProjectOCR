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
    print("\n🔤 Тест OCR штампа")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ---------------------------------------------------------
    # Сегментация штампа
    # ---------------------------------------------------------

    segmenter = StampSegmenter(IMAGE_PATH)

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    cells = segmenter.find_cells()

    print(f"▦ Найдено ячеек: {len(cells)}")

    # ---------------------------------------------------------
    # OCR
    # ---------------------------------------------------------

    ocr = StampOCR(
        segmenter.image,
        cells,
    )

    # Сохраняем debug-изображения всех ячеек.
    crops = ocr.save_debug(
        OUTPUT_DIR
    )

    print("\n🔎 Распознавание текста")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        results = ocr.recognize()

        for result in results:
            text = result["text"]
            raw_text = result["raw_text"]
            confidence = result["confidence"]

            if not text:
                continue

            if raw_text != text:
                print(
                    f"#{result['index']:03} "
                    f"{raw_text!r} -> {text!r} | "
                    f"{confidence:.3f}"
                )
            else:
                print(
                    f"#{result['index']:03} "
                    f"{text!r} | "
                    f"{confidence:.3f}"
                )

    finally:
        ocr.close()

    # ---------------------------------------------------------
    # Итог
    # ---------------------------------------------------------

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✂️ Подготовлено ячеек: {len(crops)}")
    print(f"🔤 Распознано непустых ячеек: {len(results)}")
    print(f"📁 Debug: {OUTPUT_DIR}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()