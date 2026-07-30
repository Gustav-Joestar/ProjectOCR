import csv
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

from src.segmentation.text_line_segmenter import segment_text_lines


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "cell_filter_batch"
    / "non_empty"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
)

CSV_PATH = OUTPUT_DIR / "ocr_results.csv"


def create_recognizer():
    """
    Загружает PaddleOCR и получает только recognition-модель.

    Detection нам здесь не нужен:
    на вход уже поступают отдельные строки текста.
    """

    print("Загрузка PaddleOCR...")

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    recognizer = (
        ocr
        .paddlex_pipeline
        ._pipeline
        .text_rec_model
    )

    print("Recognition-модель загружена.")
    print()

    return ocr, recognizer


def recognize_line(
    recognizer,
    image
):
    """
    Распознаёт одну готовую строку текста.
    """

    if image is None or image.size == 0:
        return "", 0.0

    # Recognition-модель ожидает 3 канала.
    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    results = list(
        recognizer(image)
    )

    if not results:
        return "", 0.0

    result = results[0]

    text = str(
        result.get(
            "rec_text",
            ""
        )
    ).strip()

    score = float(
        result.get(
            "rec_score",
            0.0
        )
    )

    return text, score


def get_images():
    """
    Находит все непустые ячейки.
    """

    return sorted(
        path
        for path in INPUT_DIR.glob("*.png")
        if path.is_file()
    )


def main():
    print("=" * 80)
    print("BATCH OCR")
    print("=" * 80)
    print()

    if not INPUT_DIR.exists():
        print("Не найдена папка:")
        print(INPUT_DIR)
        return

    images = get_images()

    if not images:
        print("Не найдено изображений:")
        print(INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Найдено ячеек: {len(images)}")
    print()

    ocr, recognizer = create_recognizer()

    rows = []

    successful = 0
    empty_ocr = 0
    errors = 0

    try:
        for number, path in enumerate(
            images,
            start=1
        ):
            print(
                f"[{number}/{len(images)}] "
                f"{path.name}"
            )

            image = cv2.imread(
                str(path)
            )

            if image is None:
                print("  ERROR: изображение не открыто")

                errors += 1

                rows.append({
                    "file": path.name,
                    "raw_text": "",
                    "confidence": "",
                    "line_count": 0,
                    "status": "IMAGE_ERROR",
                })

                continue

            try:
                lines = segment_text_lines(
                    image
                )

                texts = []
                scores = []

                for line_number, line in enumerate(
                    lines,
                    start=1
                ):
                    text, score = recognize_line(
                        recognizer,
                        line
                    )

                    if text:
                        texts.append(text)
                        scores.append(score)

                    print(
                        f"  line {line_number}: "
                        f"{text!r} | "
                        f"{score:.3f}"
                    )

                # В CSV многострочную ячейку пока
                # сохраняем через пробел.
                raw_text = " ".join(
                    texts
                ).strip()

                if scores:
                    confidence = sum(scores) / len(scores)
                else:
                    confidence = 0.0

                if raw_text:
                    status = "OK"
                    successful += 1
                else:
                    status = "EMPTY_OCR"
                    empty_ocr += 1

                rows.append({
                    "file": path.name,
                    "raw_text": raw_text,
                    "confidence": f"{confidence:.6f}",
                    "line_count": len(lines),
                    "status": status,
                })

                print(
                    f"  -> {raw_text!r} | "
                    f"{confidence:.3f}"
                )

            except Exception as error:
                print(
                    f"  OCR ERROR: {error}"
                )

                errors += 1

                rows.append({
                    "file": path.name,
                    "raw_text": "",
                    "confidence": "",
                    "line_count": 0,
                    "status": "OCR_ERROR",
                })

    finally:
        # Корректно освобождаем pipeline.
        try:
            ocr.close()
        except Exception:
            pass

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "raw_text",
                "confidence",
                "line_count",
                "status",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    print(f"Всего ячеек: {len(images)}")
    print(f"Распознано:  {successful}")
    print(f"Без текста:   {empty_ocr}")
    print(f"Ошибок:       {errors}")

    print()
    print("CSV:")
    print(CSV_PATH)


if __name__ == "__main__":
    main()