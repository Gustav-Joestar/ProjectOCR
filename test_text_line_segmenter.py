from pathlib import Path
import csv
import re

import cv2
from paddleocr import PaddleOCR

from src.segmentation.text_line_segmenter import segment_text_lines
from src.ocr.ocr_postprocessor import postprocess_ocr


PROJECT_ROOT = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "stamp_ocr_debug"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "output"
    / "stamp_ocr_debug"
    / "ground_truth.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "text_line_debug"
)


def load_ground_truth():
    ground_truth = {}

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            filename = row["filename"].strip()
            field_type = row["type"].strip()
            expected = row["expected"].strip()

            ground_truth[filename] = {
                "type": field_type,
                "expected": expected,
            }

    return ground_truth


def create_recognizer():

    print("Загрузка модели распознавания...")

    ocr = PaddleOCR(
        text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rec_model = (
        ocr
        .paddlex_pipeline
        ._pipeline
        .text_rec_model
    )

    print("Модель загружена.\n")

    return ocr, rec_model


def recognize_line(rec_model, image):

    results = list(
        rec_model.predict(image)
    )

    if not results:
        return "", 0.0

    result = results[0]

    text = result.get(
        "rec_text",
        ""
    )

    score = float(
        result.get(
            "rec_score",
            0.0
        )
    )

    return text, score


def normalize_for_comparison(text):
    """
    Минимальная нормализация ТОЛЬКО для оценки результата.

    Не исправляет OCR-ошибки.
    Не меняет буквы.
    Не использует словарь.

    Убирает разницу между:
        "Сталь У8А\\nГОСТ 1435-99"
    и:
        "Сталь У8А ГОСТ 1435-99"
    """

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def main():

    ground_truth = load_ground_truth()

    print(
        f"Ground truth записей: "
        f"{len(ground_truth)}"
    )

    images = sorted(
        INPUT_DIR.glob("cell_*.png")
    )

    print(
        f"Найдено ячеек: "
        f"{len(images)}"
    )

    if not images:
        print(
            f"Нет изображений в: "
            f"{INPUT_DIR}"
        )
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for old_file in OUTPUT_DIR.glob(
        "cell_*_line_*.png"
    ):
        old_file.unlink()

    ocr, rec_model = create_recognizer()

    total = 0
    correct = 0
    errors = []

    print("=" * 80)
    print("BASELINE PADDLE OCR")
    print("=" * 80)

    for image_path in images:

        gt = ground_truth.get(
            image_path.name
        )

        if gt is None:
            print(
                f"\nНет ground truth: "
                f"{image_path.name}"
            )
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"\nНе удалось открыть: "
                f"{image_path.name}"
            )
            continue

        lines = segment_text_lines(
            image
        )

        recognized_lines = []
        scores = []

        for i, line in enumerate(
            lines,
            start=1
        ):

            output_path = (
                OUTPUT_DIR
                / (
                    f"{image_path.stem}"
                    f"_line_{i:02d}.png"
                )
            )

            cv2.imwrite(
                str(output_path),
                line
            )

            try:
                text, score = recognize_line(
                    rec_model,
                    line
                )

            except Exception as e:

                print(
                    f"\n{image_path.name}: "
                    f"ошибка OCR: {e}"
                )

                continue

            recognized_lines.append(
                text
            )

            scores.append(
                score
            )

        # Сохраняем переносы строк как реальный
        # результат OCR.
        recognized = "\n".join(
            recognized_lines
        )

        expected = gt["expected"]
        field_type = gt["type"]

        processed = postprocess_ocr(
            recognized,
            field_type
        )

        expected_cmp = (
            normalize_for_comparison(
                expected
            )
        )

        recognized_cmp = (
            normalize_for_comparison(
                processed
            )
        )

        is_correct = (
            expected_cmp
            == recognized_cmp
        )

        total += 1

        if is_correct:
            correct += 1
        else:
            errors.append({
                "filename": image_path.name,
                "type": field_type,
                "expected": expected,
                "recognized": recognized,
                "scores": scores,
            })

        status = (
            "OK"
            if is_correct
            else "ERROR"
        )

        print()
        print("-" * 80)

        print(
            f"{image_path.name} "
            f"[{field_type}] "
            f"-> {status}"
        )

        print(
            f"Ожидалось:   "
            f"{expected!r}"
        )

        print(
            f"После:       "
            f"{processed!r}"
        )

        if scores:

            score_string = ", ".join(
                f"{score:.3f}"
                for score in scores
            )

            print(
                f"Confidence:  "
                f"{score_string}"
            )

        print(
            f"Строк:       "
            f"{len(lines)}"
        )

    print()
    print("=" * 80)
    print("ИТОГ BASELINE")
    print("=" * 80)

    print(
        f"Всего:       {total}"
    )

    print(
        f"Правильно:   {correct}"
    )

    print(
        f"Ошибочно:    "
        f"{total - correct}"
    )

    if total:

        accuracy = (
            correct / total * 100
        )

        print(
            f"Точность:    "
            f"{accuracy:.2f}%"
        )

    print()
    print("=" * 80)
    print("ОШИБКИ")
    print("=" * 80)

    if not errors:

        print(
            "Ошибок нет."
        )

    else:

        for error in errors:

            print()

            print(
                f"{error['filename']} "
                f"[{error['type']}]"
            )

            print(
                f"  GT:  "
                f"{error['expected']!r}"
            )

            print(
                f"  OCR: "
                f"{error['recognized']!r}"
            )

    print()
    print("=" * 80)

    _ = ocr


if __name__ == "__main__":
    main()