from pathlib import Path

import cv2
from paddleocr import PaddleOCR

from src.ocr.stamp_preprocessor import get_preprocessing_methods


PROJECT_ROOT = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR"
)

CELLS_DIR = (
    PROJECT_ROOT
    / "output"
    / "stamp_ocr_debug"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "output"
    / "stamp_preprocessing_debug"
)


def create_recognizer():
    """
    Создаёт PaddleOCR и возвращает только recognition-модель.
    Детектор текста для наших сегментированных ячеек не используется.
    """

    print("Загрузка PaddleOCR...")

    ocr = PaddleOCR(
        text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    rec_model = (
        ocr
        .paddlex_pipeline
        ._pipeline
        .text_rec_model
    )

    print("Recognition-модель загружена.\n")

    return ocr, rec_model


def recognize(rec_model, image):
    """
    Распознаёт одно уже подготовленное изображение.

    Возвращает:
        text
        score
    """

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


def main():

    methods = get_preprocessing_methods()

    images = sorted(
        CELLS_DIR.glob("cell_*.png")
    )

    print(
        f"Найдено ячеек: {len(images)}"
    )

    if not images:
        print(
            f"Ячейки не найдены: {CELLS_DIR}"
        )
        return

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Важно сохранить объект ocr живым,
    # пока используется его внутренняя recognition-модель.
    ocr, rec_model = create_recognizer()

    for image_path in images:

        print("\n" + "=" * 70)
        print(
            f"Файл: {image_path.name}"
        )
        print("-" * 70)

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "Не удалось открыть изображение."
            )
            continue

        for method_name, method in methods.items():

            try:

                processed = method(
                    image
                )

                method_dir = (
                    DEBUG_DIR
                    / method_name
                )

                method_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_path = (
                    method_dir
                    / image_path.name
                )

                cv2.imwrite(
                    str(output_path),
                    processed
                )

                text, score = recognize(
                    rec_model,
                    processed
                )

                print(
                    f"{method_name:12} | "
                    f"{text!r:30} | "
                    f"{score:.3f}"
                )

            except Exception as e:

                print(
                    f"{method_name:12} | "
                    f"ОШИБКА: {e}"
                )

    # Просто чтобы явно удерживать PaddleOCR
    # до конца выполнения программы.
    _ = ocr


if __name__ == "__main__":
    main()