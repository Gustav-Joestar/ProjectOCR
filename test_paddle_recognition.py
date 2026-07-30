from pathlib import Path

from paddleocr import PaddleOCR


CELLS_DIR = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR\output\stamp_ocr_debug"
)


def main():

    print("Загрузка recognition модели...")

    ocr = PaddleOCR(
        text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    print("Модель загружена\n")


    # Получаем внутренний predictor распознавания
    rec_model = ocr._pipeline.text_rec_model


    images = sorted(
        CELLS_DIR.glob("cell_*.png")
    )


    print(
        f"Найдено ячеек: {len(images)}\n"
    )


    for image_path in images:

        print("=" * 60)
        print(
            f"Файл: {image_path.name}"
        )


        try:

            result = rec_model.predict(
                str(image_path)
            )


            print(result)


        except Exception as e:

            print(
                "Ошибка:"
            )

            print(e)


if __name__ == "__main__":
    main()