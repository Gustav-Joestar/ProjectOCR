from pathlib import Path

from paddleocr import PaddleOCR


# Папка с непустыми ячейками штампа
CELLS_DIR = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR\output\stamp_ocr_debug"
)


def main():

    print("Загрузка PaddleOCR...")

    ocr = PaddleOCR(
        lang="ru",

        # отключаем лишние этапы
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # оставляем модель распознавания
        text_detection_model_name=None,
        text_recognition_model_name="eslav_PP-OCRv5_mobile_rec"
    )

    print("PaddleOCR загружен\n")


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
            result = ocr.predict(
                str(image_path)
            )


            if not result:
                print(
                    "Результат пустой"
                )
                continue


            for res in result:

                try:

                    texts = res["rec_texts"]
                    scores = res["rec_scores"]


                    if not texts:
                        print(
                            "Текст не найден"
                        )
                        continue


                    for text, score in zip(
                        texts,
                        scores
                    ):

                        print(
                            f"Текст: {text}"
                        )

                        print(
                            f"Confidence: {score:.3f}"
                        )


                except Exception:

                    print(
                        "Не удалось разобрать результат:"
                    )

                    print(res)


        except Exception as e:

            print(
                "Ошибка OCR:"
            )

            print(e)


if __name__ == "__main__":
    main()