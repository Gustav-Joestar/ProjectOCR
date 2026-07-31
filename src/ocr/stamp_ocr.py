from pathlib import Path

import cv2
from paddleocr import PaddleOCR

from src.segmentation.text_line_segmenter import segment_text_lines
from src.segmentation.cell_filter import analyze_cell
from src.ocr.correction_dictionary import OCRCorrectionDictionary


class StampOCR:
    def __init__(self, image, cells):
        self.image = image
        self.cells = cells
        self.correction_dictionary = OCRCorrectionDictionary()

    def crop_cells(self, padding=5):
        """Вырезает содержимое ячеек без линий рамки."""

        height, width = self.image.shape[:2]
        crops = []

        for index, (x1, y1, x2, y2) in enumerate(
            self.cells,
            start=1
        ):
            left = max(0, x1 + padding)
            top = max(0, y1 + padding)
            right = min(width, x2 - padding)
            bottom = min(height, y2 - padding)

            if right <= left or bottom <= top:
                continue

            crops.append({
                "index": index,
                "bounds": (x1, y1, x2, y2),
                "image": self.image[top:bottom, left:right]
            })

        return crops

    @staticmethod
    def preprocess_crop(image, scale=2):
        """Подготавливает содержимое ячейки для OCR."""

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        return binary

    def load_recognizer(self):
        """
        Загружает PaddleOCR и получает recognition-модель.

        Detection не используется, потому что на вход
        поступают уже сегментированные ячейки штампа.
        """

        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.recognizer = (
            self.ocr
            .paddlex_pipeline
            ._pipeline
            .text_rec_model
        )

    def recognize_line(self, image):
        """Распознаёт одну готовую строку текста."""

        if image is None or image.size == 0:
            return "", 0.0

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        results = list(
            self.recognizer(image)
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

        confidence = float(
            result.get(
                "rec_score",
                0.0
            )
        )

        return text, confidence

    def recognize_cell(self, image):
        """
        Распознаёт содержимое одной ячейки.

        Многострочная ячейка разбивается на строки,
        после чего результаты объединяются пробелом.
        """

        lines = segment_text_lines(image)

        texts = []
        scores = []

        for line in lines:
            text, score = self.recognize_line(
                line
            )

            if text:
                texts.append(text)
                scores.append(score)

        text = " ".join(
            texts
        ).strip()

        confidence = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        return text, confidence

    def recognize(self, padding=5):
        """
        Распознаёт текст только в непустых ячейках штампа.
        """

        if not hasattr(self, "recognizer"):
            self.load_recognizer()

        crops = self.crop_cells(
            padding=padding
        )

        results = []

        for cell in crops:
            image = cell["image"]

            analysis = analyze_cell(image)

            if analysis["is_empty"]:
                continue

            text, confidence = self.recognize_cell(
                image
            )

            raw_text = text

            corrected_text = (
                self.correction_dictionary.correct(
                    raw_text
                )
            )

            results.append({
                "index": cell["index"],
                "bounds": cell["bounds"],
                "raw_text": raw_text,
                "text": corrected_text,
                "confidence": confidence,
            })

        return results

    def close(self):
        """Освобождает PaddleOCR pipeline."""

        if hasattr(self, "ocr"):
            try:
                self.ocr.close()
            except Exception:
                pass

    def save_debug(self, output_dir):
        """Сохраняет вырезанные и подготовленные ячейки."""

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        crops = self.crop_cells()

        for cell in crops:
            index = cell["index"]
            image = cell["image"]

            prepared = self.preprocess_crop(image)

            cv2.imwrite(
                str(
                    output_dir /
                    f"cell_{index:03}.png"
                ),
                prepared
            )

        return crops