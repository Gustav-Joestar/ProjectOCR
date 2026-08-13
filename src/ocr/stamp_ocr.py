import logging
import warnings

import cv2
from paddleocr import PaddleOCR

# Убираем служебный INFO-вывод Paddle/PaddleX:
# "Creating model..."
# "Model files already exist..."
for logger_name in (
    "paddlex",
    "paddleocr",
):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message=r"No ccache found.*",
    category=UserWarning,
)

from src.segmentation.text_line_segmenter import segment_text_lines
from src.segmentation.cell_filter import analyze_cell
from src.ocr.correction_dictionary import OCRCorrectionDictionary


class StampOCR:
    def __init__(
        self,
        image,
        cells,
        ocr=None,
        recognizer=None,
    ):
        self.image = image
        self.cells = cells
        self.ocr = ocr
        self.recognizer = recognizer
        self.correction_dictionary = OCRCorrectionDictionary()

    def crop_cells(self, padding=5):
        """Вырезает содержимое ячеек без линий рамки."""

        height, width = self.image.shape[:2]
        crops = []

        for index, (x1, y1, x2, y2) in enumerate(
            self.cells,
            start=1,
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
                "image": self.image[
                    top:bottom,
                    left:right
                ]
            })

        return crops

    def load_recognizer(self):
        """
        Загружает PaddleOCR и получает recognition-модель.

        Detection не используется, потому что на вход
        поступают уже сегментированные ячейки штампа.
        """

        if self.recognizer is not None:
            return self.recognizer

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

        return self.recognizer

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

        text = " ".join(texts).strip()

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