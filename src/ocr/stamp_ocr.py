from pathlib import Path

import cv2


class StampOCR:
    def __init__(self, image, cells):
        self.image = image
        self.cells = cells

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