from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, image_path: str):

        self.image_path = Path(image_path)

        self.image = None
        self.gray = None
        self.binary = None

    def load(self):

        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.image_path}"
            )

        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise RuntimeError(
                "Не удалось открыть изображение."
            )

        height, width = self.image.shape[:2]

        print(f"Изображение загружено.")
        print(f"Размер: {width} x {height}")

    def to_grayscale(self):

        if self.image is None:
            raise RuntimeError("Изображение не загружено.")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        print("Изображение преобразовано в оттенки серого.")

    def save_gray(self, output_path: str):
        """
        Сохраняет изображение в оттенках серого.
        """

        if self.gray is None:
            raise RuntimeError("Серое изображение отсутствует.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), self.gray)

        print(f"Серое изображение сохранено: {output_path}")

    def to_binary(self):
        """
        Выполняет адаптивную бинаризацию изображения.
        """

        if self.gray is None:
            raise RuntimeError("Серое изображение отсутствует.")

        self.binary = cv2.adaptiveThreshold(
            self.gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

        print("Бинаризация выполнена.")

    def save_binary(self, output_path: str):
        """
        Сохраняет бинарное изображение.
        """

        if self.binary is None:
            raise RuntimeError("Бинарное изображение отсутствует.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), self.binary)

        print(f"Бинарное изображение сохранено: {output_path}")