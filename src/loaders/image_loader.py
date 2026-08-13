from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class ImageLoader:
    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path)

    @property
    def name(self) -> str:
        return self.image_path.stem

    def load(self) -> np.ndarray:
        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Изображение не найдено: {self.image_path}"
            )

        if not self.image_path.is_file():
            raise FileNotFoundError(
                f"Указанный путь не является файлом: {self.image_path}"
            )

        if self.image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемый формат изображения: "
                f"{self.image_path.suffix}"
            )

        image = cv2.imread(
            str(self.image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Не удалось прочитать изображение: {self.image_path}"
            )

        return image

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        pass
