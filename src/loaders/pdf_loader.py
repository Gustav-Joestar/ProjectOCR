from pathlib import Path

import cv2
import fitz
import numpy as np


class PDFLoader:
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.document = None

    @property
    def name(self) -> str:
        return self.pdf_path.stem

    @property
    def page_count(self) -> int:
        if self.document is None:
            raise RuntimeError("PDF не открыт.")

        return len(self.document)

    def open(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF-файл не найден: {self.pdf_path}"
            )

        if not self.pdf_path.is_file():
            raise FileNotFoundError(
                f"Указанный путь не является файлом: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Файл не является PDF: {self.pdf_path}"
            )

        if self.document is not None:
            self.close()

        try:
            self.document = fitz.open(str(self.pdf_path))
        except Exception as error:
            raise RuntimeError(
                f"Не удалось открыть PDF: {self.pdf_path}"
            ) from error

    def render_page(
        self,
        page_number: int,
        dpi: int = 600,
    ) -> np.ndarray:
        if self.document is None:
            raise RuntimeError("PDF не открыт.")

        if not 0 <= page_number < self.page_count:
            raise IndexError(
                f"Страница {page_number + 1} отсутствует. "
                f"Всего страниц: {self.page_count}"
            )

        if dpi <= 0:
            raise ValueError(
                f"DPI должен быть больше нуля: {dpi}"
            )

        page = self.document.load_page(page_number)

        pixmap = page.get_pixmap(
            dpi=dpi,
            alpha=False,
        )

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        return image

    def close(self):
        if self.document is not None:
            self.document.close()
            self.document = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()