import fitz
from pathlib import Path


class PDFLoader:
    """
    Класс для работы с PDF-документами.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.document = None

    def open(self):
        """
        Открывает PDF-документ.
        """

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.pdf_path}"
            )

        self.document = fitz.open(self.pdf_path)

        print("PDF успешно открыт.")
        print(f"Количество страниц: {self.page_count}")

    @property
    def page_count(self):
        """
        Возвращает количество страниц.
        """

        if self.document is None:
            return 0

        return len(self.document)