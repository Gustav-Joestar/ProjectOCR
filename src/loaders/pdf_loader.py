from pathlib import Path

import fitz


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

    def save_page(
        self,
        page_number: int,
        output_dir: str | Path,
        dpi: int = 600
    ) -> Path:
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

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        page = self.document.load_page(page_number)

        pixmap = page.get_pixmap(
            dpi=dpi,
            alpha=False
        )

        output_path = (
            output_dir
            / f"page_{page_number + 1:03d}.png"
        )

        try:
            pixmap.save(str(output_path))
        except Exception as error:
            raise RuntimeError(
                f"Не удалось сохранить страницу: {output_path}"
            ) from error

        return output_path

    def close(self):
        if self.document is not None:
            self.document.close()
            self.document = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()