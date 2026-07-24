import fitz
from pathlib import Path


class PDFLoader:

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.document = None

    def open(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.pdf_path}"
            )

        self.document = fitz.open(self.pdf_path)

        print("PDF успешно открыт.")
        print(f"Количество страниц: {self.page_count}")

    @property
    def page_count(self):

        if self.document is None:
            return 0

        return len(self.document)
    
    def save_page(self, page_number: int, dpi: int = 300) -> str:

        if self.document is None:
            raise RuntimeError("PDF не открыт.")

        if page_number < 0 or page_number >= self.page_count:
            raise ValueError("Некорректный номер страницы.")

        page = self.document.load_page(page_number)

        pix = page.get_pixmap(dpi=dpi)

        output_dir = Path("data/images")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"page_{page_number + 1:03}.png"

        pix.save(output_path)

        print(f"Страница сохранена: {output_path}")

        return str(output_path)
    
    def save_all_pages(self, dpi: int = 600) -> list[str]:

        if self.document is None:
            raise RuntimeError("PDF не открыт.")

        saved_pages = []

        for page_number in range(self.page_count):
            image_path = self.save_page(page_number, dpi)
            saved_pages.append(image_path)

        print(f"\nСохранено страниц: {len(saved_pages)}")

        return saved_pages