from time import perf_counter

from src.interface import select_pdf_queue
from src.pipeline import process_pdf, format_time


def main():
    pdf_queue = select_pdf_queue()

    if pdf_queue is None:
        return

    total_pages = 0
    total_drawings = 0

    started = perf_counter()

    for pdf_path in pdf_queue:
        pages, drawings = process_pdf(
            pdf_path
        )

        total_pages += pages
        total_drawings += drawings

    elapsed = perf_counter() - started

    print()
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print("🎉 Обработка завершена")
    print()
    print(
        f"📚 PDF: {len(pdf_queue)}"
    )
    print(
        f"📄 Страниц: {total_pages}"
    )
    print(
        f"✂️ Чертежей: {total_drawings}"
    )
    print(
        f"⏱️ Время: {format_time(elapsed)}"
    )
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


if __name__ == "__main__":
    main()