from time import perf_counter

from src.interface import select_input_queue
from src.pipeline import (
    process_pdf,
    process_image,
    format_time,
)


def main():
    selection = select_input_queue()

    if selection is None:
        return

    input_type, queue = selection

    total_pages = 0
    total_drawings = 0

    started = perf_counter()

    if input_type == "pdf":
        for pdf_path in queue:
            pages, drawings = process_pdf(
                pdf_path
            )

            total_pages += pages
            total_drawings += drawings

    else:
        for image_path in queue:
            pages, drawings = process_image(
                image_path
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

    if input_type == "pdf":
        print(
            f"📚 PDF: {len(queue)}"
        )
        print(
            f"📄 Страниц: {total_pages}"
        )
    else:
        print(
            f"🖼️ Изображений: {len(queue)}"
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