from pathlib import Path
from time import perf_counter
import shutil
import tkinter as tk
from tkinter import filedialog

from src.loaders.pdf_loader import PDFLoader
from src.detectors.drawing_detector import DrawingDetector


PDF_DIR = Path("data/pdf")

PAGES_DIR = Path("output/pages")
DETECTION_DIR = Path("output/detection")
DRAWING_DIR = Path("output/drawing")


def format_time(seconds: float) -> str:
    """
    Форматирует время в MM:SS или HH:MM:SS.
    """
    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"

def print_progress(
    current: int,
    total: int,
    elapsed: float
):
    """
    Выводит прогресс обработки в одной строке.
    """
    progress = current / total
    percent = progress * 100

    bar_length = 30
    filled_length = int(bar_length * progress)

    bar = (
        "█" * filled_length
        + "░" * (bar_length - filled_length)
    )

    if current > 0:
        average_time = elapsed / current
        remaining = average_time * (total - current)
    else:
        remaining = 0

    print(
        f"\r⏳ {bar} "
        f"{percent:6.2f}% | "
        f"{format_time(elapsed)} / "
        f"{format_time(remaining)}",
        end="",
        flush=True
    )

def process_page(
    image_path: Path,
    page_number: int,
    detection_dir: Path,
    drawing_dir: Path
) -> int:
    """
    Выполняет детекцию одной страницы и сохраняет результаты.

    Возвращает количество найденных чертежей.
    """
    detector = DrawingDetector(image_path)

    detector.load()
    detector.preprocess()

    detector.extract_horizontal_lines()
    detector.extract_vertical_lines()
    detector.combine_lines()
    detector.connect_lines()

    drawings = detector.find_drawings()

    preview_path = (
        detection_dir
        / f"page_{page_number:03d}.png"
    )

    detector.save_detection_preview(
        preview_path
    )

    detector.save_drawings(
        drawing_dir,
        page_number
    )

    return len(drawings)

def clear_pdf_output(pdf_name: str):
    """
    Удаляет результаты предыдущей обработки PDF.
    """
    output_dirs = (
        PAGES_DIR / pdf_name,
        DETECTION_DIR / pdf_name,
        DRAWING_DIR / pdf_name,
    )

    for output_dir in output_dirs:
        if output_dir.exists():
            shutil.rmtree(output_dir)

def process_pdf(pdf_path: Path):
    """
    Полностью обрабатывает один PDF:
    PDF -> страницы -> детекция -> чертежи.
    """
    pdf_name = pdf_path.stem

    pages_dir = PAGES_DIR / pdf_name
    detection_dir = DETECTION_DIR / pdf_name
    drawing_dir = DRAWING_DIR / pdf_name

    clear_pdf_output(pdf_name)

    print()
    print(f"📘 Обработка: {pdf_path.name}")

    start_time = perf_counter()
    total_drawings = 0

    with PDFLoader(pdf_path) as loader:
        total_pages = loader.page_count

        print(f"📄 Страниц: {total_pages}")
        print()

        print_progress(
            current=0,
            total=total_pages,
            elapsed=0
        )

        for page_index in range(total_pages):
            page_number = page_index + 1

            image_path = loader.save_page(
                page_number=page_index,
                output_dir=pages_dir
            )

            drawing_count = process_page(
                image_path=image_path,
                page_number=page_number,
                detection_dir=detection_dir,
                drawing_dir=drawing_dir
            )

            total_drawings += drawing_count

            elapsed = perf_counter() - start_time

            print_progress(
                current=page_number,
                total=total_pages,
                elapsed=elapsed
            )

    elapsed = perf_counter() - start_time

    print()
    print()
    print(f"✅ {pdf_path.name} обработан")
    print(f"📄 Обработано страниц: {total_pages}")
    print(f"✂️ Извлечено чертежей: {total_drawings}")
    print(f"⏱️ Время: {format_time(elapsed)}")

    return total_pages, total_drawings

def get_pdf_files() -> list[Path]:
    """
    Возвращает список PDF-файлов из data/pdf.
    """
    return sorted(
        PDF_DIR.glob("*.pdf"),
        key=lambda path: path.name.lower()
    )

def add_pdf_files() -> int:
    """
    Открывает системное окно выбора PDF и копирует
    выбранные файлы в data/pdf.

    Возвращает количество добавленных файлов.
    """
    root = tk.Tk()
    root.withdraw()

    try:
        selected_files = filedialog.askopenfilenames(
            title="Выберите PDF-файлы",
            filetypes=[
                ("PDF-файлы", "*.pdf"),
                ("Все файлы", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not selected_files:
        print()
        print("ℹ️ Добавление отменено.")
        return 0

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0

    for selected_file in selected_files:
        source = Path(selected_file)

        if source.suffix.lower() != ".pdf":
            skipped += 1
            continue

        destination = PDF_DIR / source.name

        # Если пользователь выбрал файл,
        # который уже лежит в data/pdf.
        try:
            if source.resolve() == destination.resolve():
                skipped += 1
                continue
        except OSError:
            pass

        # Существующий PDF не перезаписываем.
        if destination.exists():
            skipped += 1
            continue

        shutil.copy2(
            source,
            destination
        )

        added += 1

    print()

    if added:
        print(f"📥 Добавлено PDF: {added}")

    if skipped:
        print(f"⏭️ Пропущено PDF: {skipped}")

    return added

def show_pdf_menu(pdf_files: list[Path]):
    """
    Выводит список доступных PDF-файлов.
    """
    print()
    print("🚀 ProjectOCR")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("📚 Обнаружены PDF-файлы:")
    print()

    for number, pdf_path in enumerate(pdf_files, start=1):
        print(f"  [{number}] {pdf_path.name}")

    print()
    print("  [0] Обработать все PDF")
    print("  [a] Добавить PDF")
    print("  [q] Выход")
    print()
    print("💡 Можно выбрать несколько PDF, указав номера через пробел.")
    print("   Например: 1 3")
    print()

def select_pdf_files(
    pdf_files: list[Path]
) -> list[Path] | None:
    """
    Запрашивает у пользователя PDF-файлы для обработки.

    Возвращает:
        list[Path] — очередь обработки;
        []         — запрос на добавление PDF;
        None       — выход из программы.
    """
    while True:
        choice = input("👉 Выберите действие: ").strip().lower()

        if choice == "q":
            return None

        if choice == "a":
            return []

        if choice == "0":
            return pdf_files.copy()

        try:
            numbers = [
                int(value)
                for value in choice.split()
            ]
        except ValueError:
            print("❌ Введите номер PDF, 0, a или q.")
            continue

        if not numbers:
            print("❌ Ничего не выбрано.")
            continue

        invalid_numbers = [
            number
            for number in numbers
            if number < 1 or number > len(pdf_files)
        ]

        if invalid_numbers:
            invalid_text = ", ".join(
                str(number)
                for number in invalid_numbers
            )

            print(
                f"❌ PDF с номером {invalid_text} "
                "в списке нет."
            )
            continue

        # Убираем повторения, сохраняя порядок.
        unique_numbers = list(dict.fromkeys(numbers))

        return [
            pdf_files[number - 1]
            for number in unique_numbers
        ]

def show_queue(pdf_queue: list[Path]):
    """
    Выводит сформированную очередь обработки.
    """
    print()
    print("📋 Очередь обработки:")
    print()

    for number, pdf_path in enumerate(pdf_queue, start=1):
        print(f"  {number}. {pdf_path.name}")

    print()

def main():
    while True:
        pdf_files = get_pdf_files()

        if not pdf_files:
            print()
            print("🚀 ProjectOCR")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print()
            print("📭 В папке data/pdf пока нет PDF-файлов.")
            print()
            print("  [a] Добавить PDF")
            print("  [q] Выход")
            print()

            choice = input("👉 Выберите действие: ").strip().lower()

            if choice == "a":
                add_pdf_files()
                continue

            if choice == "q":
                print()
                print("👋 Работа завершена.")
                return

            print()
            print("❌ Доступные действия: a или q.")
            continue

        show_pdf_menu(pdf_files)

        pdf_queue = select_pdf_files(pdf_files)

        if pdf_queue is None:
            print()
            print("👋 Работа завершена.")
            return

        # [] означает выбор команды "a".
        if not pdf_queue:
            add_pdf_files()
            continue

        show_queue(pdf_queue)

        total_pages = 0
        total_drawings = 0

        program_start = perf_counter()

        for pdf_path in pdf_queue:
            pages, drawings = process_pdf(pdf_path)

            total_pages += pages
            total_drawings += drawings

        total_elapsed = perf_counter() - program_start

        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🎉 Обработка завершена")
        print()
        print(f"📚 Обработано PDF: {len(pdf_queue)}")
        print(f"📄 Всего страниц: {total_pages}")
        print(f"✂️ Всего чертежей: {total_drawings}")
        print(f"⏱️ Общее время: {format_time(total_elapsed)}")
        print()
        print(f"📁 Страницы:  {PAGES_DIR}")
        print(f"🔍 Детекция:  {DETECTION_DIR}")
        print(f"✂️ Чертежи:   {DRAWING_DIR}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return

if __name__ == "__main__":
    main()