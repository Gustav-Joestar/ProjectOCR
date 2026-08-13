from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog


PDF_DIR = Path("data/pdf")
IMAGE_DIR = Path("data/images")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def get_pdf_files() -> list[Path]:
    return sorted(
        PDF_DIR.glob("*.pdf"),
        key=lambda path: path.name.lower(),
    )


def get_image_files() -> list[Path]:
    return sorted(
        (
            path
            for path in IMAGE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    ) if IMAGE_DIR.exists() else []


def add_pdf_files() -> int:
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

        try:
            same_file = source.resolve() == destination.resolve()
        except OSError:
            same_file = False

        if same_file or destination.exists():
            skipped += 1
            continue

        shutil.copy2(source, destination)
        added += 1

    print()

    if added:
        print(f"📥 Добавлено PDF: {added}")

    if skipped:
        print(f"⏭️ Пропущено PDF: {skipped}")

    return added


def add_image_files() -> int:
    root = tk.Tk()
    root.withdraw()

    try:
        selected_files = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[
                (
                    "Изображения",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp",
                ),
                ("Все файлы", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not selected_files:
        print()
        print("ℹ️ Добавление отменено.")
        return 0

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0

    for selected_file in selected_files:
        source = Path(selected_file)

        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped += 1
            continue

        destination = IMAGE_DIR / source.name

        try:
            same_file = source.resolve() == destination.resolve()
        except OSError:
            same_file = False

        if same_file or destination.exists():
            skipped += 1
            continue

        shutil.copy2(source, destination)
        added += 1

    print()

    if added:
        print(f"📥 Добавлено изображений: {added}")

    if skipped:
        print(f"⏭️ Пропущено изображений: {skipped}")

    return added


def show_files_menu(
    files: list[Path],
    title: str,
    item_name: str,
):
    print()
    print("🚀 ProjectOCR")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(title)
    print()

    for number, path in enumerate(files, start=1):
        print(f"  [{number}] {path.name}")

    print()
    print(f"  [0] Обработать все {item_name}")
    print(f"  [a] Добавить {item_name}")
    print("  [q] Выход")
    print()
    print(
        "💡 Можно выбрать несколько файлов, "
        "указав номера через пробел."
    )
    print("   Например: 1 3")
    print()


def select_files(
    files: list[Path],
    item_name: str,
):
    while True:
        choice = input(
            "👉 Выберите действие: "
        ).strip().lower()

        if choice == "q":
            return None

        if choice == "a":
            return []

        if choice == "0":
            print()
            print(
                "────────────────────────────────────────────────────────────"
            )
            return files.copy()

        try:
            numbers = [
                int(value)
                for value in choice.split()
            ]
        except ValueError:
            print(
                f"❌ Введите номер файла, 0, a или q."
            )
            continue

        if not numbers:
            print("❌ Ничего не выбрано.")
            continue

        invalid = [
            number
            for number in numbers
            if not 1 <= number <= len(files)
        ]

        if invalid:
            print(
                "❌ Некорректные номера: "
                + ", ".join(map(str, invalid))
            )
            continue

        numbers = list(dict.fromkeys(numbers))

        selected = [
            files[number - 1]
            for number in numbers
        ]

        print()
        print(
            "────────────────────────────────────────────────────────────"
        )

        return selected


def show_queue(
    queue: list[Path],
    item_name: str,
):
    print()
    print(f"📋 Очередь обработки ({item_name}):")
    print()

    for number, path in enumerate(queue, start=1):
        print(f"  {number}. {path.name}")

    print()
    print(
        "────────────────────────────────────────────────────────────"
    )


def select_pdf_queue() -> list[Path] | None:
    while True:
        pdf_files = get_pdf_files()

        if not pdf_files:
            print()
            print("🚀 ProjectOCR")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print()
            print("📭 В data/pdf нет PDF-файлов.")
            print()
            print("  [a] Добавить PDF")
            print("  [q] Выход")
            print()

            choice = input(
                "👉 Выберите действие: "
            ).strip().lower()

            if choice == "a":
                add_pdf_files()
                continue

            if choice == "q":
                return None

            print("❌ Доступные действия: a или q.")
            continue

        show_files_menu(
            pdf_files,
            "📚 Обнаружены PDF-файлы:",
            "PDF",
        )

        queue = select_files(
            pdf_files,
            "PDF",
        )

        if queue is None:
            return None

        if not queue:
            add_pdf_files()
            continue

        show_queue(queue, "PDF")
        return queue


def select_image_queue() -> list[Path] | None:
    while True:
        image_files = get_image_files()

        if not image_files:
            print()
            print("🚀 ProjectOCR")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print()
            print("📭 В data/images нет изображений.")
            print()
            print("  [a] Добавить изображения")
            print("  [q] Выход")
            print()

            choice = input(
                "👉 Выберите действие: "
            ).strip().lower()

            if choice == "a":
                add_image_files()
                continue

            if choice == "q":
                return None

            print("❌ Доступные действия: a или q.")
            continue

        show_files_menu(
            image_files,
            "🖼️ Обнаружены изображения:",
            "изображения",
        )

        queue = select_files(
            image_files,
            "изображения",
        )

        if queue is None:
            return None

        if not queue:
            add_image_files()
            continue

        show_queue(queue, "изображения")
        return queue


def select_input_queue():
    while True:
        print()
        print("🚀 ProjectOCR")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print("Выберите тип входных данных:")
        print()
        print("  [1] PDF")
        print("  [2] Изображения")
        print("  [q] Выход")
        print()

        choice = input(
            "👉 Выберите действие: "
        ).strip().lower()

        if choice == "1":
            queue = select_pdf_queue()
            if queue is None:
                return None
            return ("pdf", queue)

        if choice == "2":
            queue = select_image_queue()
            if queue is None:
                return None
            return ("image", queue)

        if choice == "q":
            return None

        print("❌ Введите 1, 2 или q.")
