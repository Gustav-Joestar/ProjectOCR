from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog


PDF_DIR = Path("data/pdf")


def get_pdf_files() -> list[Path]:
    return sorted(
        PDF_DIR.glob("*.pdf"),
        key=lambda path: path.name.lower(),
    )


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

    PDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    added = 0
    skipped = 0

    for selected_file in selected_files:
        source = Path(selected_file)

        if source.suffix.lower() != ".pdf":
            skipped += 1
            continue

        destination = PDF_DIR / source.name

        try:
            same_file = (
                source.resolve()
                == destination.resolve()
            )
        except OSError:
            same_file = False

        if same_file or destination.exists():
            skipped += 1
            continue

        shutil.copy2(
            source,
            destination,
        )

        added += 1

    print()

    if added:
        print(
            f"📥 Добавлено PDF: {added}"
        )

    if skipped:
        print(
            f"⏭️ Пропущено PDF: {skipped}"
        )

    return added


def show_pdf_menu(
    pdf_files: list[Path],
):
    print()
    print("🚀 ProjectOCR")
    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    print()
    print("📚 Обнаружены PDF-файлы:")
    print()

    for number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        print(
            f"  [{number}] {pdf_path.name}"
        )

    print()
    print("  [0] Обработать все PDF")
    print("  [a] Добавить PDF")
    print("  [q] Выход")
    print()
    print(
        "💡 Можно выбрать несколько PDF, "
        "указав номера через пробел."
    )
    print("   Например: 1 3")
    print()


def select_pdf_files(
    pdf_files: list[Path],
) -> list[Path] | None:
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
            return pdf_files.copy()

        try:
            numbers = [
                int(value)
                for value in choice.split()
            ]
        except ValueError:
            print(
                "❌ Введите номер PDF, "
                "0, a или q."
            )
            continue

        if not numbers:
            print(
                "❌ Ничего не выбрано."
            )
            continue

        invalid = [
            number
            for number in numbers
            if not (
                1
                <= number
                <= len(pdf_files)
            )
        ]

        if invalid:
            print(
                "❌ Некорректные номера: "
                + ", ".join(
                    map(str, invalid)
                )
            )
            continue

        numbers = list(
            dict.fromkeys(numbers)
        )

        selected = [
            pdf_files[number - 1]
            for number in numbers
        ]

        print()
        print(
            "────────────────────────────────────────────────────────────"
        )

        return selected


def show_queue(
    pdf_queue: list[Path],
):
    print()
    print("📋 Очередь обработки:")
    print()

    for number, pdf_path in enumerate(
        pdf_queue,
        start=1,
    ):
        print(
            f"  {number}. {pdf_path.name}"
        )

    print()
    print(
        "────────────────────────────────────────────────────────────"
    )


def select_pdf_queue() -> list[Path] | None:
    """
    Полный пользовательский интерфейс выбора PDF.

    Возвращает:
        list[Path] — готовая очередь обработки;
        None       — пользователь завершил программу.
    """

    while True:
        pdf_files = get_pdf_files()

        if not pdf_files:
            print()
            print("🚀 ProjectOCR")
            print(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            print()
            print(
                "📭 В data/pdf нет PDF-файлов."
            )
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

            print(
                "❌ Доступные действия: "
                "a или q."
            )
            continue

        show_pdf_menu(
            pdf_files
        )

        pdf_queue = select_pdf_files(
            pdf_files
        )

        if pdf_queue is None:
            return None

        if not pdf_queue:
            add_pdf_files()
            continue

        show_queue(
            pdf_queue
        )

        return pdf_queue