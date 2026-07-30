import csv
from pathlib import Path

from src.ocr.correction_dictionary import OCRCorrectionDictionary


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "ocr_results.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "ocr_results_corrected.csv"
)


def main():
    print("=" * 80)
    print("BATCH OCR CORRECTIONS")
    print("=" * 80)
    print()

    if not INPUT_CSV.exists():
        print("Не найден исходный CSV:")
        print(INPUT_CSV)
        return

    dictionary = OCRCorrectionDictionary()

    print("Словарь загружен.")
    print()

    rows = []

    total = 0
    with_text = 0
    changed = 0
    unchanged = 0

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print("CSV не содержит заголовка.")
            return

        if "raw_text" not in reader.fieldnames:
            print("В CSV отсутствует колонка raw_text.")
            return

        original_fields = list(reader.fieldnames)

        for row in reader:
            total += 1

            raw_text = row.get(
                "raw_text",
                ""
            ).strip()

            corrected_text = dictionary.correct(
                raw_text
            )

            is_changed = (
                corrected_text != raw_text
            )

            if raw_text:
                with_text += 1

                if is_changed:
                    changed += 1
                else:
                    unchanged += 1

            row["corrected_text"] = corrected_text
            row["changed"] = (
                "1" if is_changed else "0"
            )

            rows.append(row)

    output_fields = original_fields.copy()

    if "corrected_text" not in output_fields:
        output_fields.append(
            "corrected_text"
        )

    if "changed" not in output_fields:
        output_fields.append(
            "changed"
        )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=output_fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    print(f"Всего записей:      {total}")
    print(f"С текстом:          {with_text}")
    print(f"Изменено словарём:  {changed}")
    print(f"Без изменений:      {unchanged}")

    if with_text:
        percent = (
            changed
            / with_text
            * 100
        )

        print(
            f"Доля изменений:     "
            f"{percent:.2f}%"
        )

    print()
    print("Результат:")
    print(OUTPUT_CSV)

    # Небольшая выборка изменений прямо в консоль.
    print()
    print("=" * 80)
    print("ПРИМЕРЫ ИЗМЕНЕНИЙ")
    print("=" * 80)

    shown = 0

    for row in rows:
        if row["changed"] != "1":
            continue

        print()
        print(f"Файл:   {row.get('file', '')}")
        print(f"Было:   {row.get('raw_text', '')!r}")
        print(f"Стало:  {row.get('corrected_text', '')!r}")

        shown += 1

        if shown >= 30:
            break


if __name__ == "__main__":
    main()