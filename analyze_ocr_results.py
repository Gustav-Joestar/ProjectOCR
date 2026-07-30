import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "ocr_results_corrected.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "batch_ocr"
    / "ocr_analysis.csv"
)


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")


def parse_confidence(value):
    """
    Пытается извлечь confidence из разных вариантов записи.

    Примеры:
        "0.923"
        "0.970, 0.877"
        "[0.970, 0.877]"
    """

    if not value:
        return None

    numbers = re.findall(
        r"(?:0(?:\.\d+)?|1(?:\.0+)?)",
        str(value)
    )

    if not numbers:
        return None

    values = []

    for number in numbers:
        try:
            values.append(float(number))
        except ValueError:
            pass

    if not values:
        return None

    # Для многострочного OCR берём минимальный confidence.
    # Если одна строка распознана плохо, это важно увидеть.
    return min(values)


def has_cyrillic(text):
    return bool(CYRILLIC_RE.search(text))


def has_latin(text):
    return bool(LATIN_RE.search(text))


def has_digit(text):
    return bool(DIGIT_RE.search(text))


def is_mixed_alphabet(text):
    """
    Кириллица + латиница одновременно.
    Например:
        Оnора
        FОСТ
    """
    return has_cyrillic(text) and has_latin(text)


def is_short_suspicious(text):
    """
    Очень короткие результаты потенциально подозрительны.

    При этом цифры сами по себе не запрещаем:
    в штампе могут быть номера листов и другие значения.
    """

    stripped = text.strip()

    if not stripped:
        return False

    return len(stripped) <= 2


def get_suspicion_reasons(text, confidence):
    reasons = []

    if is_mixed_alphabet(text):
        reasons.append("mixed_alphabet")

    if is_short_suspicious(text):
        reasons.append("short_text")

    if confidence is not None:
        if confidence < 0.40:
            reasons.append("very_low_conf")
        elif confidence < 0.60:
            reasons.append("low_conf")

    # Странный одиночный знак.
    if len(text.strip()) == 1:
        char = text.strip()

        if not char.isalnum():
            reasons.append("single_symbol")

    return reasons


def main():
    print("=" * 80)
    print("OCR RESULT ANALYSIS")
    print("=" * 80)
    print()

    if not INPUT_CSV.exists():
        print("Не найден файл:")
        print(INPUT_CSV)
        return

    groups = defaultdict(
        lambda: {
            "count": 0,
            "confidences": [],
            "files": [],
            "raw_variants": set(),
        }
    )

    total = 0
    empty = 0

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print("CSV не содержит заголовка.")
            return

        print("Колонки CSV:")
        print(", ".join(reader.fieldnames))
        print()

        for row in reader:
            total += 1

            corrected_text = (
                row.get("corrected_text")
                or row.get("raw_text")
                or ""
            ).strip()

            raw_text = (
                row.get("raw_text")
                or ""
            ).strip()

            if not corrected_text:
                empty += 1
                continue

            # Ищем confidence независимо от точного имени колонки.
            confidence_raw = (
                row.get("confidence")
                or row.get("conf")
                or row.get("score")
                or row.get("rec_score")
                or ""
            )

            confidence = parse_confidence(
                confidence_raw
            )

            group = groups[corrected_text]

            group["count"] += 1

            if confidence is not None:
                group["confidences"].append(
                    confidence
                )

            file_name = (
                row.get("file")
                or row.get("filename")
                or row.get("cell")
                or ""
            )

            if file_name:
                group["files"].append(file_name)

            if raw_text:
                group["raw_variants"].add(
                    raw_text
                )

    analysis_rows = []

    for text, data in groups.items():

        confidences = data["confidences"]

        avg_conf = (
            mean(confidences)
            if confidences
            else None
        )

        min_conf = (
            min(confidences)
            if confidences
            else None
        )

        reasons = get_suspicion_reasons(
            text,
            min_conf
        )

        analysis_rows.append({
            "text": text,
            "count": data["count"],
            "avg_confidence": (
                f"{avg_conf:.3f}"
                if avg_conf is not None
                else ""
            ),
            "min_confidence": (
                f"{min_conf:.3f}"
                if min_conf is not None
                else ""
            ),
            "mixed_alphabet": (
                1 if is_mixed_alphabet(text) else 0
            ),
            "short_text": (
                1 if is_short_suspicious(text) else 0
            ),
            "suspicious": (
                1 if reasons else 0
            ),
            "reasons": "|".join(reasons),
            "raw_variants": " | ".join(
                sorted(data["raw_variants"])
            ),
            "example_file": (
                data["files"][0]
                if data["files"]
                else ""
            ),
        })

    # Сначала подозрительные.
    # Среди них — самые частые.
    analysis_rows.sort(
        key=lambda row: (
            -row["suspicious"],
            -row["count"],
            row["text"].lower(),
        )
    )

    fields = [
        "text",
        "count",
        "avg_confidence",
        "min_confidence",
        "mixed_alphabet",
        "short_text",
        "suspicious",
        "reasons",
        "raw_variants",
        "example_file",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(analysis_rows)

    suspicious_rows = [
        row
        for row in analysis_rows
        if row["suspicious"] == 1
    ]

    mixed_rows = [
        row
        for row in analysis_rows
        if row["mixed_alphabet"] == 1
    ]

    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    print(f"Всего записей:          {total}")
    print(f"Пустых:                 {empty}")
    print(f"Уникальных текстов:     {len(groups)}")
    print(
        f"Подозрительных вариантов: "
        f"{len(suspicious_rows)}"
    )
    print(
        f"Смешанная кир/лат:      "
        f"{len(mixed_rows)}"
    )

    print()
    print("Отчёт:")
    print(OUTPUT_CSV)

    print()
    print("=" * 80)
    print("ТОП ПОДОЗРИТЕЛЬНЫХ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    for row in suspicious_rows[:50]:

        print()

        print(
            f"{row['text']!r}"
            f" | count={row['count']}"
            f" | avg={row['avg_confidence'] or '?'}"
            f" | min={row['min_confidence'] or '?'}"
        )

        print(
            f"Причины: {row['reasons']}"
        )

        if row["raw_variants"]:
            print(
                f"RAW: {row['raw_variants']}"
            )

        if row["example_file"]:
            print(
                f"Пример: {row['example_file']}"
            )

    print()
    print("=" * 80)
    print("САМЫЕ ЧАСТЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)

    frequent = sorted(
        analysis_rows,
        key=lambda row: -row["count"]
    )

    for row in frequent[:50]:
        print(
            f"{row['count']:>5}x  "
            f"{row['text']!r}"
        )


if __name__ == "__main__":
    main()