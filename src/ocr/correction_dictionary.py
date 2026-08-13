import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DICTIONARY_PATH = (
    PROJECT_ROOT
    / "dictionaries"
    / "ocr_corrections.csv"
)


class OCRCorrectionDictionary:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_DICTIONARY_PATH

        self.exact_rules = {}
        self.substring_rules = []

        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(
                f"Не найден словарь OCR: {self.path}"
            )

        with self.path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            required = {
                "wrong",
                "correct",
                "type",
                "enabled",
            }

            if not reader.fieldnames:
                raise ValueError("CSV-словарь не содержит заголовка.")

            missing = required - set(reader.fieldnames)

            if missing:
                raise ValueError(
                    "В словаре отсутствуют колонки: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:
                wrong = row["wrong"].strip()
                correct = row["correct"].strip()
                rule_type = row["type"].strip().lower()
                enabled = row["enabled"].strip().lower()

                if enabled not in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    continue

                if not wrong:
                    continue

                if rule_type == "exact":
                    self.exact_rules[wrong] = correct

                elif rule_type == "substring":
                    self.substring_rules.append(
                        (wrong, correct)
                    )

                else:
                    raise ValueError(
                        f"Неизвестный тип правила: {rule_type!r}"
                    )

        # Более длинные варианты заменяем первыми.
        self.substring_rules.sort(
            key=lambda item: len(item[0]),
            reverse=True
        )

    def correct(self, text: str) -> str:
        if not text:
            return text

        result = text.strip()

        # 1. Exact по исходной нормализованной строке.
        if result in self.exact_rules:
            return self.exact_rules[result]

        # 2. Общие substring-коррекции.
        for wrong, correct in self.substring_rules:
            result = result.replace(
                wrong,
                correct
            )

        # 3. После substring строка могла стать
        #    подходящей под exact-правило.
        if result in self.exact_rules:
            return self.exact_rules[result]

        return result