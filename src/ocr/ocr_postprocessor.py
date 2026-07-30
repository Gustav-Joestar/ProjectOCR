from difflib import SequenceMatcher
import re


# Стандартные подписи основной надписи.
# Это не значения конкретного чертежа, а фиксированные названия полей.
STAMP_LABELS = [
    "Изм.",
    "Лист",
    "№ докум.",
    "Подп.",
    "Дата",
    "Лит.",
    "Масса",
    "Масштаб",
    "Разраб.",
    "Пров.",
    "Т. контр.",
    "Н. контр.",
    "Утв.",
]


# Латинские символы, которые OCR может подставлять
# вместо визуально похожих кириллических.
LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",

    "a": "а",
    "c": "с",
    "e": "е",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
})


def normalize_spaces(text: str) -> str:
    """
    Нормализует пробелы и переносы строк.
    """

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_mixed_alphabet(text: str) -> str:
    """
    Исправляет визуально похожие латинские символы
    на кириллические.

    Пример:
        Macca -> Масса
        Yтв.  -> Утв.
    """

    return text.translate(
        LATIN_TO_CYRILLIC
    )


def normalize_label_for_matching(text: str) -> str:
    """
    Подготавливает подпись для fuzzy matching.
    """

    text = normalize_spaces(text)
    text = normalize_mixed_alphabet(text)

    text = text.lower()

    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def correct_label(
    text: str,
    threshold: float = 0.65
) -> str:
    """
    Ищет наиболее похожую стандартную подпись штампа.

    Если сходство недостаточное, исходный текст
    возвращается без изменений.
    """

    normalized = normalize_label_for_matching(
        text
    )

    best_label = None
    best_score = 0.0

    for label in STAMP_LABELS:

        label_normalized = (
            normalize_label_for_matching(
                label
            )
        )

        score = similarity(
            normalized,
            label_normalized
        )

        if score > best_score:
            best_score = score
            best_label = label

    if (
        best_label is not None
        and best_score >= threshold
    ):
        return best_label

    return text

def correct_sheet(text: str) -> str:
    """
    Нормализует поле количества листов.

    Например:
        Листов1  -> Листов 1
        Листов  3 -> Листов 3
    """

    text = normalize_spaces(text)
    text = normalize_mixed_alphabet(text)

    text = re.sub(
        r"(?i)\b(листов)\s*(\d+)\b",
        r"\1 \2",
        text
    )

    return text

def correct_material(text: str) -> str:
    """
    Нормализует распространённые OCR-ошибки
    в записи материала и обозначения ГОСТ.
    """

    text = normalize_spaces(text)

    # Исправляем смешение визуально похожих
    # латинских/кириллических символов.
    text = normalize_mixed_alphabet(text)

    # OCR может спутать первую Г в ГОСТ,
    # при этом ОСТ и номер стандарта распознаются нормально.
    text = re.sub(
        r"\b[FГ]ОСТ(?=\s*\d)",
        "ГОСТ",
        text,
        flags=re.IGNORECASE
    )

    return text

def correct_sheet(text: str) -> str:
    """
    Нормализация поля количества листов.

    Листов1   -> Листов 1
    Листов  1 -> Листов 1
    """

    text = normalize_spaces(text)
    text = normalize_mixed_alphabet(text)

    text = re.sub(
        r"(?i)\b(листов)\s*(\d+)\b",
        r"\1 \2",
        text
    )

    return text


    def correct_material(text: str) -> str:
        """
        Нормализация записи материала и ГОСТ.
        """

        text = normalize_spaces(text)
        text = normalize_mixed_alphabet(text)

        # После normalize_mixed_alphabet:
        # FОCТ 1435-99 -> FОСТ 1435-99
        #
        # F исправляем только в контексте слова ГОСТ,
        # а не глобально во всём техническом тексте.
        text = re.sub(
            r"\bFОСТ(?=\s*\d)",
            "ГОСТ",
            text,
            flags=re.IGNORECASE
        )

        return text

def postprocess_ocr(
    text: str,
    field_type: str
) -> str:

    text = normalize_spaces(text)

    if field_type == "label":
        return correct_label(text)

    if field_type == "sheet":
        return correct_sheet(text)

    if field_type == "material":
        return correct_material(text)

    return text