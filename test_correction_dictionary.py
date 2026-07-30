from src.ocr.correction_dictionary import OCRCorrectionDictionary


dictionary = OCRCorrectionDictionary()

tests = [
    "Луст",
    "Масштад",
    "Macca",
    "Разрад.",
    "Yтв.",
    "№° докум.",
    "Сталь 45 FОСТ 1050-88",
    "Сталь У8А FОCТ 1435-99",
    "Г0СТ 380-2005",
    "Опора",
    "00-000.06.01.01.07",
]


for text in tests:
    corrected = dictionary.correct(text)

    print(
        f"{text!r:<35} -> {corrected!r}"
    )