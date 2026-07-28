from pathlib import Path

from src.detectors.drawing_detector import DrawingDetector


IMAGES_DIR = Path("data/images")
OUTPUT_DIR = Path("output/detection")

FIRST_PAGE = 1
LAST_PAGE = 47


def analyze_page(page_number: int):
    image_path = IMAGES_DIR / f"page_{page_number:03d}.png"
    output_path = OUTPUT_DIR / f"page_{page_number:03d}.png"

    detector = DrawingDetector(str(image_path))

    detector.load()
    detector.preprocess()

    detector.extract_horizontal_lines()
    detector.extract_vertical_lines()
    detector.combine_lines()
    detector.connect_lines()

    drawings = detector.find_drawings()

    detector.save_detection_preview(str(output_path))

    return drawings, output_path


def main():
    pages = range(FIRST_PAGE, LAST_PAGE + 1)
    total_pages = LAST_PAGE - FIRST_PAGE + 1
    total_drawings = 0

    print()
    print("🚀 Запуск детекции чертежей")
    print(f"📄 Страниц для обработки: {total_pages}")
    print()

    for current, page_number in enumerate(pages, start=1):
        print(
            f"[{current:03d}/{total_pages:03d}] "
            f"🔍 Страница {page_number:03d}..."
        )

        drawings, output_path = analyze_page(page_number)

        drawing_count = len(drawings)
        total_drawings += drawing_count

        print(
            f"          ✅ Найдено чертежей: {drawing_count}"
        )
        print(
            f"          💾 Сохранено: {output_path}"
        )
        print()

    print("=" * 60)
    print("🎉 Детекция завершена")
    print(f"📄 Обработано страниц: {total_pages}")
    print(f"🖼️ Найдено чертежей: {total_drawings}")
    print(f"📁 Результаты: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()