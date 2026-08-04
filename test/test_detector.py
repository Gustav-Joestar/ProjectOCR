from pathlib import Path

from drawing_detector_2 import DrawingDetector

ROOT = Path(r"C:\Users\andre\Documents\Diplom\ProjectOCR")

IMAGE_PATH = ROOT / "data" / "test" / "pages_axarin" / "page_129.png"
OUTPUT_DIR = ROOT / "output" / "debug"


def main():
    print("=" * 70)
    print("DRAWING DETECTOR DEBUG")
    print("=" * 70)
    print(f"Image: {IMAGE_PATH}")

    detector = DrawingDetector(str(IMAGE_PATH))

    print("[1] load()")
    detector.load()

    print("[2] preprocess()")
    detector.preprocess()

    print("[3] extract_horizontal_lines()")
    detector.extract_horizontal_lines()

    print("[4] extract_vertical_lines()")
    detector.extract_vertical_lines()

    print("[5] combine_lines()")
    detector.combine_lines()

    print("[6] connect_lines()")
    detector.connect_lines()

    print("[7] find_drawings()")
    detector.find_drawings()

    print(f"Found drawings: {len(detector.drawings)}")

    detector.save_detection_preview(
        OUTPUT_DIR / "page_129_preview.png"
    )

    detector.save_drawings(
        OUTPUT_DIR,
        page_number=129,
    )

    print("Done.")


if __name__ == "__main__":
    main()