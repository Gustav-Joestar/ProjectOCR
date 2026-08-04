from pathlib import Path

from hough_detector import HoughDetector

INPUT_DIR = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR\output\pages\Axarin"
)

OUTPUT_DIR = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR\output\test\hough"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for page in range(6, 11):

    img = INPUT_DIR / f"page_{page:03d}.png"

    if not img.exists():
        continue

    print(f"[{page:03d}] {img.name}")

    detector = HoughDetector(img)

    detector.load()

    detector.preprocess()

    detector.detect_lines()

    print(f"    Raw: {len(detector.lines)}")

    detector.keep_horizontal_vertical()

    print(f"    H/V: {len(detector.lines)}")

    detector.save_debug(
        OUTPUT_DIR / f"hough_{page:03d}.png"
    )

print("Done.")