from pathlib import Path
import time

import cv2

from src.detectors.stamp_detector import StampDetector


INPUT_DIR = Path(
    "output/drawing/album_dukmasova"
)

OUTPUT_DIR = Path(
    "output/stamp_batch"
)


def main():
    images = sorted(INPUT_DIR.glob("*.png"))

    if not images:
        print(f"❌ Изображения не найдены: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []
    start_time = time.time()

    print("\n🔍 Массовый тест детектора штампа")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🖼️ Найдено изображений: {len(images)}\n")

    for index, image_path in enumerate(images, start=1):
        try:
            detector = StampDetector(image_path)
            detector.load()
            detector.preprocess()
            detector.detect_lines()

            bounds = detector.find_stamp_bounds()

            if bounds is None:
                failed.append(image_path.name)
                print(
                    f"[{index:02}/{len(images)}] "
                    f"❌ {image_path.name}"
                )
                continue

            x_left, y_top, x_right, y_bottom = bounds

            stamp = detector.image[
                y_top:y_bottom + 1,
                x_left:x_right + 1
            ]

            debug = detector.image.copy()

            cv2.rectangle(
                debug,
                (x_left, y_top),
                (x_right, y_bottom),
                (0, 0, 255),
                4
            )

            cv2.imwrite(
                str(OUTPUT_DIR / f"{image_path.stem}_detected.png"),
                debug
            )

            cv2.imwrite(
                str(OUTPUT_DIR / f"{image_path.stem}_stamp.png"),
                stamp
            )

            success += 1

            print(
                f"[{index:02}/{len(images)}] "
                f"✅ {image_path.name} | "
                f"{stamp.shape[1]}×{stamp.shape[0]}"
            )

        except Exception as error:
            failed.append(image_path.name)

            print(
                f"[{index:02}/{len(images)}] "
                f"💥 {image_path.name} | {error}"
            )

    elapsed = time.time() - start_time

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎉 Тест завершён")
    print(f"🖼️ Проверено: {len(images)}")
    print(f"✅ Найден штамп: {success}")
    print(f"❌ Не найден: {len(failed)}")
    print(f"⏱️ Время: {elapsed:.1f} сек.")
    print(f"📁 Результаты: {OUTPUT_DIR}")

    if failed:
        print("\n⚠️ Проблемные изображения:")

        for name in failed:
            print(f"   • {name}")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()