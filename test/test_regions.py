from pathlib import Path

import cv2

from region_detector import RegionDetector

PROJECT_DIR = Path(
    r"C:\Users\andre\Documents\Diplom\ProjectOCR"
)

INPUT_ROOT = PROJECT_DIR / "data" / "test"

OUTPUT_ROOT = PROJECT_DIR / "output" / "test"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

datasets = sorted(

    d for d in INPUT_ROOT.iterdir()

    if d.is_dir()

)

print()

print("=" * 60)

print("TEST DATASETS")

print("=" * 60)

print()

# ---------------------------------------------------------

for dataset in datasets:

    print()

    print(f"[{dataset.name}]")

    output_dir = OUTPUT_ROOT / dataset.name

    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    images = sorted(

        list(dataset.glob("*.png"))

    )

    drawings_total = 0

    for image_path in images:

        print()

        print(f"  {image_path.name}")

        detector = RegionDetector(

            image_path

        )

        detector.run()

        dataset_name = dataset.name.replace("pages_", "")

        detector.save_projection_debug(
            output_dir /
            f"projection_{dataset_name}_{image_path.stem}.png"
        )

        dataset_name = dataset.name.replace("pages_", "")

        detector.save_mask(

            output_dir /

            f"mask_{dataset_name}_{image_path.stem}.png"

        )

        """detector.save_regions(

            output_dir /

            f"regions_{dataset_name}_{image_path.stem}.png"

        )

        regions = detector.get_regions()

        drawings_total += len(regions)

        print(

            f"    Drawings: {len(regions)}"

        )"""

    print()

    print(

        f"Dataset total: {drawings_total}"

    )

print()

print("=" * 60)

print("DONE")

print("=" * 60)