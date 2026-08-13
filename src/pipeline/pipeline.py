from pathlib import Path
from time import perf_counter
import shutil

import cv2

from .progress import PipelineProgress

from src.loaders.pdf_loader import PDFLoader
from src.loaders.image_loader import ImageLoader
from src.detectors.yolo_detector import YOLODetector
from src.segmentation.stamp_segmenter import StampSegmenter
from src.ocr.stamp_ocr import StampOCR
from src.ocr.stamp_field_mapper import StampFieldMapper
from src.ocr.stamp_data import StampData
from src.exporters.xml_exporter import XMLExporter


OUTPUT_DIR = Path("output")

MODEL_PATH = Path("models/final_detect_model.pt")
CONFIDENCE_THRESHOLD = 0.85
PAGE_DPI = 600

def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"

def clear_output(name: str):
    output_dir = OUTPUT_DIR / name

    if output_dir.exists():
        shutil.rmtree(output_dir)

def process_page(
    image,
    page_index: int,
    pdf_dir: Path,
    detector: YOLODetector,
    is_pdf: bool = True,
):
    detections = detector.detect(image)

    drawings = detections["drawings"]
    stamps = detections["stamps"]

    if not drawings:
        return []

    if is_pdf:
        page_dir = (
            pdf_dir
            / f"{pdf_dir.name}_page_{page_index:03d}"
        )
    else:
        page_dir = pdf_dir

    results = []

    for drawing_index, drawing in enumerate(
        drawings,
        start=1,
    ):
        drawing_image = detector.crop(
            image,
            drawing["bbox"],
        )

        if drawing_image is None:
            continue

        if is_pdf:
            drawing_name = (
                f"{pdf_dir.name}_page_{page_index:03d}"
                f"_drawing_{drawing_index:03d}.png"
            )
        elif len(drawings) == 1:
            drawing_name = "drawing.png"
        else:
            drawing_name = (
                f"drawing_{drawing_index:03d}.png"
            )

        if not page_dir.exists():
            page_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        drawing_path = page_dir / drawing_name

        if not cv2.imwrite(
            str(drawing_path),
            drawing_image,
        ):
            raise RuntimeError(
                f"Не удалось сохранить чертёж: {drawing_path}"
            )

        stamp = detector.find_stamp_for_drawing(
            drawing["bbox"],
            stamps,
        )

        stamp_image = None

        if stamp is not None:
            stamp_image = detector.crop(
                image,
                stamp["bbox"],
            )

        results.append(
            {
                "page_index": page_index,
                "drawing_path": drawing_path,
                "stamp_image": stamp_image,
            }
        )

    return results

def recognize_stamp(
    stamp_image,
    recognizer=None,
):
    if stamp_image is None:
        return StampData()

    segmenter = StampSegmenter(stamp_image)

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    cells = segmenter.find_cells()

    ocr = StampOCR(
        segmenter.image,
        cells,
        recognizer=recognizer,
    )

    try:
        if recognizer is None:
            ocr.load_recognizer()

        ocr_results = ocr.recognize()
    finally:
        if recognizer is None:
            ocr.close()

    height, width = segmenter.image.shape[:2]

    mapper = StampFieldMapper(
        stamp_width=width,
        stamp_height=height,
    )

    return mapper.map(ocr_results)

def process_stamps(
    drawing_results,
    progress: PipelineProgress,
):
    total = len(drawing_results)

    progress.start_stage(
        "ocr",
        total,
    )

    results = []

    if total == 0:
        progress.finish_stage(
            "ocr",
            total,
        )
        return results

    ocr_engine = StampOCR(None, [])
    recognizer = ocr_engine.load_recognizer()

    try:
        for index, item in enumerate(
            drawing_results,
            start=1,
        ):
            stamp_data = recognize_stamp(
                item["stamp_image"],
                recognizer=recognizer,
            )

            results.append(
                {
                    "drawing_path": item["drawing_path"],
                    "stamp_data": stamp_data,
                }
            )

            item["stamp_image"] = None

            progress.update(
                "ocr",
                index,
                total,
            )
    finally:
        ocr_engine.close()

    progress.finish_stage(
        "ocr",
        total,
    )

    return results

def export_xml_files(
    stamp_results,
    progress: PipelineProgress,
):
    total = len(stamp_results)

    progress.start_stage(
        "xml",
        total,
    )

    exporter = XMLExporter()
    xml_paths = []

    for index, item in enumerate(
        stamp_results,
        start=1,
    ):
        drawing_path = item["drawing_path"]
        stamp_data = item["stamp_data"]

        xml_path = drawing_path.with_suffix(".xml")

        exporter.export(
            stamp_data=stamp_data,
            output_path=xml_path,
        )

        xml_paths.append(xml_path)

        progress.update(
            "xml",
            index,
            total,
        )

    progress.finish_stage(
        "xml",
        total,
    )

    return xml_paths

def _process_images(
    image_queue: list[Path],
    progress: PipelineProgress,
):
    total = len(image_queue)

    progress.start_stage(
        "extraction",
        total,
    )

    detector = YOLODetector(
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
    )

    drawing_results = []

    for index, image_path in enumerate(
        image_queue,
        start=1,
    ):
        image = ImageLoader(image_path).load()

        image_results = process_page(
            image=image,
            page_index=index,
            pdf_dir=OUTPUT_DIR / image_path.stem,
            detector=detector,
            is_pdf=False,
        )

        drawing_results.extend(image_results)

        del image

        progress.update(
            "extraction",
            index,
            total,
        )

    progress.finish_stage(
        "extraction",
        total,
    )

    return drawing_results

def process_image(image_path: Path):
    """Обрабатывает одно изображение напрямую через YOLO."""

    clear_output(image_path.stem)

    print()
    print(f"🖼️ Обработка: {image_path.name}")
    print()

    start_time = perf_counter()
    progress = PipelineProgress()

    drawing_results = _process_images(
        [image_path],
        progress,
    )

    stamp_results = process_stamps(
        drawing_results=drawing_results,
        progress=progress,
    )

    xml_paths = export_xml_files(
        stamp_results=stamp_results,
        progress=progress,
    )

    progress.finish()

    elapsed = perf_counter() - start_time

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ {image_path.name} обработано")
    print(f"✂️ Чертежей: {len(drawing_results)}")
    print(f"🔤 Штампов: {len(stamp_results)}")
    print(f"📄 XML: {len(xml_paths)}")
    print(f"⏱️ Время: {format_time(elapsed)}")
    print("────────────────────────────────────────────────────────────")
    print()

    return 1, len(drawing_results)

def process_pdf(pdf_path: Path):
    pdf_name = pdf_path.stem
    pdf_dir = OUTPUT_DIR / pdf_name

    clear_output(pdf_name)

    print()
    print(f"📘 Обработка: {pdf_path.name}")
    print()

    start_time = perf_counter()
    progress = PipelineProgress()

    drawing_results = []

    detector = YOLODetector(
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
    )

    with PDFLoader(pdf_path) as loader:
        total_pages = loader.page_count

        print(f"📄 Страниц: {total_pages}")
        print()

        print(
            "────────────────────────────────────────────────────────────"
        )
        print()

        progress.start_stage(
            "extraction",
            total_pages,
        )

        for page_index in range(total_pages):
            image = loader.render_page(
                page_number=page_index,
                dpi=PAGE_DPI,
            )

            page_results = process_page(
                image=image,
                page_index=page_index + 1,
                pdf_dir=pdf_dir,
                detector=detector,
                is_pdf=True,
            )

            drawing_results.extend(page_results)

            del image

            progress.update(
                "extraction",
                page_index + 1,
                total_pages,
            )

        progress.finish_stage(
            "extraction",
            total_pages,
        )

    stamp_results = process_stamps(
        drawing_results=drawing_results,
        progress=progress,
    )

    xml_paths = export_xml_files(
        stamp_results=stamp_results,
        progress=progress,
    )

    progress.finish()

    elapsed = perf_counter() - start_time

    drawing_count = len(drawing_results)
    stamp_count = sum(
        item["stamp_data"].designation != "-"
        or item["stamp_data"].name != "-"
        or item["stamp_data"].material != "-"
        or item["stamp_data"].scale != "-"
        or item["stamp_data"].sheet_count != "-"
        for item in stamp_results
    )

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ {pdf_path.name} обработан")
    print(f"📄 Страниц: {total_pages}")
    print(f"✂️ Чертежей: {drawing_count}")
    print(f"🔤 Штампов: {stamp_count}")
    print(f"📄 XML: {len(xml_paths)}")
    print(f"⏱️ Время: {format_time(elapsed)}")
    print("────────────────────────────────────────────────────────────")
    print()

    return total_pages, drawing_count