from pathlib import Path
from time import perf_counter
import shutil

import cv2

from .progress import PipelineProgress

from src.loaders.pdf_loader import PDFLoader
from src.detectors.drawing_detector import DrawingDetector
from src.detectors.stamp_detector import StampDetector
from src.segmentation.stamp_segmenter import StampSegmenter
from src.ocr.stamp_ocr import StampOCR
from src.ocr.stamp_field_mapper import StampFieldMapper
from src.exporters.xml_exporter import XMLExporter


PAGES_DIR = Path("output/pages")
DETECTION_DIR = Path("output/detection")
DRAWING_DIR = Path("output/drawing")
STAMP_DIR = Path("output/stamp")
XML_DIR = Path("output/xml")


def format_time(seconds: float) -> str:
    seconds = max(
        0,
        int(seconds),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours > 0:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def clear_pdf_output(
    pdf_name: str,
):
    """
    Удаляет результаты предыдущей обработки PDF.
    """

    output_dirs = (
        PAGES_DIR / pdf_name,
        DETECTION_DIR / pdf_name,
        DRAWING_DIR / pdf_name,
        STAMP_DIR / pdf_name,
        XML_DIR / pdf_name,
    )

    for output_dir in output_dirs:
        if output_dir.exists():
            shutil.rmtree(
                output_dir
            )


def extract_pages(
    loader: PDFLoader,
    pages_dir: Path,
    progress: PipelineProgress,
) -> list[Path]:
    """
    Извлекает все страницы PDF.
    """

    total_pages = loader.page_count

    progress.start_stage(
        "pages",
        total_pages,
    )

    page_paths = []

    for page_index in range(
        total_pages
    ):
        image_path = loader.save_page(
            page_number=page_index,
            output_dir=pages_dir,
        )

        page_paths.append(
            Path(image_path)
        )

        progress.update(
            "pages",
            page_index + 1,
            total_pages,
        )

    progress.finish_stage(
        "pages",
        len(page_paths),
    )

    return page_paths


def detect_drawings(
    page_paths: list[Path],
    detection_dir: Path,
    drawing_dir: Path,
    progress: PipelineProgress,
) -> list[Path]:
    """
    Находит и сохраняет чертежи
    на всех изображениях страниц.
    """

    total_pages = len(page_paths)

    progress.start_stage(
        "drawings",
        total_pages,
    )

    drawing_paths = []

    for page_index, image_path in enumerate(
        page_paths,
        start=1,
    ):
        detector = DrawingDetector(
            image_path
        )

        detector.load()
        detector.preprocess()
        detector.extract_horizontal_lines()
        detector.extract_vertical_lines()
        detector.combine_lines()
        detector.connect_lines()

        detector.find_drawings()

        preview_path = (
            detection_dir
            / f"page_{page_index:03d}.png"
        )

        detector.save_detection_preview(
            preview_path
        )

        page_drawings = (
            detector.save_drawings(
                drawing_dir,
                page_index,
            )
        )

        drawing_paths.extend(
            Path(path)
            for path in page_drawings
        )

        progress.update(
            "drawings",
            page_index,
            total_pages,
        )

    progress.finish_stage(
        "drawings",
        len(drawing_paths),
    )

    return drawing_paths


def extract_stamp(
    drawing_path: Path,
    stamp_dir: Path,
) -> Path | None:
    """
    Находит и сохраняет основную надпись.

    StampDetector сначала использует основной
    алгоритм, а при его неудаче — fallback.
    """

    detector = StampDetector(
        drawing_path
    )

    detector.load()
    detector.preprocess()
    detector.detect_lines()

    stamp_image = (
        detector.crop_stamp()
    )

    if (
        stamp_image is None
        or stamp_image.size == 0
    ):
        return None

    stamp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp_path = (
        stamp_dir
        / f"{drawing_path.stem}_stamp.png"
    )

    if not cv2.imwrite(
        str(stamp_path),
        stamp_image,
    ):
        raise RuntimeError(
            "Не удалось сохранить штамп: "
            f"{stamp_path}"
        )

    return stamp_path


def recognize_stamp(
    stamp_path: Path,
):
    """
    Сегментация основной надписи + OCR + mapping.

    Возвращает StampData.
    """

    segmenter = StampSegmenter(
        stamp_path
    )

    segmenter.load()
    segmenter.preprocess()
    segmenter.detect_grid()

    cells = segmenter.find_cells()

    ocr = StampOCR(
        segmenter.image,
        cells,
    )

    try:
        ocr_results = (
            ocr.recognize()
        )
    finally:
        ocr.close()

    height, width = (
        segmenter.image.shape[:2]
    )

    mapper = StampFieldMapper(
        stamp_width=width,
        stamp_height=height,
    )

    return mapper.map(
        ocr_results
    )


def process_stamps(
    drawing_paths: list[Path],
    stamp_dir: Path,
    progress: PipelineProgress,
):
    """
    Для каждого чертежа:
    stamp detection -> segmentation -> OCR -> mapping.

    Возвращает список:
        [(drawing_path, stamp_data), ...]
    """

    total = len(
        drawing_paths
    )

    progress.start_stage(
        "ocr",
        total,
    )

    results = []

    for index, drawing_path in enumerate(
        drawing_paths,
        start=1,
    ):
        stamp_path = extract_stamp(
            drawing_path,
            stamp_dir,
        )

        if stamp_path is None:
            progress.skip_drawing()

            progress.update(
                "ocr",
                index,
                total,
            )

            continue

        stamp_data = recognize_stamp(
            stamp_path
        )

        results.append(
            (
                drawing_path,
                stamp_data,
            )
        )

        progress.update(
            "ocr",
            index,
            total,
        )

    progress.finish_stage(
        "ocr",
        len(results),
    )

    return results


def export_xml_files(
    stamp_results,
    xml_dir: Path,
    progress: PipelineProgress,
) -> list[Path]:
    """
    Формирует XML для успешно
    распознанных чертежей.
    """

    total = len(
        stamp_results
    )

    progress.start_stage(
        "xml",
        total,
    )

    xml_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exporter = XMLExporter()

    xml_paths = []

    for index, (
        drawing_path,
        stamp_data,
    ) in enumerate(
        stamp_results,
        start=1,
    ):
        xml_path = (
            xml_dir
            / f"{drawing_path.stem}.xml"
        )

        exporter.export(
            stamp_data=stamp_data,
            output_path=xml_path,
        )

        xml_paths.append(
            xml_path
        )

        progress.update(
            "xml",
            index,
            total,
        )

    progress.finish_stage(
        "xml",
        len(xml_paths),
    )

    return xml_paths


def process_pdf(
    pdf_path: Path,
):
    """
    Полный pipeline одного PDF:

    PDF
      -> страницы
      -> поиск чертежей
      -> сохранение чертежей
      -> поиск штампов
      -> сегментация
      -> OCR
      -> mapping
      -> XML
    """

    pdf_name = (
        pdf_path.stem
    )

    pages_dir = (
        PAGES_DIR
        / pdf_name
    )

    detection_dir = (
        DETECTION_DIR
        / pdf_name
    )

    drawing_dir = (
        DRAWING_DIR
        / pdf_name
    )

    stamp_dir = (
        STAMP_DIR
        / pdf_name
    )

    xml_dir = (
        XML_DIR
        / pdf_name
    )

    clear_pdf_output(
        pdf_name
    )

    print()
    print(
        f"📘 Обработка: "
        f"{pdf_path.name}"
    )
    print()

    start_time = (
        perf_counter()
    )

    progress = (
        PipelineProgress()
    )

    # ---------------------------------------------------------
    # 1. PDF -> страницы
    # ---------------------------------------------------------

    with PDFLoader(
        pdf_path
    ) as loader:
        total_pages = (
            loader.page_count
        )

        print(
            f"📄 Страниц: "
            f"{total_pages}"
        )
        print()

        print(
            "────────────────────────────────────────────────────────────"
        )
        print()

        page_paths = extract_pages(
            loader=loader,
            pages_dir=pages_dir,
            progress=progress,
        )

    # ---------------------------------------------------------
    # 2. Страницы -> чертежи
    # ---------------------------------------------------------

    drawing_paths = detect_drawings(
        page_paths=page_paths,
        detection_dir=detection_dir,
        drawing_dir=drawing_dir,
        progress=progress,
    )

    # ---------------------------------------------------------
    # 3. Чертежи -> StampData
    # ---------------------------------------------------------

    stamp_results = process_stamps(
        drawing_paths=drawing_paths,
        stamp_dir=stamp_dir,
        progress=progress,
    )

    # ---------------------------------------------------------
    # 4. StampData -> XML
    # ---------------------------------------------------------

    xml_paths = export_xml_files(
        stamp_results=stamp_results,
        xml_dir=xml_dir,
        progress=progress,
    )

    progress.finish()

    elapsed = (
        perf_counter()
        - start_time
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"✅ {pdf_path.name} обработан"
    )

    print(
        f"📄 Страниц: "
        f"{len(page_paths)}"
    )

    print(
        f"✂️ Чертежей: "
        f"{len(drawing_paths)}"
    )

    print(
        f"🔤 Штампов: "
        f"{len(stamp_results)}"
    )

    print(
        f"📄 XML: "
        f"{len(xml_paths)}"
    )

    if progress.skipped_drawings:
        print(
            f"⚠️ Пропущено: "
            f"{progress.skipped_drawings}"
        )

    print(
        f"⏱️ Время: "
        f"{format_time(elapsed)}"
    )

    print(
        "────────────────────────────────────────────────────────────"
    )
    print()

    return (
        len(page_paths),
        len(drawing_paths),
    )