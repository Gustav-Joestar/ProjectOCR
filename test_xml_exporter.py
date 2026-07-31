from pathlib import Path

from src.exporters.xml_exporter import XMLExporter
from src.ocr.stamp_field_mapper import StampData


PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "test"
    / "page_001_drawing_001.xml"
)


def main():
    stamp_data = StampData(
        designation="00-000.06.01.01.07",
        name="Опора",
        material="Сталь У8А ГОСТ 1435-99",
        letter="-",
        mass="-",
        scale="5:1",
        sheet="-",
        sheet_count="1",
        developer="-",
        checker="-",
        technical_controller="-",
        norm_controller="-",
        approved_by="-",
        document_number="-",
        signature="-",
        date="-",
    )

    exporter = XMLExporter()

    output_path = exporter.export(
        stamp_data=stamp_data,
        output_path=OUTPUT_PATH,
    )

    print("=" * 70)
    print("XML EXPORT TEST")
    print("=" * 70)
    print()
    print(f"Создан: {output_path}")
    print()
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()