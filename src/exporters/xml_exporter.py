from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


class XMLExporter:
    """
    Экспорт структурированных данных чертежа в XML.

    Сейчас экспортируется основная надпись (stamp).
    Позже сюда можно добавить geometry и другие разделы.
    """

    def export(
        self,
        stamp_data,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        root = ET.Element("drawing")

        stamp_element = ET.SubElement(
            root,
            "stamp",
        )

        self._write_stamp(
            stamp_element,
            stamp_data,
        )

        tree = ET.ElementTree(root)

        ET.indent(
            tree,
            space="    ",
        )

        tree.write(
            output_path,
            encoding="utf-8",
            xml_declaration=True,
        )

        return output_path

    @staticmethod
    def _write_stamp(
        parent: ET.Element,
        stamp_data,
    ) -> None:
        data = stamp_data.to_dict()

        for field, value in data.items():
            element = ET.SubElement(
                parent,
                field,
            )

            element.text = (
                str(value)
                if value not in (None, "")
                else "-"
            )