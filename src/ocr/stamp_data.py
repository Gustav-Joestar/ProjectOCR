from dataclasses import dataclass, asdict


@dataclass
class StampData:
    """
    Структурированные данные основной надписи чертежа.

    "-" означает, что поле предусмотрено структурой штампа,
    но его значение не заполнено или не было распознано.
    """

    designation: str = "-"
    name: str = "-"
    material: str = "-"

    letter: str = "-"
    mass: str = "-"
    scale: str = "-"

    sheet: str = "-"
    sheet_count: str = "-"

    developer: str = "-"
    checker: str = "-"
    technical_controller: str = "-"
    norm_controller: str = "-"
    approved_by: str = "-"

    document_number: str = "-"
    signature: str = "-"
    date: str = "-"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)