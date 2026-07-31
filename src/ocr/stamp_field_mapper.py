from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Iterable

from src.ocr.stamp_data import StampData


@dataclass(frozen=True)
class StampCell:
    """
    OCR-ячейка штампа с абсолютными и нормализованными координатами.
    """

    index: int
    text: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    nx1: float
    ny1: float
    nx2: float
    ny2: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def normalized_width(self) -> float:
        return self.nx2 - self.nx1

    @property
    def normalized_height(self) -> float:
        return self.ny2 - self.ny1

    @property
    def normalized_center_x(self) -> float:
        return (self.nx1 + self.nx2) / 2

    @property
    def normalized_center_y(self) -> float:
        return (self.ny1 + self.ny2) / 2


class StampFieldMapper:
    """
    Определяет смысловые поля основной надписи
    по OCR-тексту и геометрии ячеек.
    """

    def __init__(self, stamp_width: int, stamp_height: int):
        if stamp_width <= 0:
            raise ValueError("stamp_width must be greater than 0")

        if stamp_height <= 0:
            raise ValueError("stamp_height must be greater than 0")

        self.stamp_width = stamp_width
        self.stamp_height = stamp_height

    def prepare_cell(self, cell: dict) -> StampCell:
        """
        Преобразует одну OCR-ячейку в StampCell.
        """

        index = int(cell["index"])

        x1, y1, x2, y2 = cell["bounds"]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounds for cell {index}: "
                f"{(x1, y1, x2, y2)}"
            )

        text = str(cell.get("text", "") or "").strip()

        confidence = cell.get("confidence", 0.0)

        if confidence is None:
            confidence = 0.0

        confidence = float(confidence)

        return StampCell(
            index=index,
            text=text,
            confidence=confidence,

            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,

            nx1=x1 / self.stamp_width,
            ny1=y1 / self.stamp_height,
            nx2=x2 / self.stamp_width,
            ny2=y2 / self.stamp_height,
        )

    def prepare_cells(
        self,
        cells: Iterable[dict],
        include_empty: bool = False,
    ) -> list[StampCell]:
        """
        Подготавливает набор OCR-ячеек.

        По умолчанию ячейки без текста исключаются.
        """

        prepared = []

        for cell in cells:
            prepared_cell = self.prepare_cell(cell)

            if not include_empty and not prepared_cell.text:
                continue

            prepared.append(prepared_cell)

        return prepared

    def map(
        self,
        cells: Iterable[dict] | Iterable[StampCell],
    ) -> StampData:
        """
        Преобразует OCR-ячейки штампа в структурированные данные.

        Первая версия mapper:
        - designation
        - name
        - material
        - scale
        - sheet_count

        Остальные поля пока остаются "-".
        """

        cells = list(cells)

        if not cells:
            return StampData()

        if isinstance(cells[0], StampCell):
            prepared = cells
        else:
            prepared = self.prepare_cells(
                cells,
                include_empty=False,
            )

        designation = self._find_designation(prepared)
        name = self._find_name(prepared)
        material = self._find_material(prepared)
        scale = self._find_scale(prepared)
        sheet_count = self._find_sheet_count(prepared)

        return StampData(
            designation=designation,
            name=name,
            material=material,
            scale=scale,
            sheet_count=sheet_count,
        )

    # ------------------------------------------------------------------
    # DESIGNATION
    # ------------------------------------------------------------------

    def _find_designation(
        self,
        cells: list[StampCell],
    ) -> str:
        """
        Обозначение детали.

        В стандартной основной надписи находится
        в крупной верхней правой области.
        """

        candidates = []

        for cell in cells:
            cx = cell.normalized_center_x
            cy = cell.normalized_center_y

            if not (
                0.50 <= cx <= 0.85
                and 0.03 <= cy <= 0.25
            ):
                continue

            if cell.normalized_width < 0.25:
                continue

            if not self._looks_like_designation(cell.text):
                continue

            candidates.append(cell)

        return self._best_text(candidates)

    # ------------------------------------------------------------------
    # NAME
    # ------------------------------------------------------------------

    def _find_name(
        self,
        cells: list[StampCell],
    ) -> str:
        """
        Наименование детали.

        Ищем крупную центральную область основной надписи.
        """

        candidates = []

        for cell in cells:
            cx = cell.normalized_center_x
            cy = cell.normalized_center_y

            if not (
                0.42 <= cx <= 0.68
                and 0.38 <= cy <= 0.62
            ):
                continue

            if cell.normalized_width < 0.20:
                continue

            if self._is_anchor(cell.text):
                continue

            if self._looks_like_material(cell.text):
                continue

            if self._looks_like_designation(cell.text):
                continue

            candidates.append(cell)

        return self._best_text(candidates)

    # ------------------------------------------------------------------
    # MATERIAL
    # ------------------------------------------------------------------

    def _find_material(
        self,
        cells: list[StampCell],
    ) -> str:
        """
        Материал детали.

        Используем одновременно положение и характер текста.
        """

        candidates = []

        for cell in cells:
            cx = cell.normalized_center_x
            cy = cell.normalized_center_y

            if not (
                0.42 <= cx <= 0.68
                and 0.72 <= cy <= 0.98
            ):
                continue

            if not self._looks_like_material(cell.text):
                continue

            candidates.append(cell)

        return self._best_text(candidates)

    # ------------------------------------------------------------------
    # SCALE
    # ------------------------------------------------------------------

    def _find_scale(
        self,
        cells: list[StampCell],
    ) -> str:
        """
        Масштаб.

        Основной сигнал:
        - правая часть штампа;
        - под anchor "Масштаб";
        - формат числа вида 1:1, 2:1, 1:2, 2,5:1 и т.д.
        """

        scale_anchor = self._find_anchor(
            cells,
            "Масштаб",
        )

        candidates = []

        for cell in cells:
            if not self._looks_like_scale(cell.text):
                continue

            cx = cell.normalized_center_x
            cy = cell.normalized_center_y

            if not (
                0.88 <= cx <= 1.01
                and 0.35 <= cy <= 0.65
            ):
                continue

            if scale_anchor is not None:
                if cy <= scale_anchor.normalized_center_y:
                    continue

                horizontal_distance = abs(
                    cx - scale_anchor.normalized_center_x
                )

                if horizontal_distance > 0.08:
                    continue

            candidates.append(cell)

        return self._best_text(candidates)

    # ------------------------------------------------------------------
    # SHEET COUNT
    # ------------------------------------------------------------------

    def _find_sheet_count(
        self,
        cells: list[StampCell],
    ) -> str:
        """
        Количество листов.

        OCR может вернуть:

            Листов 1

        одной строкой, поэтому сначала извлекаем число
        непосредственно из текста.

        Позже сюда можно добавить случай, когда anchor
        "Листов" и значение находятся в разных ячейках.
        """

        candidates = []

        pattern = re.compile(
            r"^Листов\s*([0-9]+)$",
            re.IGNORECASE,
        )

        for cell in cells:
            match = pattern.match(cell.text)

            if not match:
                continue

            cx = cell.normalized_center_x
            cy = cell.normalized_center_y

            if not (
                0.80 <= cx <= 1.01
                and 0.58 <= cy <= 0.78
            ):
                continue

            candidates.append(
                (
                    cell,
                    match.group(1),
                )
            )

        if not candidates:
            return "-"

        candidates.sort(
            key=lambda item: item[0].confidence,
            reverse=True,
        )

        return candidates[0][1]

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _find_anchor(
        self,
        cells: list[StampCell],
        text: str,
    ) -> StampCell | None:
        """
        Находит наиболее уверенно распознанный anchor
        с указанным текстом.
        """

        candidates = [
            cell
            for cell in cells
            if cell.text == text
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda cell: cell.confidence,
        )

    def _best_text(
        self,
        candidates: list[StampCell],
    ) -> str:
        """
        Возвращает текст наиболее уверенного кандидата.
        """

        if not candidates:
            return "-"

        best = max(
            candidates,
            key=lambda cell: cell.confidence,
        )

        return best.text or "-"

    @staticmethod
    def _looks_like_designation(text: str) -> bool:
        """
        Грубая проверка обозначения детали.

        Не привязываемся к конкретному
        00-000.06.01.01.07.
        """

        text = text.strip()

        if len(text) < 5:
            return False

        has_digit = any(
            char.isdigit()
            for char in text
        )

        has_separator = any(
            char in ".-/"
            for char in text
        )

        return has_digit and has_separator

    @staticmethod
    def _looks_like_scale(text: str) -> bool:
        """
        Примеры:
            1:1
            1:2
            2:1
            2,5:1
            5:1
        """

        return bool(
            re.fullmatch(
                r"\d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?",
                text.strip(),
            )
        )

    @staticmethod
    def _looks_like_material(text: str) -> bool:
        """
        Пока используем только общие признаки материалов,
        встречающиеся в нашем наборе.

        Это не словарь всех возможных материалов.
        """

        normalized = text.upper()

        material_markers = (
            "ГОСТ",
            "СТАЛЬ",
            "ОТЛИВКА",
        )

        return any(
            marker in normalized
            for marker in material_markers
        )

    @staticmethod
    def _is_anchor(text: str) -> bool:
        anchors = {
            "Изм.",
            "Лист",
            "№ докум.",
            "Подп.",
            "Дата",
            "Разраб.",
            "Пров.",
            "Т. контр.",
            "Н. контр.",
            "Утв.",
            "Лит.",
            "Масса",
            "Масштаб",
            "Листов",
        }

        return text in anchors