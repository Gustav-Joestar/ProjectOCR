from time import perf_counter

from rich.console import Console
from rich.live import Live
from rich.text import Text


class PipelineProgress:
    BAR_LENGTH = 30

    STAGE_WEIGHTS = {
        "extraction": 0.30,
        "ocr": 0.65,
        "xml": 0.05,
    }

    STAGE_NAMES = {
        "extraction": "Извлечение чертежей",
        "ocr": "Распознавание штампов",
        "xml": "Формирование XML",
    }

    STAGE_ORDER = (
        "extraction",
        "ocr",
        "xml",
    )

    def __init__(self):
        self.start_time = perf_counter()
        self.current_stage = None

        self.stage_progress = {
            stage: 0.0
            for stage in self.STAGE_ORDER
        }

        self.stage_counts = {
            stage: (0, 0)
            for stage in self.STAGE_ORDER
        }

        self.stage_started = {
            stage: None
            for stage in self.STAGE_ORDER
        }

        self.stage_times = {
            stage: None
            for stage in self.STAGE_ORDER
        }

        self.console = Console()

        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )

        self.live.start()

    @staticmethod
    def format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))

        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        return f"{minutes:02d}:{seconds:02d}"

    def start_stage(self, stage: str, total: int):
        self.current_stage = stage
        self.stage_started[stage] = perf_counter()
        self.stage_counts[stage] = (0, total)
        self.stage_progress[stage] = 0.0
        self._refresh()

    def update(self, stage: str, current: int, total: int):
        self.current_stage = stage
        self.stage_counts[stage] = (current, total)

        progress = (
            current / total
            if total > 0
            else 1.0
        )

        self.stage_progress[stage] = max(
            0.0,
            min(1.0, progress),
        )

        self._refresh()

    def finish_stage(
        self,
        stage: str,
        count: int | None = None,
    ):
        current, total = self.stage_counts[stage]

        if count is not None:
            current = count

        self.stage_counts[stage] = (
            current,
            total,
        )

        self.stage_progress[stage] = 1.0

        started = self.stage_started[stage]

        if started is not None:
            self.stage_times[stage] = (
                perf_counter() - started
            )

        self._refresh()

    def overall_progress(self) -> float:
        return sum(
            self.stage_progress[stage]
            * self.STAGE_WEIGHTS[stage]
            for stage in self.STAGE_ORDER
        )

    def estimate_remaining(self) -> float | None:
        progress = self.overall_progress()

        if progress <= 0:
            return None

        elapsed = (
            perf_counter()
            - self.start_time
        )

        estimated_total = (
            elapsed / progress
        )

        return max(
            0.0,
            estimated_total - elapsed,
        )

    def make_bar(self, progress: float) -> str:
        progress = max(
            0.0,
            min(1.0, progress),
        )

        filled = int(
            self.BAR_LENGTH * progress
        )

        return (
            "█" * filled
            + "░" * (
                self.BAR_LENGTH - filled
            )
        )

    def _stage_line(self, stage: str) -> str:
        name = self.STAGE_NAMES[stage]
        progress = self.stage_progress[stage]
        current, total = self.stage_counts[stage]

        if progress >= 1.0:
            elapsed = self.stage_times[stage]

            time_text = (
                ""
                if elapsed is None
                else " | " + self.format_time(elapsed)
            )

            return (
                f"✓ {name:<29}"
                f"{current}"
                f"{time_text}"
            )

        if stage == self.current_stage:
            count_text = (
                f"{current} / {total}"
                if total > 0
                else str(current)
            )

            return (
                f"⏳ {name:<28}"
                f"{count_text}"
            )

        return f"○ {name}"

    def _build_display(self) -> Text:
        overall = self.overall_progress()

        elapsed = (
            perf_counter()
            - self.start_time
        )

        remaining = self.estimate_remaining()

        remaining_text = (
            "--:--"
            if remaining is None
            else self.format_time(remaining)
        )

        bar = self.make_bar(overall)

        lines = [
            "ОБЩИЙ ПРОГРЕСС",
            (
                f"⏳ {bar} "
                f"{overall * 100:6.2f}% | "
                f"{self.format_time(elapsed)} / "
                f"{remaining_text}"
            ),
            "",
            "",
        ]

        for stage in self.STAGE_ORDER:
            lines.append(
                self._stage_line(stage)
            )

        return Text(
            "\n".join(lines)
        )

    def _refresh(self):
        self.live.update(
            self._build_display(),
            refresh=True,
        )

    def finish(self):
        self.current_stage = None
        self._refresh()
        self.live.stop()