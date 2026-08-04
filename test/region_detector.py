from pathlib import Path

import cv2
import numpy as np

MIN_REGION_AREA = 20000

MIN_REGION_WIDTH = 1200
MIN_REGION_HEIGHT = 1200

HORIZONTAL_KERNEL_DIV = 40
VERTICAL_KERNEL_DIV = 40

MIN_KERNEL_SIZE = 80

CONNECT_KERNEL = 15
CONNECT_ITERATIONS = 2

class RegionDetector:

    def __init__(self, image_path):

        self.image_path = Path(image_path)

        self.image = None
        self.gray = None
        self.binary = None

        self.horizontal = None
        self.vertical = None

        self.mask = None

        self.labels = None
        self.stats = None
        self.centroids = None
        self.num_labels = 0

    def run(self):

        self.load()

        self.preprocess()

        self.build_projections()

        self.find_gaps()
    
    def load(self):

        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise FileNotFoundError(self.image_path)

        self.gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

    def preprocess(self):
        # Простая бинаризация через Otsu
        _, self.binary = cv2.threshold(
            self.gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Карта плотности чертежа
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (91, 91))

        self.density = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        self.density = cv2.erode(
            self.density,
            np.ones((25, 25), np.uint8),
            iterations=1
        )

    def extract_horizontal(self):

        h, w = self.binary.shape

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(MIN_KERNEL_SIZE, w // HORIZONTAL_KERNEL_DIV),
                1
            )
        )

        self.horizontal = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            kernel
        )

    def extract_vertical(self):

        h, w = self.binary.shape

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(MIN_KERNEL_SIZE, h // VERTICAL_KERNEL_DIV)
            )
        )

        self.vertical = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_OPEN,
            kernel
        )

    def combine(self):

        self.mask = cv2.bitwise_or(

            self.horizontal,

            self.vertical

        )

    def connect(self):

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                CONNECT_KERNEL,
                CONNECT_KERNEL
            )
        )

        self.mask = cv2.morphologyEx(
            self.mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=CONNECT_ITERATIONS
        )

    def find_regions(self):

        # Ищем компоненты не по маске линий,
        # а по карте плотности чертежа

        (
            self.num_labels,
            self.labels,
            self.stats,
            self.centroids

        ) = cv2.connectedComponentsWithStats(

            self.density,

            connectivity=8

        )

    def draw_regions(self):

        result = self.image.copy()

        count = 0

        regions = self.get_regions()

        for i, region in enumerate(regions):

            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]
            area = region["area"]

            print(
                f"    Drawing {i+1}: "
                f"x={x} y={y} "
                f"w={w} h={h} "
                f"area={area}"
            )

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                5
            )

        print(f"    Drawings found: {len(regions)}")

        return result

    def build_projections(self):

        # белые линии -> 1
        binary = (self.binary > 0).astype(np.uint8)

        # количество пикселей в каждом столбце
        self.vertical_projection = np.sum(binary, axis=0)

        # количество пикселей в каждой строке
        self.horizontal_projection = np.sum(binary, axis=1)

    def find_gaps(self):

        h, w = self.binary.shape

        vertical_limit = h * 0.02
        horizontal_limit = w * 0.02

        self.vertical_gaps = []
        self.horizontal_gaps = []

        # ---------- вертикальные ----------

        start = None

        for x, value in enumerate(self.vertical_projection):

            if value < vertical_limit:

                if start is None:
                    start = x

            else:

                if start is not None:

                    if x - start > 120:
                        self.vertical_gaps.append((start, x))

                    start = None

        # ---------- горизонтальные ----------

        start = None

        for y, value in enumerate(self.horizontal_projection):

            if value < horizontal_limit:

                if start is None:
                    start = y

            else:

                if start is not None:

                    if y - start > 120:
                        self.horizontal_gaps.append((start, y))

                    start = None

    def save_projection_debug(self, output_path):

        image = self.image.copy()

        # Вертикальные разделители (красные)
        for x1, x2 in self.vertical_gaps:

            cv2.rectangle(
                image,
                (x1, 0),
                (x2, image.shape[0]),
                (0, 0, 255),
                2
            )

        # Горизонтальные разделители (синие)
        for y1, y2 in self.horizontal_gaps:

            cv2.rectangle(
                image,
                (0, y1),
                (image.shape[1], y2),
                (255, 0, 0),
                2
            )

        cv2.imwrite(str(output_path), image)

    def save_mask(self, output_path):

        cv2.imwrite(

            str(output_path),

            self.density

        )

    def save_regions(self, output_path):

        image = self.draw_regions()

        cv2.imwrite(

            str(output_path),

            image

        )

    def get_regions(self):

        regions = []

        # ---------- собираем кандидатов ----------

        for i in range(1, self.num_labels):

            x = self.stats[i, cv2.CC_STAT_LEFT]
            y = self.stats[i, cv2.CC_STAT_TOP]
            w = self.stats[i, cv2.CC_STAT_WIDTH]
            h = self.stats[i, cv2.CC_STAT_HEIGHT]
            area = self.stats[i, cv2.CC_STAT_AREA]

            if area < MIN_REGION_AREA:
                continue

            if w < MIN_REGION_WIDTH:
                continue

            if h < MIN_REGION_HEIGHT:
                continue

            regions.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area
            })

        # ---------- удаляем вложенные ----------

        filtered = []

        for i, a in enumerate(regions):

            inside = False

            ax1 = a["x"]
            ay1 = a["y"]
            ax2 = ax1 + a["w"]
            ay2 = ay1 + a["h"]

            for j, b in enumerate(regions):

                if i == j:
                    continue

                bx1 = b["x"] - 30
                by1 = b["y"] - 30
                bx2 = b["x"] + b["w"] + 30
                by2 = b["y"] + b["h"] + 30

                if (
                    ax1 >= bx1 and
                    ay1 >= by1 and
                    ax2 <= bx2 and
                    ay2 <= by2
                ):

                    if b["area"] > a["area"]:
                        inside = True
                        break

            if not inside:
                filtered.append(a)

        return filtered