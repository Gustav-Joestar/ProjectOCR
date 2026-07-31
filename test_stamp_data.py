from src.ocr.stamp_data import StampData


def main():
    stamp = StampData(
        designation="00-000.06.01.01.07",
        name="Опора",
        material="Сталь У8А ГОСТ 1435-99",
        scale="5:1",
        sheet_count="1",
    )

    print("=" * 70)
    print("STAMP DATA TEST")
    print("=" * 70)

    for field, value in stamp.to_dict().items():
        print(f"{field:<24} = {value!r}")


if __name__ == "__main__":
    main()