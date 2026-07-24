from src.loaders.pdf_loader import PDFLoader
from src.preprocessing.image_preprocessor import ImagePreprocessor

def main():
    pdf_path = "data/pdf/album_dukmasova.pdf"

    loader = PDFLoader(pdf_path)
    loader.open()

    #image_paths = loader.save_all_pages(dpi=600)
    preprocessor = ImagePreprocessor("data/images/page_001.png")
    preprocessor.load()
    preprocessor.to_grayscale()
    preprocessor.to_binary()
    preprocessor.save_binary("output/page_001_binary.png")
    preprocessor.save_gray("output/page_001_gray.png")

    print("\nСохраненные страницы:")

    #for path in image_paths:
    #    print(path)


if __name__ == "__main__":
    main()