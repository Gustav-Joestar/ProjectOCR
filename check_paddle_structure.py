from paddleocr import PaddleOCR


ocr = PaddleOCR(
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


print("Тип объекта:")
print(type(ocr))

print("\nАтрибуты PaddleOCR:")
for item in dir(ocr):
    print(item)