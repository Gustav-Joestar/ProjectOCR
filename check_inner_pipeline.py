from paddleocr import PaddleOCR


ocr = PaddleOCR(
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


inner = ocr.paddlex_pipeline._pipeline


print("Тип внутреннего pipeline:")
print(type(inner))


print("\nАтрибуты внутреннего pipeline:")
for item in dir(inner):
    print(item)