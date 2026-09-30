from PIL import Image

from src.ocr import (
    TesseractOCREngine,
    enhance_contrast,
    grayscale_contrast_image,
    grayscale_image,
    resize_image,
    threshold_image,
)


def test_resize_preserves_dimensions_proportionally_without_mutating_input() -> None:
    image = Image.new("RGB", (10, 6), "white")
    original_size = image.size

    resized = resize_image(image, scale=1.5)

    assert resized.size == (15, 9)
    assert image.size == original_size
    assert image.mode == "RGB"


def test_grayscale_returns_l_image() -> None:
    image = Image.new("RGB", (4, 3), (10, 20, 30))

    grayscale = grayscale_image(image)

    assert grayscale.mode == "L"
    assert image.mode == "RGB"


def test_contrast_enhancement_returns_valid_image_without_mutating_input() -> None:
    image = Image.new("L", (4, 3), 100)

    enhanced = enhance_contrast(image, factor=2.0)

    assert enhanced.mode == "L"
    assert enhanced.size == image.size
    assert image.getpixel((0, 0)) == 100


def test_grayscale_contrast_pipeline_returns_l_image_without_mutating_input() -> None:
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    original_pixel = image.getpixel((0, 0))

    enhanced = grayscale_contrast_image(image)

    assert enhanced.mode == "L"
    assert enhanced.size == image.size
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == original_pixel


def test_threshold_returns_binary_image() -> None:
    image = Image.new("L", (2, 1))
    image.putdata([50, 200])

    thresholded = threshold_image(image, threshold=128)

    assert thresholded.mode == "1"
    assert thresholded.size == image.size
    assert [thresholded.getpixel((x, 0)) for x in range(2)] == [0, 255]


def test_tesseract_engine_applies_optional_preprocessor(monkeypatch) -> None:
    seen_sizes = []
    data = {
        "text": ["Receipt"],
        "left": [0],
        "top": [0],
        "width": [10],
        "height": [10],
        "conf": ["90"],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
    }

    def preprocess(image: Image.Image) -> Image.Image:
        processed = grayscale_contrast_image(image)
        seen_sizes.append((processed.mode, processed.size))
        return processed

    monkeypatch.setattr("src.ocr.engine.pytesseract.image_to_data", lambda *args, **kwargs: data)

    result = TesseractOCREngine(preprocess=preprocess).recognize_image(Image.new("RGB", (5, 4)))

    assert result.status == "success"
    assert seen_sizes == [("L", (5, 4))]