import pytest

from src.evaluation.kie import evaluate_extractions
from src.extraction.extractor import RuleBasedExtractor
from src.extraction.layout import group_words_into_lines
from src.extraction.normalization import normalize_money, normalize_text
from src.extraction.types import ExtractedField, ExtractionResult
from src.ocr import OCRResult, OCRWord


def make_ocr(text: str, words: tuple[OCRWord, ...] = ()) -> OCRResult:
    return OCRResult(text=text, words=words, status="success")


def test_normalization_preserves_content_and_normalizes_money() -> None:
    assert normalize_text("  MR\n  SHOP   SDN BHD ") == "mr shop sdn bhd"
    assert normalize_money("RM 112.45") == "112.45"
    assert normalize_money("112,45") == "112.45"
    assert normalize_money("Total (RM): 112.45") == "112.45"


def test_line_grouping_uses_vertical_centers_and_horizontal_order() -> None:
    words = (
        OCRWord("World", (50, 12, 30, 10), 80.0),
        OCRWord("Hello", (5, 10, 30, 10), 90.0),
        OCRWord("Next", (5, 40, 30, 10), 70.0),
    )

    lines = group_words_into_lines(words, y_tolerance=4)

    assert [line.text for line in lines] == ["Hello World", "Next"]
    assert lines[0].confidence == 85.0


def test_rule_extractor_finds_date_total_company_and_address() -> None:
    ocr = make_ocr(
        "MY SHOP SDN BHD\n123 JALAN UTAMA\nJOHOR BAHRU 80000\n25/12/2018\nTOTAL RM 112.45",
        (
            OCRWord("MY SHOP SDN BHD", (5, 5, 100, 12), 90.0),
            OCRWord("123 JALAN UTAMA", (5, 25, 130, 12), 85.0),
            OCRWord("JOHOR BAHRU 80000", (5, 45, 150, 12), 80.0),
            OCRWord("25/12/2018", (5, 70, 90, 12), 95.0),
            OCRWord("TOTAL RM 112.45", (5, 95, 140, 12), 88.0),
        ),
    )

    result = RuleBasedExtractor().extract(ocr)

    assert result.company.value == "MY SHOP SDN BHD"
    assert result.address.value == "123 JALAN UTAMA JOHOR BAHRU 80000"
    assert result.date.value == "25/12/2018"
    assert result.total.value == "RM 112.45"


def test_address_rule_does_not_reuse_company_line() -> None:
    ocr = make_ocr(
        "TAMAN JOHOR STORE\n12 JALAN UTAMA\n25/12/2018",
        (
            OCRWord("TAMAN JOHOR STORE", (5, 5, 120, 12), 90.0),
            OCRWord("12 JALAN UTAMA", (5, 25, 110, 12), 85.0),
            OCRWord("25/12/2018", (5, 45, 90, 12), 95.0),
        ),
    )

    result = RuleBasedExtractor().extract(ocr)

    assert result.company.value == "TAMAN JOHOR STORE"
    assert result.address.value == "12 JALAN UTAMA"


def test_rule_extractor_handles_missing_and_malformed_ocr() -> None:
    extractor = RuleBasedExtractor()

    missing = extractor.extract(OCRResult("", (), "empty"))
    malformed = extractor.extract(OCRResult("Visible text", (object(),), "success"))  # type: ignore[arg-type]

    assert missing == ExtractionResult()
    assert malformed.company.value == "Visible text"
    assert malformed.date.value is None
    assert malformed.total.value is None


def test_evaluation_handles_normalized_matches_missing_fields_and_macro_f1() -> None:
    predictions = [
        ExtractionResult(
            company=ExtractedField("mr shop"),
            date=ExtractedField("25/12/2018"),
            address=ExtractedField("1 jalan utama"),
            total=ExtractedField("RM 112.45"),
        ),
        ExtractionResult(),
    ]
    ground_truth = [
        {"company": "MR  SHOP", "date": "25/12/2018", "address": "1 JALAN UTAMA", "total": "112,45"},
        {"company": "Other Store", "date": "01/01/2019", "address": "2 Road", "total": "20.00"},
    ]

    metrics = evaluate_extractions(predictions, ground_truth)

    assert metrics.loc["company", "normalized_exact_match"] == 0.5
    assert metrics.loc["total", "normalized_exact_match"] == 0.5
    assert metrics.loc["total", "f1"] == pytest.approx(2 / 3)
    assert 0.0 <= metrics.loc["macro_avg", "f1"] <= 1.0
    
def test_line_grouping_handles_realistic_ocr_words() -> None:
    words = (
        OCRWord("MY", (5, 10, 20, 12), 95.0),
        OCRWord("SHOP", (30, 11, 35, 11), 94.0),
        OCRWord("SDN", (70, 10, 25, 12), 93.0),
        OCRWord("BHD", (100, 12, 25, 10), 92.0),
        OCRWord("NEXT", (5, 45, 35, 11), 90.0),
    )

    lines = group_words_into_lines(words)

    assert len(lines) == 2
    assert lines[0].text == "MY SHOP SDN BHD"
    assert lines[1].text == "NEXT"


def test_total_can_be_on_following_line() -> None:
    ocr = make_ocr(
        "TOTAL\nRM 112.45",
        (
            OCRWord("TOTAL", (5, 100, 50, 12), 95.0),
            OCRWord("RM", (5, 120, 20, 12), 94.0),
            OCRWord("112.45", (30, 120, 50, 12), 93.0),
        ),
    )

    result = RuleBasedExtractor().extract(ocr)

    assert result.total.value == "RM 112.45"


def test_total_ignores_subtotal_and_cash_lines() -> None:
    ocr = make_ocr(
        "SUBTOTAL RM 200.00\n"
        "CASH RM 500.00\n"
        "CHANGE RM 300.00\n"
        "TOTAL RM 112.45",
        (
            OCRWord("SUBTOTAL RM 200.00", (5, 10, 150, 12), 90.0),
            OCRWord("CASH RM 500.00", (5, 30, 140, 12), 90.0),
            OCRWord("CHANGE RM 300.00", (5, 50, 150, 12), 90.0),
            OCRWord("TOTAL RM 112.45", (5, 70, 140, 12), 90.0),
        ),
    )

    result = RuleBasedExtractor().extract(ocr)

    assert result.total.value == "RM 112.45"


def test_address_does_not_append_unrelated_comma_line() -> None:
    ocr = make_ocr(
        "MY SHOP SDN BHD\n"
        "12 JALAN UTAMA\n"
        "JOHOR BAHRU 80000\n"
        "TEL: 12345678\n"
        "THANK YOU, HAVE A NICE DAY",
        (
            OCRWord("MY SHOP SDN BHD", (5, 5, 120, 12), 90.0),
            OCRWord("12 JALAN UTAMA", (5, 25, 120, 12), 90.0),
            OCRWord("JOHOR BAHRU 80000", (5, 45, 150, 12), 90.0),
            OCRWord("TEL: 12345678", (5, 65, 100, 12), 90.0),
            OCRWord(
                "THANK YOU, HAVE A NICE DAY",
                (5, 85, 180, 12),
                90.0,
            ),
        ),
    )

    result = RuleBasedExtractor().extract(ocr)

    assert result.address.value == (
        "12 JALAN UTAMA JOHOR BAHRU 80000"
    )


def test_address_can_span_multiple_lines() -> None:
    ocr = make_ocr(
        "MY SHOP SDN BHD\n"
        "12 JALAN UTAMA\n"
        "TAMAN JOHOR\n"
        "JOHOR BAHRU 80000\n"
        "DATE 25/12/2018",
        (
            OCRWord("MY SHOP SDN BHD", (5, 5, 120, 12), 90.0),
            OCRWord("12 JALAN UTAMA", (5, 25, 120, 12), 90.0),
            OCRWord("TAMAN JOHOR", (5, 45, 100, 12), 90.0),
            OCRWord(
                "JOHOR BAHRU 80000",
                (5, 65, 150, 12),
                90.0,
            ),
            OCRWord(
                "DATE 25/12/2018",
                (5, 85, 120, 12),
                90.0,
            ),
        ),
    )

    result = RuleBasedExtractor().extract(ocr)

    assert result.address.value == (
        "12 JALAN UTAMA TAMAN JOHOR JOHOR BAHRU 80000"
    )