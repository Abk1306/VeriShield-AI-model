from pathlib import Path

import pytest

from verishield_ai import analyze_document


AUTHENTIC = Path("dataset/authentic/test_document.png")
TAMPERED = Path("dataset/tampered/tampered_copy_move.png")


def test_analyze_authentic_document():
    result = analyze_document(AUTHENTIC)

    assert isinstance(result, dict)
    assert result["classification"] in {
        "AUTHENTIC",
        "SUSPICIOUS",
        "TAMPERED",
    }


def test_analyze_tampered_document():
    result = analyze_document(TAMPERED)

    assert isinstance(result, dict)
    assert result["classification"] in {
        "AUTHENTIC",
        "SUSPICIOUS",
        "TAMPERED",
    }


def test_result_schema():
    result = analyze_document(AUTHENTIC)

    required_fields = {
        "classification",
        "tampering_score",
        "confidence",
        "tampered_probability",
        "authentic_probability",
        "evidence",
        "regions",
        "feature_count",
    }

    assert required_fields.issubset(result.keys())


def test_score_range():
    result = analyze_document(AUTHENTIC)

    assert 0 <= result["tampering_score"] <= 100


def test_probability_range():
    result = analyze_document(AUTHENTIC)

    assert 0 <= result["tampered_probability"] <= 1
    assert 0 <= result["authentic_probability"] <= 1


def test_probability_sum():
    result = analyze_document(AUTHENTIC)

    total = (
        result["tampered_probability"]
        + result["authentic_probability"]
    )

    assert abs(total - 1.0) < 1e-6


def test_confidence_range():
    result = analyze_document(AUTHENTIC)

    assert 0 <= result["confidence"] <= 1


def test_feature_count():
    result = analyze_document(AUTHENTIC)

    assert result["feature_count"] == 16


def test_evidence_is_list():
    result = analyze_document(AUTHENTIC)

    assert isinstance(result["evidence"], list)


def test_regions_is_list():
    result = analyze_document(AUTHENTIC)

    assert isinstance(result["regions"], list)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        analyze_document("dataset/authentic/does_not_exist.png")


def test_directory_input():
    with pytest.raises(ValueError):
        analyze_document("dataset/authentic")


def test_unsupported_extension(tmp_path):
    unsupported = tmp_path / "document.txt"
    unsupported.write_text("not an image")

    with pytest.raises(ValueError):
        analyze_document(unsupported)


def test_region_schema():
    result = analyze_document(AUTHENTIC)

    for region in result["regions"]:
        assert {
            "x",
            "y",
            "width",
            "height",
            "area",
            "confidence",
        }.issubset(region.keys())

        assert region["width"] > 0
        assert region["height"] > 0
        assert region["area"] > 0
        assert 0 <= region["confidence"] <= 1
