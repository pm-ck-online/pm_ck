"""
Unit test cho core/chart_annotations.py
"""

from __future__ import annotations

from datetime import date

import pytest

from core.chart_annotations import InvalidAnnotationError, create_annotation


class TestCreateAnnotation:
    def test_creates_annotation_with_expected_fields(self):
        ann = create_annotation("HPG", date(2025, 11, 11), "Mỹ tấn công Iran")
        assert ann["symbol"] == "HPG"
        assert ann["date"] == "2025-11-11"
        assert ann["text"] == "Mỹ tấn công Iran"
        assert "annotation_id" in ann

    def test_annotation_id_is_unique(self):
        a1 = create_annotation("HPG", date(2025, 11, 11), "Sự kiện A")
        a2 = create_annotation("HPG", date(2025, 11, 11), "Sự kiện A")
        assert a1["annotation_id"] != a2["annotation_id"]

    def test_empty_symbol_defaults_to_all_in_annotation_id(self):
        ann = create_annotation("", date(2025, 11, 11), "Sự kiện chung toàn thị trường")
        assert ann["annotation_id"].startswith("ALL-2025-11-11")

    def test_raises_for_empty_text(self):
        with pytest.raises(InvalidAnnotationError):
            create_annotation("HPG", date(2025, 11, 11), "")

    def test_raises_for_whitespace_only_text(self):
        with pytest.raises(InvalidAnnotationError):
            create_annotation("HPG", date(2025, 11, 11), "    ")

    def test_strips_whitespace_from_text(self):
        ann = create_annotation("HPG", date(2025, 11, 11), "  Có khoảng trắng thừa  ")
        assert ann["text"] == "Có khoảng trắng thừa"
