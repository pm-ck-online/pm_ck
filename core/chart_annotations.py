"""
chart_annotations.py
======================
[Bổ sung — Chú thích sự kiện trên biểu đồ]

Cho phép người dùng tự ghi chú các sự kiện quan trọng lên một ngày cụ thể
trên biểu đồ giá (ví dụ: "11/11/2025: Mỹ tấn công Iran" ảnh hưởng tới thị
trường chung) — hiển thị dưới dạng đường thẳng đứng + điểm đánh dấu kèm
chú thích khi di chuột vào, giúp đối chiếu diễn biến giá với sự kiện.
"""

from __future__ import annotations

import uuid
from datetime import date


class InvalidAnnotationError(ValueError):
    """Dữ liệu chú thích không hợp lệ."""


def create_annotation(symbol: str, note_date: date, text: str) -> dict:
    """Tạo một chú thích mới cho MỘT mã tại MỘT ngày cụ thể.

    `symbol` rỗng nghĩa là chú thích áp dụng chung cho MỌI mã (ví dụ sự
    kiện vĩ mô/địa chính trị ảnh hưởng toàn thị trường) — quy ước: truyền
    `symbol=""` hoặc `symbol="ALL"` để đánh dấu chú thích dùng chung.

    Trả về dict có `annotation_id` DUY NHẤT để tham chiếu/xóa sau này.
    """
    if not text or not text.strip():
        raise InvalidAnnotationError("text (nội dung chú thích) không được để trống.")

    annotation_id = f"{symbol or 'ALL'}-{note_date.isoformat()}-{uuid.uuid4().hex[:8]}"

    return {
        "annotation_id": annotation_id,
        "symbol": symbol,
        "date": note_date.isoformat(),
        "text": text.strip(),
    }
