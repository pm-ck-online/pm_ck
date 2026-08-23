"""
conftest.py — cấu hình chung cho toàn bộ test suite.

Bỏ qua bước đăng nhập Google (`require_login()` trong `dashboard/app.py`)
khi chạy pytest — set biến môi trường `PM_CK_SKIP_LOGIN=1` tự động cho
MỌI test (autouse=True), không cần sửa từng file test riêng lẻ.

KHÔNG ảnh hưởng khi chạy thật bằng `streamlit run dashboard/app.py` —
biến này chỉ được set trong tiến trình pytest, không tồn tại khi chạy
lệnh streamlit bình thường.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_login_in_tests(monkeypatch):
    monkeypatch.setenv("PM_CK_SKIP_LOGIN", "1")
