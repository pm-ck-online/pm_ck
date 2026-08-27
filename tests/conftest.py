"""
conftest.py — cấu hình chung cho toàn bộ test suite.

Bỏ qua bước đăng nhập Google (`require_login()` trong `dashboard/app.py`)
khi chạy pytest — set biến môi trường `PM_CK_SKIP_LOGIN=1` tự động cho
MỌI test (autouse=True), không cần sửa từng file test riêng lẻ.

Tương tự, chặn mọi lệnh gọi giá REALTIME thật ra vnstock (qua
`PM_CK_SKIP_REALTIME=1`, đọc trong
`dashboard.app._tao_data_collector_cho_tra_cuu_realtime()`) — BẮT BUỘC
dùng biến môi trường thay vì `monkeypatch.setattr()` trực tiếp hàm, vì
`streamlit.testing.v1.AppTest` chạy script dashboard trong namespace
RIÊNG, không dùng lại `sys.modules["dashboard.app"]` của tiến trình
pytest (đã xác nhận thực tế 27/08/2026: monkeypatch hàm không hề được
gọi, nhưng dashboard vẫn âm thầm gọi API vnstock THẬT trong lúc chạy
test). `os.environ` thì luôn có tác dụng vì được đọc lại tại runtime.

KHÔNG ảnh hưởng khi chạy thật bằng `streamlit run dashboard/app.py` —
2 biến này chỉ được set trong tiến trình pytest, không tồn tại khi chạy
lệnh streamlit bình thường.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_login_in_tests(monkeypatch):
    monkeypatch.setenv("PM_CK_SKIP_LOGIN", "1")


@pytest.fixture(autouse=True)
def _skip_realtime_in_tests(monkeypatch):
    monkeypatch.setenv("PM_CK_SKIP_REALTIME", "1")
