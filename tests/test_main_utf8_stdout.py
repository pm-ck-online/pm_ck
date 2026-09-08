"""Test cho việc `main.py` tự reconfigure stdout/stderr sang UTF-8 khi
import — bổ sung 08/09/2026 sau sự cố thực tế: setup máy MỚI (Windows),
cả 3 script `run_full_market.py`/`update_indices.py`/`update_vcp.py`
(đều import từ `main.py`) dừng ngay ở print() ĐẦU TIÊN có dấu tiếng Việt
khi output bị CHUYỂN HƯỚNG ra file (`update_pm_ck_daily.bat` chạy qua
Task Scheduler) mà không có biến môi trường PYTHONIOENCODING=utf-8 —
UnicodeEncodeError vì Windows dùng codepage hệ thống (cp1252) thay vì
UTF-8 cho stream không phải TTY thật."""

from __future__ import annotations

import sys


class TestDamBaoStdoutUtf8SauKhiImportMain:
    def test_stdout_va_stderr_la_utf8(self):
        import main  # noqa: F401 — kích hoạt reconfigure nếu chưa import trước đó

        assert sys.stdout.encoding.lower() == "utf-8"
        assert sys.stderr.encoding.lower() == "utf-8"

    def test_in_duoc_tieng_viet_co_dau_khong_loi(self, capsys):
        import main  # noqa: F401

        print("Chạy batch toàn bộ thị trường — kiểm tra ký tự có dấu")
        ket_qua = capsys.readouterr()
        assert "Chạy batch toàn bộ thị trường" in ket_qua.out
