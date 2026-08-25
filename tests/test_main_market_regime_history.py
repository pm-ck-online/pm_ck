"""Unit test cho core lưu trữ chuỗi giai đoạn lịch sử (bổ sung 05/08/2026)
— core/market_regime_detector.tinh_chuoi_giai_doan_theo_ngay() đã có test
riêng ở tests/test_market_regime_detector.py; file này kiểm tra riêng
phần LƯU TRỮ VÀO STORAGE (main.run_market_regime_history_step)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.storage import Storage
from main import _luu_chuoi_giai_doan, run_market_regime_history_step


def _make_ohlcv_records(seed: int, n: int = 500, xu_huong: float = 0.0) -> list[dict]:
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    closes = 100 + np.cumsum(np.random.normal(xu_huong, 0.6, n))
    return pd.DataFrame({
        "date": dates.astype(str), "open": closes, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes, "volume": [1_000_000.0] * n,
    }).to_dict("records")


@pytest.fixture
def storage_voi_du_lieu():
    s = Storage(db_path=":memory:")
    s.save("ohlcv_history", "A", {"records": _make_ohlcv_records(1, xu_huong=0.15)})
    s.save("ohlcv_history", "B", {"records": _make_ohlcv_records(2, xu_huong=-0.15)})
    s.save("ohlcv_history", "C", {"records": _make_ohlcv_records(3, xu_huong=0.15)})
    s.save("symbol_sector", "A", {"sector": "banking"})
    s.save("symbol_sector", "B", {"sector": "banking"})
    s.save("symbol_sector", "C", {"sector": "steel"})
    yield s
    s.close()


class TestLuuChuoiGiaiDoan:
    def test_luu_va_doc_lai_dung(self):
        s = Storage(db_path=":memory:")
        chuoi = pd.Series(
            ["uptrend", "downtrend", "sideway"],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        _luu_chuoi_giai_doan(s, "test_key", chuoi)

        record = s.get_latest("chuoi_giai_doan_lich_su", "test_key")
        assert record is not None
        records = record["data"]["records"]
        assert len(records) == 3
        assert records[0] == {"date": "2024-01-01", "giai_doan": "uptrend"}
        s.close()

    def test_tu_ket_noi_lai_khi_mat_ket_noi_giua_chung(self, tmp_path):
        """SỬA LỖI 25/08/2026: sự cố thực tế — Supabase tự đóng kết nối
        giữa chừng khi lưu chuỗi giai đoạn cho ~40+ ngành liên tiếp, làm
        sập cả pipeline. `_luu_chuoi_giai_doan()` phải tự kết nối lại và
        lưu thành công, trả về đối tượng Storage MỚI."""
        db_path = str(tmp_path / "test.db")
        s = Storage(db_path=db_path)

        save_that = s.save
        da_loi_lan_dau = {"xong": False}

        def save_loi_1_lan(*args, **kwargs):
            if not da_loi_lan_dau["xong"]:
                da_loi_lan_dau["xong"] = True
                raise Exception("server closed the connection unexpectedly")
            return save_that(*args, **kwargs)

        s.save = save_loi_1_lan

        chuoi = pd.Series(["uptrend", "downtrend"], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        s_moi = _luu_chuoi_giai_doan(s, "nganh_x", chuoi)

        assert s_moi is not s
        record = s_moi.get_latest("chuoi_giai_doan_lich_su", "nganh_x")
        assert record is not None
        assert len(record["data"]["records"]) == 2
        s_moi.close()

    def test_khong_nuot_loi_khac_khong_phai_mat_ket_noi(self):
        s = Storage(db_path=":memory:")

        def save_loi_khac(*args, **kwargs):
            raise ValueError("lỗi khác, không liên quan mất kết nối")

        s.save = save_loi_khac
        chuoi = pd.Series(["uptrend"], index=pd.to_datetime(["2024-01-01"]))
        with pytest.raises(ValueError):
            _luu_chuoi_giai_doan(s, "test_key", chuoi)
        s.close()


class TestRunMarketRegimeHistoryStep:
    def test_luu_duoc_ca_thi_truong_va_tung_nganh(self, storage_voi_du_lieu):
        run_market_regime_history_step(storage_voi_du_lieu)

        thi_truong = storage_voi_du_lieu.get_latest("chuoi_giai_doan_lich_su", "thi_truong")
        banking = storage_voi_du_lieu.get_latest("chuoi_giai_doan_lich_su", "banking")
        steel = storage_voi_du_lieu.get_latest("chuoi_giai_doan_lich_su", "steel")

        assert thi_truong is not None
        assert banking is not None
        assert steel is not None
        assert len(thi_truong["data"]["records"]) > 0
        assert len(banking["data"]["records"]) > 0
        assert len(steel["data"]["records"]) > 0

    def test_khong_loi_khi_khong_co_du_lieu(self):
        s = Storage(db_path=":memory:")
        # Không raise lỗi, chỉ đơn giản không lưu gì (không có dữ liệu).
        run_market_regime_history_step(s)
        assert s.get_latest("chuoi_giai_doan_lich_su", "thi_truong") is None
        s.close()

    def test_gop_dung_du_lieu_tu_nhieu_lo_ohlcv(self):
        """SỬA LỖI 25/08/2026: get_latest_many() được chia thành nhiều lô
        30 mã/lô (thay vì 1 lượt duy nhất) để tránh timeout Supabase khi
        dữ liệu lớn — test này xác nhận vòng lặp gộp ĐÚNG dữ liệu từ
        NHIỀU lô, không chỉ đúng khi tình cờ vừa 1 lô như các fixture khác
        trong file này (chỉ có 3 mã)."""
        s = Storage(db_path=":memory:")
        so_ma = 35  # > 30 -> chắc chắn rơi vào 2 lô
        for i in range(so_ma):
            ma = f"MA{i:03d}"
            s.save("ohlcv_history", ma, {"records": _make_ohlcv_records(seed=i, xu_huong=0.15)})
            s.save("symbol_sector", ma, {"sector": "nganh_gop_lo"})

        run_market_regime_history_step(s)

        thi_truong = s.get_latest("chuoi_giai_doan_lich_su", "thi_truong")
        assert thi_truong is not None
        assert len(thi_truong["data"]["records"]) > 0

        nganh = s.get_latest("chuoi_giai_doan_lich_su", "nganh_gop_lo")
        assert nganh is not None
        assert len(nganh["data"]["records"]) > 0
        s.close()

    def test_steel_toan_tang_phai_ra_uptrend(self, storage_voi_du_lieu):
        # Ngành "steel" chỉ có 1 mã (C), xu hướng tăng RÕ RỆT (0.15/phiên)
        # -> phần lớn phải là "uptrend".
        run_market_regime_history_step(storage_voi_du_lieu)
        steel = storage_voi_du_lieu.get_latest("chuoi_giai_doan_lich_su", "steel")
        gia_tri = [r["giai_doan"] for r in steel["data"]["records"]]
        ty_le_uptrend = gia_tri.count("uptrend") / len(gia_tri)
        assert ty_le_uptrend > 0.5
