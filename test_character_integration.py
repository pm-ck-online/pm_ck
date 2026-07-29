"""
tests/test_character_integration.py

Unit test cho core/character_integration.py — dùng dict/DataFrame giả lập
khớp ĐÚNG cấu trúc thật của evaluate_stock_signal() (core/stock_signal_engine.py)
và get_allocation_recommendation() (core/capital_allocator.py), không gọi
2 hàm đó thật (tránh phụ thuộc lẫn nhau giữa các test module).
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.character_integration import (
    dieu_chinh_tin_hieu_theo_tinh_cach,
    dieu_chinh_phan_bo_theo_tinh_cach,
    quet_tinh_cach_watchlist,
)
from core.stock_character_classifier import NHAN_BUNG_NO_NGAN, NHAN_DUT_KHOAT_TANG


# ---------------------------------------------------------------------------
# Fixture: dict mẫu khớp cấu trúc thật đã xác nhận trong pm_ck
# ---------------------------------------------------------------------------

def _mau_ket_qua_tin_hieu_mua_breakout(stock_score=0.85):
    """Khớp đúng cấu trúc return của evaluate_stock_signal() khi khuyen_nghi=MUA."""
    return {
        "ma": "HPG",
        "khuyen_nghi": "MUA",
        "loai_ban": None,
        "uu_tien": None,
        "stock_score": stock_score,
        "fundamental_score": 1.0,
        "technical_score": 1.0,
        "chi_tiet": {
            "co_ban_dat": ["EPS tăng 18% YoY"],
            "ky_thuat_dat": ["Breakout kèm volume lớn"],
            "mau_hinh_ky_thuat": "BREAKOUT",
        },
        "khoang_gia_vao_lenh_de_xuat": [26400, 26650],
        "canh_bao": [],
        "ghi_chu": "Cần đối chiếu Module Phân bổ Vốn...",
    }


def _mau_ket_qua_tinh_cach(nhan="LINH_XINH", choppiness_score=1.4, canh_bao=None):
    return {
        "ma": "HPG",
        "nhan_tinh_cach": nhan,
        "character_score": -0.5,
        "choppiness_score": choppiness_score,
        "chi_tiet": {},
        "canh_bao": canh_bao or [],
        "khuyen_nghi_chien_luoc": "Ưu tiên chiến lược giao dịch biên độ",
        "do_tin_cay_thap": False,
    }


def _mau_allocation_result():
    """Khớp đúng cấu trúc return của get_allocation_recommendation()."""
    return {
        "target_pct": 0.20,
        "tranches": [0.3, 0.5, 0.2],
        "entry_price_range": {"low": 26400, "high": 26650},
        "stop_loss": 25900.0,
        "max_position_size": 4500,
        "notes": ["Giai đoạn UPTREND, độ tin cậy CAO."],
    }


# ---------------------------------------------------------------------------
# Test dieu_chinh_tin_hieu_theo_tinh_cach
# ---------------------------------------------------------------------------

def test_chiet_khau_stock_score_khi_breakout_va_linh_xinh():
    tin_hieu = _mau_ket_qua_tin_hieu_mua_breakout(stock_score=0.85)
    tinh_cach = _mau_ket_qua_tinh_cach(nhan="LINH_XINH", choppiness_score=1.4)

    kq = dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu, tinh_cach)

    assert kq["stock_score"] == pytest.approx(0.85 * 0.7, rel=1e-6)
    assert any("chiết khấu" in c for c in kq["canh_bao"])
    assert kq["nhan_tinh_cach"] == "LINH_XINH"
    # Không sửa dict gốc
    assert tin_hieu["stock_score"] == 0.85
    assert tin_hieu["canh_bao"] == []


def test_khong_chiet_khau_khi_khong_phai_breakout():
    tin_hieu = _mau_ket_qua_tin_hieu_mua_breakout(stock_score=0.85)
    tin_hieu["chi_tiet"]["mau_hinh_ky_thuat"] = "PULLBACK"
    tinh_cach = _mau_ket_qua_tinh_cach(nhan="LINH_XINH", choppiness_score=1.4)

    kq = dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu, tinh_cach)

    assert kq["stock_score"] == 0.85  # không đổi


def test_khong_chiet_khau_khi_choppiness_thap():
    tin_hieu = _mau_ket_qua_tin_hieu_mua_breakout(stock_score=0.85)
    tinh_cach = _mau_ket_qua_tinh_cach(nhan="DUT_KHOAT_TANG", choppiness_score=0.3)

    kq = dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu, tinh_cach)

    assert kq["stock_score"] == 0.85


def test_khong_chiet_khau_khi_khuyen_nghi_khong_phai_mua():
    tin_hieu = _mau_ket_qua_tin_hieu_mua_breakout(stock_score=0.85)
    tin_hieu["khuyen_nghi"] = "GIU_THEO_DOI"
    tinh_cach = _mau_ket_qua_tinh_cach(nhan="LINH_XINH", choppiness_score=1.5)

    kq = dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu, tinh_cach)

    assert kq["stock_score"] == 0.85


def test_gop_canh_bao_squat_churning_vao_tin_hieu():
    tin_hieu = _mau_ket_qua_tin_hieu_mua_breakout()
    tin_hieu["canh_bao"] = ["Cảnh báo gốc từ stock_signal_engine"]
    tinh_cach = _mau_ket_qua_tinh_cach(
        nhan="TRUNG_TINH", choppiness_score=0.5, canh_bao=["SQUAT — bứt phá yếu"]
    )

    kq = dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu, tinh_cach)

    assert "Cảnh báo gốc từ stock_signal_engine" in kq["canh_bao"]
    assert "SQUAT — bứt phá yếu" in kq["canh_bao"]


def test_ban_cat_lo_giu_nguyen_khong_bi_dong_vao_chi_tiet_loi():
    """Trường hợp evaluate_stock_signal trả về BÁN CẮT LỖ (chi_tiet không có
    mau_hinh_ky_thuat) — hàm điều chỉnh không được lỗi/crash."""
    tin_hieu = {
        "ma": "HPG", "khuyen_nghi": "BAN", "loai_ban": "CAT_LO", "uu_tien": "CAO",
        "stock_score": None, "fundamental_score": None, "technical_score": None,
        "chi_tiet": {"ly_do": ["Đã chạm điều kiện cắt lỗ"]},
        "khoang_gia_vao_lenh_de_xuat": None, "canh_bao": [],
        "ghi_chu": "Bán cắt lỗ được ưu tiên tuyệt đối.",
    }
    tinh_cach = _mau_ket_qua_tinh_cach()
    kq = dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu, tinh_cach)
    assert kq["khuyen_nghi"] == "BAN"
    assert kq["stock_score"] is None  # không lỗi khi stock_score=None


# ---------------------------------------------------------------------------
# Test dieu_chinh_phan_bo_theo_tinh_cach
# ---------------------------------------------------------------------------

def test_giam_ty_trong_khi_bung_no_ngan():
    alloc = _mau_allocation_result()
    tinh_cach = _mau_ket_qua_tinh_cach(nhan=NHAN_BUNG_NO_NGAN, choppiness_score=0.9)

    kq = dieu_chinh_phan_bo_theo_tinh_cach(alloc, tinh_cach)

    assert kq["target_pct"] == pytest.approx(0.10)
    assert kq["max_position_size"] == 2250
    assert any("BÙNG_NỔ_NGẮN" in n for n in kq["notes"])
    # Không sửa dict gốc
    assert alloc["target_pct"] == 0.20
    assert alloc["max_position_size"] == 4500


def test_giam_ty_trong_khi_co_canh_bao_churning():
    alloc = _mau_allocation_result()
    tinh_cach = _mau_ket_qua_tinh_cach(
        nhan="TRUNG_TINH", canh_bao=["CHURNING — nghi ngờ phân phối ẩn"]
    )

    kq = dieu_chinh_phan_bo_theo_tinh_cach(alloc, tinh_cach)

    assert kq["target_pct"] == pytest.approx(0.10)
    assert any("CHURNING" in n for n in kq["notes"])


def test_khong_giam_ty_trong_khi_binh_thuong():
    alloc = _mau_allocation_result()
    tinh_cach = _mau_ket_qua_tinh_cach(nhan=NHAN_DUT_KHOAT_TANG, choppiness_score=0.3)

    kq = dieu_chinh_phan_bo_theo_tinh_cach(alloc, tinh_cach)

    assert kq["target_pct"] == 0.20
    assert kq["max_position_size"] == 4500
    assert len(kq["notes"]) == 1  # giữ nguyên note gốc, không thêm note mới


def test_dieu_chinh_phan_bo_khi_khong_co_max_position_size():
    alloc = {"target_pct": 0.15, "tranches": [1.0], "entry_price_range": {"low": 10, "high": 11},
             "stop_loss": 9.5, "max_position_size": None, "notes": []}
    tinh_cach = _mau_ket_qua_tinh_cach(nhan=NHAN_BUNG_NO_NGAN)

    kq = dieu_chinh_phan_bo_theo_tinh_cach(alloc, tinh_cach)

    assert kq["target_pct"] == pytest.approx(0.075)
    assert kq["max_position_size"] is None  # không lỗi khi None


# ---------------------------------------------------------------------------
# Test quet_tinh_cach_watchlist
# ---------------------------------------------------------------------------

def _tao_df_gia_lap(n=600, seed=1, giam_manh=False):
    rng = np.random.default_rng(seed)
    if giam_manh:
        base = 100 + rng.normal(0, 0.3, n - 8).cumsum() * 0.05
        tail = base[-1] - np.linspace(1, 12, 8)
        close = np.concatenate([base, tail])
    else:
        close = 50 + rng.normal(0, 2.0, n)
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame(index=dates)
    df["close"] = close
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = np.maximum(df["open"], df["close"]) + 0.3
    df["low"] = np.minimum(df["open"], df["close"]) - 0.3
    df["volume"] = rng.integers(500_000, 1_500_000, n)
    return df


def test_quet_watchlist_thanh_cong_tat_ca_ma():
    watchlist = ["HPG", "SSI", "VIB"]

    def lay_ohlcv_fn(ma):
        return _tao_df_gia_lap(seed=hash(ma) % 1000)

    df_ket_qua = quet_tinh_cach_watchlist(watchlist, lay_ohlcv_fn)

    assert len(df_ket_qua) == 3
    assert set(df_ket_qua["ma"]) == set(watchlist)
    assert df_ket_qua["loi"].isna().all()
    assert "nhan_tinh_cach" in df_ket_qua.columns


def test_quet_watchlist_khong_dung_khi_1_ma_loi():
    watchlist = ["HPG", "MA_LOI", "SSI"]

    def lay_ohlcv_fn(ma):
        if ma == "MA_LOI":
            return pd.DataFrame()  # dữ liệu rỗng -> lỗi
        return _tao_df_gia_lap(seed=hash(ma) % 1000)

    df_ket_qua = quet_tinh_cach_watchlist(watchlist, lay_ohlcv_fn)

    assert len(df_ket_qua) == 3  # vẫn có đủ 3 dòng, không dừng giữa chừng
    row_loi = df_ket_qua[df_ket_qua["ma"] == "MA_LOI"].iloc[0]
    assert row_loi["loi"] is not None
    # 2 mã còn lại vẫn thành công
    assert df_ket_qua[df_ket_qua["ma"] != "MA_LOI"]["loi"].isna().all()


def test_quet_watchlist_bat_exception_khong_xac_dinh():
    watchlist = ["HPG", "MA_EXCEPTION"]

    def lay_ohlcv_fn(ma):
        if ma == "MA_EXCEPTION":
            raise RuntimeError("Lỗi API giả lập")
        return _tao_df_gia_lap(seed=1)

    df_ket_qua = quet_tinh_cach_watchlist(watchlist, lay_ohlcv_fn)

    assert len(df_ket_qua) == 2
    row_loi = df_ket_qua[df_ket_qua["ma"] == "MA_EXCEPTION"].iloc[0]
    assert "RuntimeError" in row_loi["loi"]


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
