"""
core/market_regime_ensemble.py
==================================
Ensemble 3 phương pháp học thuật độc lập để xác định giai đoạn thị
trường/ngành (UPTREND/SIDEWAY/DOWNTREND) — nâng cấp `market_regime_detector.py`
(trước đây chỉ dùng 1 phương pháp Breadth) thành hệ thống biểu quyết đa
số có trọng số từ 3 phương pháp độc lập:

  A. Market Breadth (% cổ phiếu trên EMA200)      — core/market_breadth.py
  B. Rule-based Peak-Trough (Dow Theory hiện đại hóa, Zegadło 2022)
  C. Markov Regime-Switching (Hamilton)            — statsmodels

⚠️ ĐÂY LÀ CÔNG CỤ PHÂN LOẠI THAM KHẢO tổng hợp từ nhiều phương pháp học
thuật — KHÔNG phải khuyến nghị đầu tư hay dự báo được đảm bảo. Dù đồng
thuận 3/3 phương pháp cũng KHÔNG loại trừ khả năng thị trường đảo chiều
bất ngờ do sự kiện đột biến (đúng nguyên tắc override đã thiết kế ở
Module Điểm Vĩ Mô).

NGUYÊN TẮC BẮT BUỘC cho mọi module hạ nguồn (capital_allocation_engine.py,
stock_signal_engine.py, mọi phân tích/báo cáo khác): PHẢI tham chiếu vào
`KET_LUAN_TONG_HOP` của ensemble này — KHÔNG được lấy trực tiếp kết quả
của riêng phương pháp A (breadth) như trước đây.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.market_breadth import calculate_ema200_breadth
from core.volatility_contraction_scanner import tim_dinh_day_cuc_bo

# ==============================================================================
# TRỌNG SỐ + NGƯỠNG — theo đúng bảng xếp hạng độ tin cậy học thuật đã khảo
# sát (A hạng 1, C hạng 2, B hạng 3).
# ==============================================================================

TRONG_SO_PHUONG_PHAP = {"A": 0.4, "B": 0.25, "C": 0.35}
NGUONG_BULL_MARKOV = 0.65
NGUONG_BEAR_MARKOV = 0.35
SO_PHIEN_TOI_THIEU_MARKOV = 250
NGUONG_UPTREND_BREADTH = 60.0
NGUONG_DOWNTREND_BREADTH = 40.0


# ==============================================================================
# PHƯƠNG PHÁP A — Market Breadth (tái sử dụng nguyên vẹn core/market_breadth.py)
# ==============================================================================

def phuong_phap_A_breadth(group_snapshots: list[dict]) -> dict:
    """Trả về {"nhan", "chi_tiet", "gia_tri_so"} dựa trên
    `calculate_ema200_breadth()` đã có sẵn."""
    ket_qua = calculate_ema200_breadth(group_snapshots)
    pct = ket_qua["breadth_pct"]
    if pct is None:
        return {
            "nhan": "SIDEWAY",
            "chi_tiet": "Không đủ dữ liệu breadth (không có mã nào có EMA200 hợp lệ).",
            "gia_tri_so": None,
        }
    nhan = (
        "UPTREND" if pct > NGUONG_UPTREND_BREADTH else
        "DOWNTREND" if pct < NGUONG_DOWNTREND_BREADTH else
        "SIDEWAY"
    )
    return {
        "nhan": nhan,
        "chi_tiet": f"{pct:.1f}% mã trên EMA200 ({ket_qua['n_above']}/{ket_qua['n_valid']} mã)",
        "gia_tri_so": round(pct, 2),
    }


# ==============================================================================
# PHƯƠNG PHÁP B — Rule-based Peak-Trough (Dow Theory hiện đại hóa, Zegadło 2022)
# Tái sử dụng nguyên vẹn tim_dinh_day_cuc_bo() đã có (module VCP).
# ==============================================================================

def phuong_phap_B_peak_trough(df_chi_so: Optional[pd.DataFrame], so_chu_ky_xet: int = 2) -> dict:
    """Phân loại UPTREND (Higher High + Higher Low) / DOWNTREND (Lower
    High + Lower Low) / SIDEWAY (còn lại) dựa trên `so_chu_ky_xet` cặp
    đỉnh-đáy gần nhất của `df_chi_so` (VN-Index cho toàn thị trường, chỉ
    số/đại diện ngành cho từng ngành).

    `df_chi_so` cần ĐẦY ĐỦ cột OHLCV (open/high/low/close/volume) — do
    tái sử dụng nguyên vẹn `tim_dinh_day_cuc_bo()` (module VCP) yêu cầu
    đủ cấu trúc này, không chỉ riêng cột close.
    """
    if df_chi_so is None or df_chi_so.empty or len(df_chi_so) < 20:
        return {"nhan": "SIDEWAY", "chi_tiet": "Chưa đủ dữ liệu để tìm đỉnh/đáy.", "gia_tri_so": None}

    diem = tim_dinh_day_cuc_bo(df_chi_so, khoang_cach_toi_thieu=5)
    dinh_list = [d["gia"] for d in diem if d["loai"] == "dinh"][-(so_chu_ky_xet + 1):]
    day_list = [d["gia"] for d in diem if d["loai"] == "day"][-(so_chu_ky_xet + 1):]

    if len(dinh_list) < 2 or len(day_list) < 2:
        return {"nhan": "SIDEWAY", "chi_tiet": "Chưa đủ dữ liệu đỉnh/đáy để xác định.", "gia_tri_so": None}

    dinh_tang_dan = all(dinh_list[i] < dinh_list[i + 1] for i in range(len(dinh_list) - 1))
    day_tang_dan = all(day_list[i] < day_list[i + 1] for i in range(len(day_list) - 1))
    dinh_giam_dan = all(dinh_list[i] > dinh_list[i + 1] for i in range(len(dinh_list) - 1))
    day_giam_dan = all(day_list[i] > day_list[i + 1] for i in range(len(day_list) - 1))

    if dinh_tang_dan and day_tang_dan:
        nhan = "UPTREND"
    elif dinh_giam_dan and day_giam_dan:
        nhan = "DOWNTREND"
    else:
        nhan = "SIDEWAY"

    return {
        "nhan": nhan,
        "chi_tiet": (
            f"Đỉnh gần nhất: {[round(x, 2) for x in dinh_list]}, "
            f"Đáy gần nhất: {[round(x, 2) for x in day_list]}"
        ),
        "gia_tri_so": None,
    }


# ==============================================================================
# PHƯƠNG PHÁP C — Markov Regime-Switching (Hamilton, statsmodels)
# ==============================================================================

def phuong_phap_C_markov_switching(
    df_chi_so: Optional[pd.DataFrame], so_phien_toi_thieu: int = SO_PHIEN_TOI_THIEU_MARKOV
) -> dict:
    """Fit mô hình Markov Regime-Switching 2 trạng thái (mean khác nhau
    giữa 2 chế độ, phương sai chuyển đổi theo chế độ) trên chuỗi lợi
    suất log của `df_chi_so`. Trạng thái có mean CAO HƠN = chế độ Bull.

    Yêu cầu tối thiểu `so_phien_toi_thieu` phiên (mặc định 250 — ~1 năm)
    để fit ổn định — KHÔNG cố fit mô hình không đủ tin cậy, trả về
    SIDEWAY kèm lý do rõ ràng trong `chi_tiet` (tránh hiểu nhầm là ngành
    thực sự đang đi ngang).
    """
    if df_chi_so is None or df_chi_so.empty or len(df_chi_so) < so_phien_toi_thieu:
        return {
            "nhan": "SIDEWAY",
            "chi_tiet": f"Chưa đủ dữ liệu để fit mô hình Markov (cần >= {so_phien_toi_thieu} phiên).",
            "gia_tri_so": None,
        }

    log_return = np.log(df_chi_so["close"] / df_chi_so["close"].shift(1)).dropna()
    if len(log_return) < so_phien_toi_thieu:
        return {
            "nhan": "SIDEWAY",
            "chi_tiet": "Chưa đủ dữ liệu lợi suất hợp lệ để fit mô hình Markov.",
            "gia_tri_so": None,
        }

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        mo_hinh = MarkovRegression(log_return.values, k_regimes=2, trend="c", switching_variance=True)
        ket_qua_fit = mo_hinh.fit(disp=False)

        # QUAN TRỌNG: với trend="c", switching_variance=True, thứ tự
        # tham số CHÍNH XÁC (đã kiểm chứng thực tế qua param_names) là:
        # [p[0->0], p[1->0], const[0], const[1], sigma2[0], sigma2[1]]
        # -> mean của 2 chế độ nằm ở INDEX 2 và 3, KHÔNG PHẢI 0 và 1.
        idx_const0 = mo_hinh.param_names.index("const[0]")
        idx_const1 = mo_hinh.param_names.index("const[1]")
        mean_theo_che_do = [ket_qua_fit.params[idx_const0], ket_qua_fit.params[idx_const1]]
        che_do_bull = int(np.argmax(mean_theo_che_do))

        xac_suat_muot = ket_qua_fit.smoothed_marginal_probabilities
        # Trả về numpy.ndarray shape (n, 2) — không phải DataFrame.
        xac_suat_bull_hien_tai = float(np.asarray(xac_suat_muot)[-1, che_do_bull])
    except Exception as exc:  # noqa: BLE001 — mô hình có thể không hội tụ với 1 số chuỗi dữ liệu
        return {
            "nhan": "SIDEWAY",
            "chi_tiet": f"Lỗi khi fit mô hình Markov (không hội tụ hoặc dữ liệu không phù hợp): {exc}",
            "gia_tri_so": None,
        }

    if xac_suat_bull_hien_tai >= NGUONG_BULL_MARKOV:
        nhan = "UPTREND"
    elif xac_suat_bull_hien_tai <= NGUONG_BEAR_MARKOV:
        nhan = "DOWNTREND"
    else:
        nhan = "SIDEWAY"

    return {
        "nhan": nhan,
        "chi_tiet": f"Xác suất mượt đang ở chế độ Bull: {xac_suat_bull_hien_tai:.1%}",
        "gia_tri_so": round(xac_suat_bull_hien_tai, 3),
    }


# ==============================================================================
# TỔNG HỢP 3 PHƯƠNG PHÁP — Biểu quyết đa số có trọng số
# ==============================================================================

def tong_hop_3_phuong_phap(ket_qua_A: dict, ket_qua_B: dict, ket_qua_C: dict) -> dict:
    """Quy tắc tổng hợp:
      1. Cả 3 phương pháp đồng thuận -> kết luận đó, độ tin cậy CAO.
      2. 2/3 đồng thuận -> lấy nhãn đa số, độ tin cậy TRUNG BÌNH.
      3. Cả 3 khác nhau hoàn toàn -> dùng điểm trọng số phá thế bế tắc,
         độ tin cậy THẤP, cần xem lại thủ công.
    """
    phieu = {"A": ket_qua_A["nhan"], "B": ket_qua_B["nhan"], "C": ket_qua_C["nhan"]}
    dem_phieu: dict[str, float] = {}
    for pp, nhan in phieu.items():
        dem_phieu[nhan] = dem_phieu.get(nhan, 0.0) + TRONG_SO_PHUONG_PHAP[pp]

    nhan_cuoi_cung = max(dem_phieu, key=dem_phieu.get)
    so_phuong_phap_dong_thuan = sum(1 for nhan in phieu.values() if nhan == nhan_cuoi_cung)

    do_tin_cay = (
        "CAO" if so_phuong_phap_dong_thuan == 3 else
        "TRUNG_BINH" if so_phuong_phap_dong_thuan == 2 else
        "THAP"
    )

    return {
        "nhan_tong_hop": nhan_cuoi_cung,
        "do_tin_cay": do_tin_cay,
        "so_phuong_phap_dong_thuan": so_phuong_phap_dong_thuan,
        "phieu_bau_chi_tiet": phieu,
        "diem_trong_so": {k: round(v, 3) for k, v in dem_phieu.items()},
    }


# ==============================================================================
# HÀM CHÍNH — Chạy cho 1 nhóm (Toàn thị trường HOẶC 1 ngành), và tổng hợp
# cho TẤT CẢ nhóm dưới dạng bảng (DataFrame).
# ==============================================================================

def phan_tich_ensemble_theo_nhom(
    ten_nhom: str,
    group_snapshots: list[dict],
    df_chi_so_dai_dien: Optional[pd.DataFrame],
) -> dict:
    """Chạy cả 3 phương pháp cho 1 nhóm (Toàn thị trường hoặc 1 ngành),
    trả về dict đầy đủ kết quả từng phương pháp + kết luận tổng hợp."""
    ket_qua_A = phuong_phap_A_breadth(group_snapshots)
    ket_qua_B = phuong_phap_B_peak_trough(df_chi_so_dai_dien)
    ket_qua_C = phuong_phap_C_markov_switching(df_chi_so_dai_dien)

    tong_hop = tong_hop_3_phuong_phap(ket_qua_A, ket_qua_B, ket_qua_C)

    return {
        "nhom": ten_nhom,
        "phuong_phap_A_breadth": ket_qua_A["nhan"],
        "phuong_phap_B_peak_trough": ket_qua_B["nhan"],
        "phuong_phap_C_markov": ket_qua_C["nhan"],
        "KET_LUAN_TONG_HOP": tong_hop["nhan_tong_hop"],
        "do_tin_cay": tong_hop["do_tin_cay"],
        "chi_tiet": {"A": ket_qua_A["chi_tiet"], "B": ket_qua_B["chi_tiet"], "C": ket_qua_C["chi_tiet"]},
    }


def phan_tich_ensemble_toan_bo(
    group_snapshots_toan_thi_truong: list[dict],
    df_vnindex: Optional[pd.DataFrame],
    danh_sach_nganh: dict[str, tuple[list[dict], Optional[pd.DataFrame]]],
) -> pd.DataFrame:
    """Hàm chính — trả về DataFrame dạng BẢNG: mỗi dòng 1 nhóm (Toàn thị
    trường + từng ngành), mỗi phương pháp 1 cột, cột cuối là kết luận
    tổng hợp + độ tin cậy.

    `danh_sach_nganh`: {"Ngân hàng": (snapshots, df_chi_so_nganh), ...}
    """
    hang = [phan_tich_ensemble_theo_nhom("Toàn thị trường", group_snapshots_toan_thi_truong, df_vnindex)]
    for ten_nganh, (snapshots, df_chi_so) in danh_sach_nganh.items():
        hang.append(phan_tich_ensemble_theo_nhom(ten_nganh, snapshots, df_chi_so))

    df = pd.DataFrame(hang)
    return df[[
        "nhom", "phuong_phap_A_breadth", "phuong_phap_B_peak_trough",
        "phuong_phap_C_markov", "KET_LUAN_TONG_HOP", "do_tin_cay",
    ]]


# ==============================================================================
# TIỆN ÍCH DÙNG CHUNG — dựng "chỉ số đại diện" PROXY cho 1 nhóm mã (VD 1
# ngành) khi KHÔNG có sẵn chỉ số ngành thật trong hệ thống. Dùng chung
# cho cả `main.py` (lưu trữ hằng ngày) và `dashboard/app.py` (tính live).
# ==============================================================================

def dung_chi_so_dai_dien_tu_gia_dong_cua(danh_sach_gia_dong_cua: list[pd.Series]) -> Optional[pd.DataFrame]:
    """Dựng 1 chỉ số đại diện PROXY từ danh sách chuỗi giá đóng cửa (mỗi
    phần tử là `pd.Series`, index=ngày) — CHUẨN HÓA mỗi chuỗi về gốc 100
    tại điểm đầu tiên, rồi lấy TRUNG BÌNH CỘNG qua các mã — tránh 1 mã
    giá trị tuyệt đối lớn (VD giá 200 nghìn) lấn át mã giá nhỏ khi tính
    trung bình thô. Trả về DataFrame dạng OHLCV (open=high=low=close,
    volume=0) — đủ cấu trúc để truyền vào `phuong_phap_B_peak_trough()`/
    `phuong_phap_C_markov_switching()`.
    """
    danh_sach_hop_le = [s for s in danh_sach_gia_dong_cua if s is not None and len(s) >= 20]
    if not danh_sach_hop_le:
        return None

    danh_sach_chuan_hoa = []
    for s in danh_sach_hop_le:
        gia_dau = s.iloc[0]
        if gia_dau and gia_dau > 0:
            danh_sach_chuan_hoa.append(s / gia_dau * 100)

    if not danh_sach_chuan_hoa:
        return None

    chi_so_trung_binh = pd.concat(danh_sach_chuan_hoa, axis=1).mean(axis=1).dropna()
    if chi_so_trung_binh.empty:
        return None

    return pd.DataFrame({
        "date": chi_so_trung_binh.index, "close": chi_so_trung_binh.values,
        "open": chi_so_trung_binh.values, "high": chi_so_trung_binh.values * 1.001,
        "low": chi_so_trung_binh.values * 0.999, "volume": [0.0] * len(chi_so_trung_binh),
    }).reset_index(drop=True)
