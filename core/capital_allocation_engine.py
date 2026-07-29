"""
capital_allocation_engine.py
==============================
[Bổ sung — Module Khuyến nghị Phân bổ Vốn theo Giai đoạn Thị trường]

Nối tiếp module xác định giai đoạn thị trường (`market_regime_detector`),
nhận trạng thái UPTREND/DOWNTREND/SIDEWAY + độ tin cậy + breadth theo
ngành làm input, kết hợp NAV mô phỏng + watchlist, xuất ra khuyến nghị
phân bổ vốn có cấu trúc: tỷ trọng, danh sách mã theo đợt giải ngân,
khoảng giá vào lệnh (dựa trên ATR14), cắt lỗ, chốt lời, khối lượng.

NGUYÊN TẮC BẮT BUỘC: đây là module tạo KHUYẾN NGHỊ THAM KHẢO có cấu
trúc — KHÔNG PHẢI lệnh giao dịch tự động. Mọi lệnh đều do người dùng tự
xác nhận và đặt thủ công. Toàn bộ số giá trong ví dụ minh họa (nếu có)
chỉ để minh họa công thức — khi dùng thật, phải lấy dữ liệu giá/ATR/hỗ
trợ/kháng cự từ thị trường thời gian thực, KHÔNG hardcode.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

Strategy = Literal["pullback", "breakout", "support"]


class InvalidCapitalAllocationError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module phân bổ vốn."""


# ==============================================================================
# BẢNG PHÂN BỔ THEO GIAI ĐOẠN (mục 2 tài liệu — giữ nguyên không đổi số liệu)
# ==============================================================================

ALLOCATION_TABLE = {
    "UPTREND": {
        "co_phieu_range": (0.70, 1.00),
        "dot_giai_ngan": [0.30, 0.50, 0.20],
        "nganh_uu_tien": ["Ngân hàng", "Chứng khoán", "Bất động sản"],
    },
    "DOWNTREND": {
        "co_phieu_range": (0.10, 0.30),
        "dot_giai_ngan": None,  # không chia đợt cố định — chỉ giải ngân khi có phân kỳ tăng
        "nganh_uu_tien": ["Tiêu dùng thiết yếu", "Y tế", "Dược phẩm"],
    },
    "SIDEWAY": {
        "co_phieu_range": (0.30, 0.50),
        "dot_giai_ngan": [0.50, 0.50],
        "nganh_uu_tien": ["Giá trị / Cổ tức cao"],
    },
}

CONFIDENCE_WEIGHTS = {"CAO": 1.0, "TRUNG_BINH": 0.5, "THAP": 0.0}

ENTRY_RANGE_K = 0.3        # hệ số ATR mặc định cho biên độ entry tổng quát
STOP_LOSS_ATR_OFFSET = 0.2
TAKE_PROFIT_ATR_OFFSET = 0.3


# ==============================================================================
# BƯỚC 1 — TỶ TRỌNG CỔ PHIẾU TỔNG THỂ (nội suy theo độ tin cậy)
# ==============================================================================

def calculate_stock_allocation_pct(trang_thai: str, do_tin_cay: str) -> float:
    """Nội suy tỷ trọng cổ phiếu khuyến nghị trong khoảng [min, max] của
    giai đoạn, theo độ tin cậy: CAO -> cận trên, THẤP -> cận dưới,
    TRUNG_BÌNH -> giữa khoảng.
    """
    if trang_thai not in ALLOCATION_TABLE:
        raise InvalidCapitalAllocationError(
            f"trang_thai '{trang_thai}' không hợp lệ. Cần một trong {list(ALLOCATION_TABLE)}."
        )
    if do_tin_cay not in CONFIDENCE_WEIGHTS:
        raise InvalidCapitalAllocationError(
            f"do_tin_cay '{do_tin_cay}' không hợp lệ. Cần một trong {list(CONFIDENCE_WEIGHTS)}."
        )

    low, high = ALLOCATION_TABLE[trang_thai]["co_phieu_range"]
    weight = CONFIDENCE_WEIGHTS[do_tin_cay]
    return low + (high - low) * weight


# ==============================================================================
# BƯỚC 4.2-4.3 — KHOẢNG GIÁ VÀO LỆNH / CẮT LỖ / CHỐT LỜI (dựa trên ATR14)
# ==============================================================================

def calculate_entry_price_range(
    reference_price: float,
    atr14: float,
    strategy: Strategy = "breakout",
    support_level: Optional[float] = None,
) -> tuple[float, float]:
    """Tính khoảng giá vào lệnh theo chiến lược (mục 4.2):

        pullback  -> [ref - 0.6*ATR, ref - 0.1*ATR]  (chờ giá hồi mới vào)
        breakout  -> [ref + 0.05*ATR, ref + 0.4*ATR] (xác nhận đã phá kháng cự)
        support   -> [hỗ trợ - 0.2*ATR, hỗ trợ + 0.2*ATR]
    """
    if reference_price <= 0:
        raise InvalidCapitalAllocationError("reference_price phải > 0.")
    if atr14 <= 0:
        raise InvalidCapitalAllocationError("atr14 phải > 0.")

    if strategy == "pullback":
        return (reference_price - 0.6 * atr14, reference_price - 0.1 * atr14)
    if strategy == "breakout":
        return (reference_price + 0.05 * atr14, reference_price + 0.4 * atr14)
    if strategy == "support":
        base = support_level if support_level is not None else reference_price
        return (base - 0.2 * atr14, base + 0.2 * atr14)

    raise InvalidCapitalAllocationError(
        f"strategy '{strategy}' không hợp lệ. Cần 'pullback' | 'breakout' | 'support'."
    )


def calculate_stop_loss_range(support_level: float, atr14: float) -> tuple[float, float]:
    """Khoảng cắt lỗ (mục 4.3): [hỗ trợ - 0.2*ATR, hỗ trợ]."""
    if support_level <= 0:
        raise InvalidCapitalAllocationError("support_level phải > 0.")
    return (support_level - STOP_LOSS_ATR_OFFSET * atr14, support_level)


def calculate_take_profit_range(resistance_level: float, atr14: float) -> tuple[float, float]:
    """Khoảng chốt lời tham khảo (mục 4.3): [kháng cự, kháng cự + 0.3*ATR]."""
    if resistance_level <= 0:
        raise InvalidCapitalAllocationError("resistance_level phải > 0.")
    return (resistance_level, resistance_level + TAKE_PROFIT_ATR_OFFSET * atr14)


# ==============================================================================
# BƯỚC 4.4 — KHỐI LƯỢNG MUA (dùng kịch bản XẤU NHẤT trong khoảng giá)
# ==============================================================================

def round_to_lot(qty: float, lot_size: int = 100) -> int:
    """Làm tròn XUỐNG theo lô giao dịch HOSE (mặc định lô 100 cổ phiếu)."""
    if qty < lot_size:
        return 0
    return int(qty // lot_size) * lot_size


def calculate_position_size(
    nav: float,
    risk_per_trade_pct: float,
    entry_price_range: tuple[float, float],
    stop_loss_range: tuple[float, float],
    capital_budget: Optional[float] = None,
    lot_size: int = 100,
) -> int:
    """Tính khối lượng mua tối đa, dùng kịch bản XẤU NHẤT trong khoảng giá
    (mục 4.4): giá vào ở cận CAO, cắt lỗ ở cận THẤP — đảm bảo KHÔNG BAO
    GIỜ vượt quá `risk_per_trade_pct` NAV dù giá khớp ở bất kỳ điểm nào
    trong khoảng entry.

    Đồng thời giới hạn thêm bởi `capital_budget` (nếu có) — lấy giá trị
    NHỎ HƠN giữa 2 giới hạn (rủi ro và ngân sách vốn), rồi làm tròn theo
    lô giao dịch.
    """
    entry_max = max(entry_price_range)
    stop_min = min(stop_loss_range)

    if entry_max <= stop_min:
        raise InvalidCapitalAllocationError(
            "Giá vào lệnh (cận cao) phải LỚN HƠN giá cắt lỗ (cận thấp)."
        )

    risk_amount = nav * risk_per_trade_pct
    risk_per_share = entry_max - stop_min
    qty_by_risk = risk_amount / risk_per_share

    if capital_budget is not None:
        entry_avg = sum(entry_price_range) / 2
        qty_by_budget = capital_budget / entry_avg
        final_qty = min(qty_by_risk, qty_by_budget)
    else:
        final_qty = qty_by_risk

    return round_to_lot(final_qty, lot_size)


# ==============================================================================
# PHÂN BỔ VỐN THEO % BREADTH NGÀNH (tỷ trọng tương đối giữa các mã)
# ==============================================================================

def allocate_capital_by_breadth(
    total_capital: float, symbols_breadth: dict[str, float]
) -> dict[str, float]:
    """Chia `total_capital` cho các mã theo tỷ trọng breadth ngành tương
    đối (mục Bước 3 ví dụ tài liệu): mã thuộc ngành có breadth cao hơn
    được phân bổ vốn nhiều hơn.
    """
    if not symbols_breadth:
        raise InvalidCapitalAllocationError("symbols_breadth không được rỗng.")

    total_breadth = sum(symbols_breadth.values())
    if total_breadth <= 0:
        # Breadth bằng 0 ở mọi mã -> chia đều
        n = len(symbols_breadth)
        return {symbol: total_capital / n for symbol in symbols_breadth}

    return {
        symbol: total_capital * (breadth / total_breadth)
        for symbol, breadth in symbols_breadth.items()
    }


def find_support_resistance(df, lookback: int = 60) -> tuple[float, float]:
    """Tự động xác định mức hỗ trợ/kháng cự gần nhất từ dữ liệu OHLCV
    (mục 8.1 tài liệu: "hỗ trợ/kháng cự = đáy/đỉnh swing gần nhất trong N
    phiên").

    Cách làm ĐƠN GIẢN (đủ dùng cho mục đích tính entry/stop/target ở đây):
        - Hỗ trợ = giá THẤP NHẤT trong `lookback` phiên gần nhất.
        - Kháng cự = giá CAO NHẤT trong `lookback` phiên gần nhất.

    (Có thể nâng cấp sau này bằng thuật toán pivot point/swing point phức
    tạp hơn nếu cần độ chính xác cao hơn — đây là bản khởi điểm hợp lý.)
    """
    if len(df) == 0:
        raise InvalidCapitalAllocationError("df không được rỗng.")

    recent = df.tail(lookback)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance


# ==============================================================================
# HÀM CHÍNH — tổng hợp toàn bộ khuyến nghị phân bổ vốn
# ==============================================================================

def calculate_capital_allocation(
    trang_thai: str,
    do_tin_cay: str,
    breadth_theo_nganh: dict[str, float],
    nav: float,
    watchlist: list[dict],
    risk_per_trade_pct: float = 0.02,
    risk_total_pct: float = 0.20,
    lot_size: int = 100,
) -> dict:
    """Tính khuyến nghị phân bổ vốn ĐẦY ĐỦ theo giai đoạn thị trường.

    `watchlist`: danh sách dict, mỗi mã cần có:
        {"ma": str, "nganh": str, "atr14": float, "gia_tham_chieu": float,
         "chien_luoc": "pullback"|"breakout"|"support",
         "ho_tro": float, "khang_cu": float,
         "co_phan_ky_tang": bool (chỉ cần khi trang_thai="DOWNTREND")}

    Trả về dict theo đúng cấu trúc OUTPUT ở mục 7 tài liệu (rút gọn tên
    khóa sang tiếng Anh/snake_case cho nhất quán với phần code còn lại
    của dự án, giữ nguyên Ý NGHĨA và cấu trúc lồng nhau).

    KHÔNG tự động thực thi bất kỳ lệnh nào — chỉ trả về khuyến nghị.
    """
    if nav <= 0:
        raise InvalidCapitalAllocationError("nav phải > 0.")
    if not watchlist:
        raise InvalidCapitalAllocationError("watchlist không được rỗng.")

    canh_bao: list[str] = []

    # --- Bước 1: tỷ trọng cổ phiếu tổng thể ---
    ty_trong_co_phieu = calculate_stock_allocation_pct(trang_thai, do_tin_cay)
    von_co_phieu = nav * ty_trong_co_phieu
    von_tien_mat = nav - von_co_phieu

    cfg = ALLOCATION_TABLE[trang_thai]

    # --- Bước 2: lọc watchlist theo ngành ưu tiên ---
    ma_uu_tien = [m for m in watchlist if m.get("nganh") in cfg["nganh_uu_tien"]]
    if not ma_uu_tien:
        # Không có mã nào khớp ngành ưu tiên -> vẫn dùng toàn bộ watchlist,
        # nhưng cảnh báo rõ để người dùng biết đang lệch khỏi khuyến nghị ngành.
        canh_bao.append(
            f"Không có mã nào trong watchlist thuộc nhóm ngành ưu tiên của "
            f"{trang_thai} ({', '.join(cfg['nganh_uu_tien'])}) -> dùng toàn bộ watchlist."
        )
        ma_uu_tien = list(watchlist)

    # --- Bước 3: Downtrend — bắt buộc có phân kỳ tăng mới cho phép đề xuất mua ---
    if trang_thai == "DOWNTREND":
        ma_uu_tien = [m for m in ma_uu_tien if m.get("co_phan_ky_tang", False)]
        if not ma_uu_tien:
            return {
                "nav_mo_phong": nav,
                "trang_thai_thi_truong": trang_thai,
                "ty_trong_co_phieu_khuyen_nghi": 0.0,
                "von_tien_mat_du_phong": nav,
                "cac_dot_giai_ngan": [],
                "tong_rui_ro_danh_muc_hien_tai_pct": 0.0,
                "canh_bao": [
                    "KHÔNG GIẢI NGÂN — chưa có tín hiệu phân kỳ tăng (Bullish "
                    "Divergence) xác nhận trên bất kỳ mã nào trong watchlist."
                ],
                "ghi_chu": "Downtrend: không bắt đáy khi chưa có phân kỳ tăng rõ ràng.",
            }

    # --- Bước 4: chia theo đợt giải ngân, tính entry/stop/target/khối lượng từng mã ---
    dot_giai_ngan_ty_le = cfg["dot_giai_ngan"] or [1.0]
    breadth_theo_ma = {
        m["ma"]: breadth_theo_nganh.get(m.get("nganh"), 0.0) for m in ma_uu_tien
    }

    cac_dot_giai_ngan = []
    tong_rui_ro_tuyet_doi = 0.0

    for dot_idx, ty_le_dot in enumerate(dot_giai_ngan_ty_le, start=1):
        von_dot = von_co_phieu * ty_le_dot
        phan_bo_von = allocate_capital_by_breadth(von_dot, breadth_theo_ma)

        danh_sach_ma = []
        for ma_info in ma_uu_tien:
            symbol = ma_info["ma"]
            von_ma = phan_bo_von[symbol]

            entry_range = calculate_entry_price_range(
                ma_info["gia_tham_chieu"], ma_info["atr14"],
                strategy=ma_info.get("chien_luoc", "breakout"),
                support_level=ma_info.get("ho_tro"),
            )
            stop_loss_range = calculate_stop_loss_range(ma_info["ho_tro"], ma_info["atr14"])
            take_profit_range = calculate_take_profit_range(ma_info["khang_cu"], ma_info["atr14"])

            khoi_luong = calculate_position_size(
                nav=nav, risk_per_trade_pct=risk_per_trade_pct,
                entry_price_range=entry_range, stop_loss_range=stop_loss_range,
                capital_budget=von_ma, lot_size=lot_size,
            )

            if khoi_luong <= 0:
                canh_bao.append(
                    f"[{symbol}] Vốn phân bổ ({von_ma:,.0f}) không đủ mua tối "
                    f"thiểu 1 lô ({lot_size} cổ phiếu) ở vùng giá khuyến nghị."
                )
                continue

            rui_ro_tren_cp = max(entry_range) - min(stop_loss_range)
            rui_ro_thuc_te = khoi_luong * rui_ro_tren_cp
            tong_rui_ro_tuyet_doi += rui_ro_thuc_te

            danh_sach_ma.append({
                "ma": symbol,
                "nganh": ma_info.get("nganh"),
                "von_phan_bo": round(von_ma, 0),
                "khoang_gia_vao_lenh": [round(entry_range[0], 2), round(entry_range[1], 2)],
                "khoi_luong_du_kien": khoi_luong,
                "khoang_cat_lo": [round(stop_loss_range[0], 2), round(stop_loss_range[1], 2)],
                "khoang_chot_loi_tham_khao": [
                    round(take_profit_range[0], 2), round(take_profit_range[1], 2)
                ],
                "ty_le_rui_ro_tren_nav": round(rui_ro_thuc_te / nav, 4),
            })

        cac_dot_giai_ngan.append({
            "dot": dot_idx,
            "ty_le_dot": ty_le_dot,
            "von_dot": round(von_dot, 0),
            "danh_sach_ma": danh_sach_ma,
        })

    # --- Bước 5: kiểm tra tổng rủi ro danh mục không vượt ngưỡng ---
    tong_rui_ro_pct = tong_rui_ro_tuyet_doi / nav
    if tong_rui_ro_pct > risk_total_pct:
        canh_bao.append(
            f"CẢNH BÁO: tổng rủi ro danh mục ước tính ({tong_rui_ro_pct * 100:.1f}%) "
            f"VƯỢT ngưỡng an toàn {risk_total_pct * 100:.0f}% NAV -> cân nhắc giảm "
            f"khối lượng hoặc số lệnh đồng thời."
        )

    return {
        "nav_mo_phong": nav,
        "trang_thai_thi_truong": trang_thai,
        "ty_trong_co_phieu_khuyen_nghi": round(ty_trong_co_phieu, 4),
        "von_tien_mat_du_phong": round(von_tien_mat, 0),
        "cac_dot_giai_ngan": cac_dot_giai_ngan,
        "tong_rui_ro_danh_muc_hien_tai_pct": round(tong_rui_ro_pct, 4),
        "canh_bao": canh_bao,
        "ghi_chu": (
            "Đợt 2, 3 (nếu có) chỉ nên kích hoạt khi đợt trước đang có lãi và "
            "trạng thái thị trường còn được duy trì tại thời điểm đánh giá lại — "
            "KHÔNG tự động giải ngân theo lịch cố định."
        ),
    }
