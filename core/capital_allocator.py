"""
capital_allocator.py
=====================
[Giai đoạn 3 — Khuyến nghị phân bổ vốn]

CHỈ ĐƯA RA KHUYẾN NGHỊ THAM KHẢO — KHÔNG tự thực thi bất kỳ lệnh nào.
Toàn bộ output của module này mang tính chất tham khảo trên danh mục MÔ
PHỎNG, không phải lời khuyên đầu tư.

BẢNG TỶ TRỌNG (giữ nguyên theo yêu cầu dự án — KHÔNG chỉnh sửa số liệu):
    - Uptrend:   70-100%, giải ngân chia 3 đợt 30%-50%-20% (pyramiding),
                 ưu tiên nhóm ngành dẫn dắt (ngân hàng, chứng khoán, bất
                 động sản — trừ khi ngành đó đang bị gắn cờ "thận trọng").
    - Downtrend: 10-30%, ưu tiên nhóm phòng thủ (tiêu dùng thiết yếu, y
                 tế, dược phẩm); KHÔNG khuyến nghị bắt đáy khi chưa có tín
                 hiệu phân kỳ tăng rõ ràng.
    - Sideway:   30-50%, lướt biên độ vốn hạn chế, phần còn lại ưu tiên
                 cổ phiếu giá trị/cổ tức cao hoặc giữ tiền mặt.

NGUYÊN TẮC QUẢN TRỊ RỦI RO (giữ nguyên): rủi ro mỗi lệnh không quá 2%
NAV mô phỏng, rủi ro toàn danh mục không quá 20% NAV mô phỏng.

YÊU CẦU BẮT BUỘC (điểm quan trọng nhất của module): MỖI khuyến nghị PHẢI
kèm theo một KHOẢNG GIÁ VÀO LỆNH (entry_price_range) cụ thể — không phải
một mức giá đơn lẻ. Nếu giá thị trường hiện tại đã vượt ra ngoài khoảng
này, module PHẢI cảnh báo "không còn vùng entry hợp lệ" thay vì vẫn đưa
khuyến nghị mua.
"""

from __future__ import annotations

from typing import Optional

VALID_REGIMES = {"uptrend", "downtrend", "sideway"}

DEFAULT_CONFIG = {
    "risk_per_trade_pct": 2.0,
    "risk_total_portfolio_pct": 20.0,
    "allocation": {
        "uptrend": {"target_pct_range": (70.0, 100.0), "tranches_pct": [30, 50, 20]},
        "downtrend": {"target_pct_range": (10.0, 30.0), "tranches_pct": [100]},
        "sideway": {"target_pct_range": (30.0, 50.0), "tranches_pct": [100]},
    },
    # % dưới cận dưới entry_price_range dùng làm mức cắt lỗ mặc định, nếu
    # không có tham số `stop_loss_pct` truyền riêng.
    "default_stop_loss_pct": 7.0,
    "leading_sectors": ["banking", "securities", "real_estate"],
    "defensive_sectors": ["consumer_staples", "healthcare", "pharma"],
}


class InvalidAllocationInputError(ValueError):
    """Dữ liệu đầu vào không hợp lệ để tính khuyến nghị phân bổ vốn."""


# ==============================================================================
# CÁC HÀM TÍNH TOÁN NHỎ (tách riêng để dễ test độc lập)
# ==============================================================================

def compute_stop_loss(entry_low: float, stop_loss_pct: float) -> float:
    """Tính mức cắt lỗ dựa trên % dưới CẬN DƯỚI của vùng giá vào lệnh.

    Mức cắt lỗ chỉ có ý nghĩa nếu lệnh được vào ĐÚNG điểm entry theo hệ
    thống — vì vậy luôn neo theo `entry_low` (kịch bản entry tệ nhất
    trong vùng được khuyến nghị), không neo theo giá thị trường hiện tại.
    """
    if entry_low <= 0:
        raise InvalidAllocationInputError("entry_low phải > 0.")
    if stop_loss_pct <= 0:
        raise InvalidAllocationInputError("stop_loss_pct phải > 0.")
    return entry_low * (1 - stop_loss_pct / 100.0)


def compute_max_position_size(
    nav: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss: float,
    capital_budget: Optional[float] = None,
) -> int:
    """Tính khối lượng tối đa được phép mua, tuân thủ ĐỒNG THỜI 2 giới hạn:

    1. Rủi ro mỗi lệnh không vượt quá `risk_per_trade_pct`% NAV — tức số
       tiền có thể mất nếu giá chạm cắt lỗ không vượt quá ngưỡng này.
    2. Không vượt quá `capital_budget` (nếu có truyền) — số vốn thực tế
       được phân bổ cho lệnh này theo tỷ trọng/đợt giải ngân.

    `entry_price` nên dùng CẬN TRÊN của entry_price_range (kịch bản vào
    lệnh ở mức giá cao nhất được chấp nhận trong vùng khuyến nghị) để tính
    toán theo hướng THẬN TRỌNG (bảo thủ) hơn là lạc quan.
    """
    if entry_price <= stop_loss:
        raise InvalidAllocationInputError(
            "entry_price phải LỚN HƠN stop_loss để có thể tính rủi ro/lệnh."
        )

    risk_amount = nav * (risk_per_trade_pct / 100.0)
    risk_per_share = entry_price - stop_loss
    max_qty_by_risk = int(risk_amount // risk_per_share)

    if capital_budget is not None:
        max_qty_by_capital = int(capital_budget // entry_price)
        return max(0, min(max_qty_by_risk, max_qty_by_capital))

    return max(0, max_qty_by_risk)


# ==============================================================================
# HÀM CHÍNH — get_allocation_recommendation
# ==============================================================================

def get_allocation_recommendation(
    regime_result: dict,
    nav: float,
    signal_price_context: dict,
    config: Optional[dict] = None,
    existing_portfolio_risk_pct: float = 0.0,
) -> dict:
    """Đưa ra khuyến nghị phân bổ vốn tham khảo cho MỘT cơ hội giao dịch cụ
    thể, dựa trên giai đoạn thị trường đã xác định (từ
    market_regime_detector) và bối cảnh giá hiện tại.

    Tham số:
        regime_result: dict từ `market_regime_detector.detect_market_regime()`,
            tối thiểu cần khóa 'regime' ("uptrend"|"downtrend"|"sideway"|None)
            và 'confidence' (float). Nếu 'regime' là None (chưa được xác
            nhận qua độ trễ), hàm trả về khuyến nghị KHÔNG giao dịch.
        nav: giá trị tài sản ròng (NAV) của danh mục mô phỏng hiện tại.
        signal_price_context: dict gồm:
            - current_price (bắt buộc): giá thị trường hiện tại.
            - entry_low, entry_high (bắt buộc): vùng giá vào lệnh đề xuất
              (ví dụ lấy từ `accumulation_high` của pattern_detector làm
              tham chiếu cận trên, hoặc do người dùng tự xác định).
            - sector (tùy chọn): ngành của mã đang xét — dùng để kiểm tra
              ưu tiên nhóm dẫn dắt/phòng thủ và đối chiếu affected_sectors.
            - has_bullish_divergence (tùy chọn, mặc định False): CHỈ có ý
              nghĩa khi regime="downtrend" — xác nhận có tín hiệu phân kỳ
              tăng rõ ràng hay chưa, làm căn cứ có nên "bắt đáy" hay không.
            - stop_loss_pct (tùy chọn): override % cắt lỗ mặc định.
        existing_portfolio_risk_pct: % NAV đã bị "chiếm dụng" bởi rủi ro
            từ các vị thế khác đang mở — dùng để cảnh báo khi tổng rủi ro
            toàn danh mục có nguy cơ vượt ngưỡng 20% NAV nếu thêm lệnh này.

    Trả về dict:
        {"target_pct": float, "tranches": [...], "entry_price_range":
         {"low":..., "high":...}, "stop_loss": float|None,
         "max_position_size": int, "notes": [...]}

    LƯU Ý: Toàn bộ output chỉ mang tính khuyến nghị tham khảo trên danh
    mục mô phỏng — KHÔNG phải lời khuyên đầu tư, và KHÔNG tự động thực
    thi bất kỳ giao dịch nào.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    notes: list[str] = []

    # --- Xác thực đầu vào cơ bản ---
    if nav <= 0:
        raise InvalidAllocationInputError("nav phải > 0.")

    required_keys = {"current_price", "entry_low", "entry_high"}
    missing = required_keys - set(signal_price_context.keys())
    if missing:
        raise InvalidAllocationInputError(
            f"signal_price_context thiếu các khóa bắt buộc: {sorted(missing)}."
        )

    current_price = signal_price_context["current_price"]
    entry_low = signal_price_context["entry_low"]
    entry_high = signal_price_context["entry_high"]
    sector = signal_price_context.get("sector")
    has_bullish_divergence = signal_price_context.get("has_bullish_divergence", False)
    stop_loss_pct = signal_price_context.get("stop_loss_pct", cfg["default_stop_loss_pct"])

    if entry_low <= 0 or entry_high <= 0:
        raise InvalidAllocationInputError("entry_low/entry_high phải > 0.")
    if entry_low >= entry_high:
        raise InvalidAllocationInputError("entry_low phải NHỎ HƠN entry_high.")

    entry_price_range = {"low": entry_low, "high": entry_high}

    # --- Trường hợp giai đoạn thị trường chưa được xác nhận ---
    regime = regime_result.get("regime")
    if regime is None:
        return {
            "target_pct": 0.0,
            "tranches": [],
            "entry_price_range": entry_price_range,
            "stop_loss": None,
            "max_position_size": 0,
            "notes": [
                "Giai đoạn thị trường CHƯA được xác nhận (đang chờ tín hiệu "
                "ổn định qua đủ số phiên) -> tạm thời KHÔNG đưa khuyến nghị "
                "giải ngân."
            ],
        }

    if regime not in VALID_REGIMES:
        raise InvalidAllocationInputError(
            f"regime '{regime}' không hợp lệ. Cần một trong {VALID_REGIMES}."
        )

    allocation_cfg = cfg["allocation"][regime]
    range_low, range_high = allocation_cfg["target_pct_range"]
    tranches = list(allocation_cfg["tranches_pct"])

    confidence = max(0.0, min(1.0, regime_result.get("confidence", 0.5)))
    # Confidence càng cao -> tỷ trọng khuyến nghị càng gần cận trên của
    # khoảng cho phép trong bảng tỷ trọng.
    target_pct = range_low + (range_high - range_low) * confidence

    # --- Kiểm tra vùng entry còn hợp lệ so với giá thị trường hiện tại ---
    entry_still_valid = current_price <= entry_high
    if not entry_still_valid:
        notes.append(
            f"Giá thị trường hiện tại ({current_price:,.2f}) đã VƯỢT RA NGOÀI "
            f"vùng entry khuyến nghị (tối đa {entry_high:,.2f}) -> KHÔNG CÒN "
            f"VÙNG ENTRY HỢP LỆ. Không khuyến nghị mua thêm ở mức giá hiện tại."
        )
    elif current_price < entry_low:
        notes.append(
            f"Giá thị trường hiện tại ({current_price:,.2f}) còn THẤP HƠN vùng "
            f"entry khuyến nghị (tối thiểu {entry_low:,.2f}) -> có thể chờ giá "
            f"về đúng vùng trước khi giải ngân."
        )

    # --- Ưu tiên ngành theo từng giai đoạn ---
    affected_sectors = regime_result.get("affected_sectors", [])

    if regime == "uptrend":
        if sector in cfg["leading_sectors"] and sector not in affected_sectors:
            notes.append(
                f"Ngành '{sector}' thuộc nhóm ngành dẫn dắt trong Uptrend -> "
                f"ưu tiên tìm cơ hội giải ngân."
            )
        elif sector in cfg["leading_sectors"] and sector in affected_sectors:
            notes.append(
                f"Ngành '{sector}' thuộc nhóm dẫn dắt nhưng đang bị vĩ mô gắn "
                f"cờ THẬN TRỌNG -> KHÔNG áp dụng ưu tiên đặc biệt cho ngành này."
            )
        notes.append(
            f"Uptrend: khuyến nghị giải ngân theo {len(tranches)} đợt "
            f"({'-'.join(str(t) + '%' for t in tranches)}), tăng dần theo đà thắng."
        )
    elif regime == "downtrend":
        if sector in cfg["defensive_sectors"]:
            notes.append(
                f"Ngành '{sector}' thuộc nhóm phòng thủ -> phù hợp ưu tiên "
                f"trong giai đoạn Downtrend."
            )
        if not has_bullish_divergence:
            notes.append(
                "Downtrend: CHƯA có tín hiệu phân kỳ tăng rõ ràng -> KHÔNG "
                "khuyến nghị 'bắt đáy' ở thời điểm hiện tại."
            )
            target_pct = 0.0
            entry_still_valid = False
        else:
            notes.append(
                "Downtrend: đã ghi nhận tín hiệu phân kỳ tăng -> có thể cân "
                "nhắc giải ngân thận trọng, tỷ trọng thấp."
            )
    else:  # sideway
        notes.append(
            "Sideway: giao dịch lướt biên độ với vốn hạn chế; phần vốn còn "
            "lại nên ưu tiên cổ phiếu giá trị/cổ tức cao hoặc giữ tiền mặt."
        )

    # --- Nếu vùng entry không còn hợp lệ (do giá đã vượt) -> không cấp
    #     khối lượng mua mới, nhưng vẫn trả về đầy đủ thông tin tham khảo.
    if not entry_still_valid and regime != "downtrend":
        target_pct = 0.0

    stop_loss = compute_stop_loss(entry_low, stop_loss_pct)

    if target_pct <= 0:
        max_position_size = 0
    else:
        tranche1_pct_of_target = tranches[0] / 100.0
        capital_budget = nav * (target_pct / 100.0) * tranche1_pct_of_target
        max_position_size = compute_max_position_size(
            nav=nav,
            risk_per_trade_pct=cfg["risk_per_trade_pct"],
            entry_price=entry_high,  # kịch bản thận trọng: vào ở cận trên
            stop_loss=stop_loss,
            capital_budget=capital_budget,
        )

    # --- Cảnh báo rủi ro toàn danh mục ---
    projected_total_risk_pct = existing_portfolio_risk_pct + cfg["risk_per_trade_pct"]
    if max_position_size > 0 and projected_total_risk_pct > cfg["risk_total_portfolio_pct"]:
        notes.append(
            f"CẢNH BÁO: nếu thực hiện lệnh này, tổng rủi ro danh mục ước tính "
            f"đạt {projected_total_risk_pct:.1f}% NAV, VƯỢT ngưỡng an toàn "
            f"{cfg['risk_total_portfolio_pct']:.0f}% NAV -> cân nhắc giảm khối "
            f"lượng hoặc bỏ qua lệnh này."
        )

    return {
        "target_pct": round(target_pct, 2),
        "tranches": tranches,
        "entry_price_range": entry_price_range,
        "stop_loss": round(stop_loss, 4),
        "max_position_size": max_position_size,
        "notes": notes,
    }
