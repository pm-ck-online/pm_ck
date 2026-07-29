"""
market_regime_detector.py
==========================
[Giai đoạn 3 — Nhận diện giai đoạn thị trường]

Đây là module PHỨC TẠP NHẤT trong hệ thống. Xác định giai đoạn thị trường
theo ĐÚNG TRÌNH TỰ đã chốt — KHÔNG đảo ngược thứ tự này:

    BƯỚC 1 — Bối cảnh VĨ MÔ trước tiên (lớp lọc đầu tiên).
    BƯỚC 2 — Vị trí giá so với EMA200 theo từng nhóm ngành/mã cụ thể.
    BƯỚC 3 — Chỉ báo phụ trợ (lớp xác nhận bổ sung, KHÔNG phải lớp quyết
             định chính).
    + Độ trễ xác nhận (confirmation lag) để chống nhiễu.

THIẾT KẾ MODULE: tách thành các hàm nhỏ, độc lập, mỗi hàm chịu trách
nhiệm đúng MỘT bước trong trình tự trên — để có thể viết unit test riêng
cho từng bước trước khi test tổ hợp cả hệ thống, theo đúng yêu cầu dự án:

    - `evaluate_macro_filter()`      -> chỉ Bước 1 (vĩ mô)
    - `classify_sector_trend()`      -> chỉ Bước 2 (EMA200 theo ngành)
    - `apply_confirmation_lag()`     -> chỉ cơ chế chống nhiễu
    - `detect_market_regime()`       -> hàm tổ hợp chính, kết hợp cả 3

Hàm `detect_market_regime()` đánh giá MỘT ngành/nhóm cổ phiếu cụ thể tại
một thời điểm (gọi lại nhiều lần cho nhiều ngành nếu cần). Trả về:
    {"regime": "uptrend"|"downtrend"|"sideway", "confidence": float,
     "reasoning": [...], "affected_sectors": [...]}
"""

from __future__ import annotations

from typing import Optional

from core.data_collector import MacroDataPoint

# ==============================================================================
# NGƯỠNG MẶC ĐỊNH (có thể override qua tham số `config` của từng hàm)
# ==============================================================================

DEFAULT_CONFIG = {
    # Bước 2 — phân loại theo % mã trong ngành đang ở trên EMA200
    "uptrend_threshold_pct": 0.6,     # >= 60% mã trên EMA200 -> Uptrend
    "downtrend_threshold_pct": 0.4,   # <= 40% mã trên EMA200 -> Downtrend
    # Nếu khoảng cách trung bình |close - EMA200| quá nhỏ -> Sideway (giá
    # xoay quanh trục, không tách xa rõ rệt theo hướng nào) bất kể tỷ lệ
    # trên/dưới EMA200 nghiêng về đâu.
    "sideway_distance_threshold_pct": 3.0,
    # Độ trễ xác nhận (Bước ngoài trình tự 3 bước, áp dụng xuyên suốt)
    "confirmation_lag_sessions": 3,
}

VALID_REGIMES = {"uptrend", "downtrend", "sideway"}


# ==============================================================================
# BƯỚC 1 — BỐI CẢNH VĨ MÔ (lớp lọc đầu tiên, ưu tiên cao nhất)
# ==============================================================================

def evaluate_macro_filter(macro_points: list[MacroDataPoint]) -> dict:
    """Đánh giá bối cảnh vĩ mô — LỚP LỌC ĐẦU TIÊN trước khi xét kỹ thuật.

    Một ngành được gắn cờ "thận trọng" (caution) nếu có ít nhất một điểm
    dữ liệu vĩ mô với `direction == "tightening"` (thắt chặt/tiêu cực) mà
    ngành đó nằm trong `affected_sectors` của điểm dữ liệu đó.

    Trả về dict:
        - caution_sectors: set các ngành đang bị gắn cờ thận trọng.
        - overall_bias: "tightening" | "easing" | "neutral" — xu hướng vĩ
          mô tổng thể (dựa trên direction chiếm đa số trong toàn bộ điểm
          dữ liệu, không gắn với ngành cụ thể nào).
        - reasoning: danh sách mô tả rõ từng điểm dữ liệu vĩ mô đã góp
          phần vào kết luận, để dễ audit lại.
    """
    caution_sectors: set[str] = set()
    reasoning: list[str] = []

    direction_counts = {"tightening": 0, "easing": 0, "neutral": 0}

    for point in macro_points:
        direction = point.direction or "neutral"
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

        if direction == "tightening" and point.affected_sectors:
            caution_sectors.update(point.affected_sectors)
            reasoning.append(
                f"[Vĩ mô - {point.category}] {point.description} "
                f"-> gắn cờ thận trọng cho ngành: {', '.join(point.affected_sectors)}."
            )
        elif direction == "tightening":
            reasoning.append(
                f"[Vĩ mô - {point.category}] {point.description} "
                f"-> tiêu cực chung, không chỉ định ngành cụ thể."
            )

    if direction_counts["tightening"] > direction_counts["easing"]:
        overall_bias = "tightening"
    elif direction_counts["easing"] > direction_counts["tightening"]:
        overall_bias = "easing"
    else:
        overall_bias = "neutral"

    if not reasoning:
        reasoning.append("Không có tín hiệu vĩ mô tiêu cực nào đáng chú ý.")

    return {
        "caution_sectors": caution_sectors,
        "overall_bias": overall_bias,
        "reasoning": reasoning,
    }


# ==============================================================================
# BƯỚC 2 — VỊ TRÍ GIÁ SO VỚI EMA200 THEO NGÀNH (lớp quyết định kỹ thuật)
# ==============================================================================

def classify_sector_trend(
    snapshots: list[dict], config: Optional[dict] = None
) -> dict:
    """Phân loại xu hướng kỹ thuật của MỘT ngành/nhóm cổ phiếu, dựa trên vị
    trí giá so với EMA200 của từng mã trong ngành (theo đúng nguyên tắc Bước 2).

    `snapshots`: danh sách dict, mỗi phần tử là kết quả từ
        `core.indicators.get_indicator_snapshot()` cho MỘT mã cổ phiếu
        trong ngành — cần tối thiểu các khóa: 'symbol' (tùy chọn),
        'close', 'ema200', 'price_above_ema200'.

    Nguyên tắc phân loại:
        - >= `uptrend_threshold_pct` mã đang trên EMA200 -> "uptrend".
        - <= `downtrend_threshold_pct` mã đang trên EMA200 -> "downtrend".
        - Ở giữa -> "sideway".
        - NGOẠI LỆ: nếu khoảng cách trung bình |close - EMA200| / EMA200
          nhỏ hơn `sideway_distance_threshold_pct`%, LUÔN phân loại là
          "sideway" bất kể tỷ lệ trên/dưới EMA200 nghiêng về đâu — vì giá
          đang "xoay quanh trục" mà không tách xa rõ rệt theo hướng nào.

    Trả về dict:
        - raw_regime: "uptrend" | "downtrend" | "sideway"
        - pct_above_ema200: tỷ lệ mã đang trên EMA200 (trong số mã có đủ
          dữ liệu để đánh giá)
        - avg_distance_pct: khoảng cách trung bình |close-EMA200|/EMA200 (%)
        - confidence: độ tin cậy [0, 1]
        - n_symbols_considered: số mã có đủ dữ liệu (loại các mã có
          `price_above_ema200` là None do chưa đủ lịch sử tính EMA200)
        - reasoning: mô tả căn cứ ra kết luận
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    valid = [
        s for s in snapshots
        if s.get("price_above_ema200") is not None and s.get("ema200") is not None
    ]

    if not valid:
        return {
            "raw_regime": "sideway",
            "pct_above_ema200": None,
            "avg_distance_pct": None,
            "confidence": 0.0,
            "n_symbols_considered": 0,
            "reasoning": [
                "Không có mã nào đủ dữ liệu để đánh giá vị trí so với EMA200 "
                "-> mặc định phân loại Sideway (trung tính) do thiếu căn cứ."
            ],
        }

    n = len(valid)
    n_above = sum(1 for s in valid if s["price_above_ema200"])
    pct_above = n_above / n

    distances = [
        abs(s["close"] - s["ema200"]) / s["ema200"] * 100.0
        for s in valid
        if s["ema200"] not in (0, None)
    ]
    avg_distance_pct = sum(distances) / len(distances) if distances else 0.0

    reasoning: list[str] = [
        f"{n_above}/{n} mã ({pct_above * 100:.1f}%) đang ở trên đường EMA200.",
        f"Khoảng cách trung bình |giá - EMA200| = {avg_distance_pct:.2f}%.",
    ]

    if avg_distance_pct < cfg["sideway_distance_threshold_pct"]:
        raw_regime = "sideway"
        reasoning.append(
            f"Khoảng cách trung bình ({avg_distance_pct:.2f}%) nhỏ hơn ngưỡng "
            f"{cfg['sideway_distance_threshold_pct']}% -> giá đang xoay quanh "
            f"EMA200, không tách xa rõ rệt theo hướng nào -> Sideway."
        )
        confidence = max(0.0, 1.0 - avg_distance_pct / cfg["sideway_distance_threshold_pct"])
    elif pct_above >= cfg["uptrend_threshold_pct"]:
        raw_regime = "uptrend"
        reasoning.append(
            f"Tỷ lệ mã trên EMA200 ({pct_above * 100:.1f}%) >= ngưỡng "
            f"{cfg['uptrend_threshold_pct'] * 100:.0f}% -> Uptrend."
        )
        confidence = min(1.0, (pct_above - 0.5) * 2)
    elif pct_above <= cfg["downtrend_threshold_pct"]:
        raw_regime = "downtrend"
        reasoning.append(
            f"Tỷ lệ mã trên EMA200 ({pct_above * 100:.1f}%) <= ngưỡng "
            f"{cfg['downtrend_threshold_pct'] * 100:.0f}% -> Downtrend."
        )
        confidence = min(1.0, (0.5 - pct_above) * 2)
    else:
        raw_regime = "sideway"
        reasoning.append(
            "Tỷ lệ mã trên EMA200 nằm giữa 2 ngưỡng uptrend/downtrend -> Sideway."
        )
        confidence = 1.0 - abs(pct_above - 0.5) * 2

    return {
        "raw_regime": raw_regime,
        "pct_above_ema200": pct_above,
        "avg_distance_pct": avg_distance_pct,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "n_symbols_considered": n,
        "reasoning": reasoning,
    }


# ==============================================================================
# ĐỘ TRỄ XÁC NHẬN (confirmation lag) — chống nhiễu
# ==============================================================================

def apply_confirmation_lag(
    raw_regime_history: list[str], confirmation_lag_sessions: int = 3
) -> Optional[str]:
    """Chỉ xác nhận đổi giai đoạn thị trường khi tín hiệu ổn định qua đủ
    `confirmation_lag_sessions` phiên GẦN NHẤT liên tiếp.

    `raw_regime_history`: danh sách các phân loại "thô" (raw_regime, chưa
    qua xác nhận) theo thứ tự CŨ -> MỚI, phần tử cuối cùng là phiên gần
    nhất (hôm nay).

    Trả về:
        - Tên giai đoạn (regime) nếu `confirmation_lag_sessions` phiên gần
          nhất đều giống nhau (tín hiệu đã ổn định).
        - None nếu chưa đủ lịch sử, hoặc tín hiệu chưa ổn định (còn dao
          động) — trong trường hợp này, bên gọi hàm nên GIỮ NGUYÊN giai
          đoạn đã xác nhận trước đó thay vì đổi ngay.
    """
    if len(raw_regime_history) < confirmation_lag_sessions:
        return None

    recent = raw_regime_history[-confirmation_lag_sessions:]
    if all(r == recent[0] for r in recent):
        return recent[0]
    return None


# ==============================================================================
# HÀM TỔ HỢP CHÍNH — kết hợp cả 3 bước theo ĐÚNG TRÌNH TỰ
# ==============================================================================

def detect_market_regime(
    macro_context: list[MacroDataPoint],
    sector_price_data: list[dict],
    sector_name: Optional[str] = None,
    raw_regime_history: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> dict:
    """Xác định giai đoạn thị trường cho MỘT ngành/nhóm cổ phiếu cụ thể,
    theo ĐÚNG TRÌNH TỰ: vĩ mô (Bước 1) trước, kỹ thuật EMA200 (Bước 2) sau.

    Tham số:
        macro_context: danh sách MacroDataPoint từ data_collector.
        sector_price_data: danh sách snapshot chỉ báo (từ
            core.indicators.get_indicator_snapshot) của từng mã trong
            ngành đang xét.
        sector_name: tên ngành đang xét (dùng để tra cứu xem ngành này có
            đang bị macro gắn cờ thận trọng hay không). Nếu None, bỏ qua
            việc áp đặt cờ thận trọng riêng cho ngành này (macro vẫn được
            tính vào `affected_sectors` ở output để tham khảo).
        raw_regime_history: lịch sử phân loại THÔ (chưa xác nhận) của
            NGÀNH NÀY trong các phiên trước (cũ -> mới, KHÔNG bao gồm kết
            quả của phiên hiện tại). Nếu truyền vào, hàm sẽ áp dụng độ trễ
            xác nhận trước khi trả về `regime` cuối cùng. Nếu None (mặc
            định), bỏ qua bước xác nhận độ trễ — trả về ngay kết quả của
            phiên hiện tại (phù hợp khi gọi độc lập/test riêng lẻ).
        config: override các ngưỡng mặc định (xem `DEFAULT_CONFIG`).

    Trả về:
        {"regime": "uptrend"|"downtrend"|"sideway", "confidence": float,
         "reasoning": [...], "affected_sectors": [...]}

        `affected_sectors` là danh sách TẤT CẢ các ngành hiện đang bị macro
        gắn cờ thận trọng (không giới hạn riêng ngành đang xét) — để bên
        gọi có cái nhìn tổng quan toàn thị trường.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    reasoning: list[str] = []

    # --- BƯỚC 1: Vĩ mô trước tiên ---
    macro_result = evaluate_macro_filter(macro_context)
    reasoning.extend(macro_result["reasoning"])

    is_sector_cautioned = (
        sector_name is not None and sector_name in macro_result["caution_sectors"]
    )

    # --- BƯỚC 2: Kỹ thuật EMA200 theo ngành ---
    technical_result = classify_sector_trend(sector_price_data, config=cfg)
    reasoning.extend(technical_result["reasoning"])

    raw_regime_today = technical_result["raw_regime"]
    confidence = technical_result["confidence"]

    # Áp dụng lớp lọc vĩ mô: nếu ngành đang bị gắn cờ thận trọng, KHÔNG
    # được phân loại Uptrend dù kỹ thuật có tốt đến đâu — đây chính là
    # nguyên tắc "vĩ mô ưu tiên trước kỹ thuật".
    if is_sector_cautioned and raw_regime_today == "uptrend":
        raw_regime_today = "sideway"
        confidence = round(confidence * 0.7, 4)  # giảm độ tin cậy do bị ghi đè
        reasoning.append(
            f"Ngành '{sector_name}' đang bị vĩ mô gắn cờ THẬN TRỌNG -> hạ "
            f"phân loại kỹ thuật từ Uptrend xuống Sideway, bất kể tín hiệu "
            f"EMA200 đang tích cực."
        )

    # --- Độ trễ xác nhận (nếu có truyền lịch sử) ---
    if raw_regime_history is not None:
        full_history = list(raw_regime_history) + [raw_regime_today]
        confirmed = apply_confirmation_lag(
            full_history, confirmation_lag_sessions=cfg["confirmation_lag_sessions"]
        )
        if confirmed is None:
            reasoning.append(
                f"Tín hiệu '{raw_regime_today}' CHƯA đủ {cfg['confirmation_lag_sessions']} "
                f"phiên liên tiếp ổn định -> CHƯA xác nhận đổi giai đoạn "
                f"(giữ nguyên trạng thái trước đó theo quyết định của bên gọi)."
            )
            final_regime = None
        else:
            reasoning.append(
                f"Tín hiệu đã ổn định qua {cfg['confirmation_lag_sessions']} phiên "
                f"liên tiếp -> xác nhận giai đoạn: {confirmed}."
            )
            final_regime = confirmed
    else:
        final_regime = raw_regime_today

    return {
        "regime": final_regime,
        "confidence": confidence,
        "reasoning": reasoning,
        "affected_sectors": sorted(macro_result["caution_sectors"]),
    }


# ==============================================================================
# BỔ SUNG — MÔ HÌNH 3 LỚP ĐỊNH LƯỢNG (theo tài liệu kỹ thuật chi tiết)
# ==============================================================================
# Phần dưới đây cài đặt lại theo đúng tài liệu kỹ thuật "Hệ thống xác định
# trạng thái thị trường CK Việt Nam theo mô hình 3 lớp": Lớp 1 (điểm số vĩ
# mô định lượng) -> Lớp 2 (% Breadth EMA200, xem core/market_breadth.py)
# -> Lớp 3 (đối chiếu chỉ báo xác nhận). Đây là phiên bản ĐỊNH LƯỢNG hơn,
# bổ sung thêm cho `detect_market_regime()` ở trên (không thay thế) —
# dùng hàm nào tùy nhu cầu (định tính nhanh vs. định lượng chi tiết).

MACRO_CATEGORY_WEIGHTS = {
    "fx_intervention": 1.5,
    "interest_rate": 1.5,
    "omo": 1.0,
    "sector_policy": 0.5,
}
MACRO_DIRECTION_SCORES = {"tightening": -1.0, "easing": 1.0, "neutral": 0.0}


def calculate_macro_score(macro_points: list[MacroDataPoint]) -> float:
    """Tính ĐIỂM SỐ VĨ MÔ định lượng trong khoảng [-2, +2] (mục 1.1 & Lớp 1
    tài liệu kỹ thuật). Điểm càng âm -> vĩ mô càng tiêu cực.

    Mỗi điểm dữ liệu vĩ mô được gán trọng số theo mức độ ảnh hưởng tới
    toàn thị trường (tỷ giá/lãi suất ảnh hưởng rộng hơn chính sách riêng
    một ngành), rồi lấy trung bình có trọng số của điểm "tightening/easing"
    và co giãn về thang [-2, +2].
    """
    if not macro_points:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for point in macro_points:
        weight = MACRO_CATEGORY_WEIGHTS.get(point.category, 1.0)
        score = MACRO_DIRECTION_SCORES.get(point.direction or "neutral", 0.0)
        weighted_sum += weight * score
        total_weight += weight

    if total_weight == 0:
        return 0.0

    normalized = weighted_sum / total_weight  # nằm trong [-1, 1]
    return max(-2.0, min(2.0, normalized * 2))


def detect_market_regime_quant(
    macro_context: list[MacroDataPoint],
    group_snapshots: list[dict],
    breadth_history: Optional[list[float]] = None,
    layer3_indicators: Optional[dict] = None,
    group_name: str = "toàn thị trường",
    precomputed_macro_score: Optional[float] = None,
) -> dict:
    """Xác định trạng thái thị trường theo ĐÚNG 3 BƯỚC của tài liệu kỹ
    thuật chi tiết (macro score -> % breadth EMA200 -> đối chiếu Lớp 3).

    Tham số:
        macro_context: danh sách MacroDataPoint (như `detect_market_regime`).
            BỎ QUA nếu đã truyền `precomputed_macro_score`.
        group_snapshots: danh sách snapshot chỉ báo (từ
            `core.indicators.get_indicator_snapshot`) của các mã trong
            nhóm/toàn thị trường đang xét — dùng tính % Breadth EMA200.
        breadth_history: lịch sử % Breadth của nhóm này trong các phiên
            TRƯỚC đó (không bao gồm hôm nay) — dùng xác định xu hướng
            breadth đang tăng/giảm/đi ngang (mục 2.3).
        layer3_indicators: dict tùy chọn chứa kết quả các chỉ báo Lớp 3 đã
            tính sẵn (từ `core.market_breadth`), ví dụ:
                {"ma_cross": "golden_cross"|"death_cross"|"none",
                 "adx": float, "band_width_percentile": float,
                 "volume_ratio": float}
            Càng nhiều chỉ báo đồng thuận với nhãn Lớp 2 -> độ tin cậy
            càng cao. Bỏ trống nếu chưa có dữ liệu Lớp 3.
        precomputed_macro_score: nếu truyền vào, DÙNG TRỰC TIẾP giá trị
            này làm điểm vĩ mô (Lớp 1) thay vì tự tính từ `macro_context`
            — dùng khi đã có điểm chi tiết hơn từ
            `core.macro_score_engine.calculate_macro_score()` (dựa trên
            Fed Rate/CPI/tỷ giá/lãi suất liên ngân hàng/sự kiện thực tế),
            chính xác hơn nhiều so với ước lượng đơn giản từ `macro_context`.

    Trả về đúng cấu trúc OUTPUT theo tài liệu (mục 3):
        {"trang_thai", "do_tin_cay", "macro_score", "breadth_pct",
         "breadth_theo_nhom", "canh_bao", "reasoning"}
    """
    from core.market_breadth import calculate_breadth_trend, classify_breadth_label
    from core.market_breadth import calculate_ema200_breadth

    reasoning: list[str] = []
    canh_bao: list[str] = []

    # --- BƯỚC 1: Điểm số vĩ mô ---
    if precomputed_macro_score is not None:
        macro_score = precomputed_macro_score
        reasoning.append(
            f"Điểm số vĩ mô (Lớp 1, từ macro_score_engine chi tiết): "
            f"{macro_score:+.2f} (thang -2 đến +2)."
        )
    else:
        macro_score = calculate_macro_score(macro_context)
        reasoning.append(
            f"Điểm số vĩ mô (Lớp 1, ước lượng đơn giản từ macro_context): "
            f"{macro_score:+.2f} (thang -2 đến +2)."
        )
    macro_capped_to_sideway = macro_score < 0
    if macro_capped_to_sideway:
        reasoning.append(
            "Vĩ mô tiêu cực (điểm < 0) -> GIỚI HẠN TRẦN nhãn tối đa là SIDEWAY, "
            "không cho phép gán UPTREND dù Lớp 2 tích cực."
        )

    # --- BƯỚC 2: % Breadth EMA200 ---
    breadth_result = calculate_ema200_breadth(group_snapshots)
    breadth_pct = breadth_result["breadth_pct"]
    breadth_trend = calculate_breadth_trend(breadth_history) if breadth_history else None
    raw_label = classify_breadth_label(breadth_pct, breadth_trend)
    base_label = {"uptrend_extreme": "uptrend", "downtrend_extreme": "downtrend"}.get(
        raw_label, raw_label
    )

    reasoning.append(
        f"% Breadth EMA200 ({group_name}): "
        f"{breadth_pct:.1f}% ({breadth_result['n_above']}/{breadth_result['n_valid']} mã)"
        if breadth_pct is not None else
        f"% Breadth EMA200 ({group_name}): không đủ dữ liệu."
    )
    if raw_label in ("uptrend_extreme", "downtrend_extreme"):
        canh_bao.append(
            f"Breadth ở vùng CỰC ĐOAN ({raw_label}) — "
            f"{'cảnh báo quá mua diện rộng, khả năng sắp điều chỉnh' if raw_label == 'uptrend_extreme' else 'vùng washout, khả năng tạo đáy kỹ thuật'}."
        )

    final_label = base_label
    if macro_capped_to_sideway and base_label == "uptrend":
        final_label = "sideway"
        reasoning.append(
            "Áp dụng giới hạn trần từ Bước 1 -> hạ nhãn từ UPTREND xuống SIDEWAY."
        )

    # --- BƯỚC 3: Đối chiếu Lớp 3 ---
    agreements = 0
    total_checks = 0
    layer3_indicators = layer3_indicators or {}

    ma_cross = layer3_indicators.get("ma_cross")
    if ma_cross in ("golden_cross", "death_cross"):
        total_checks += 1
        if (ma_cross == "golden_cross" and final_label == "uptrend") or (
            ma_cross == "death_cross" and final_label == "downtrend"
        ):
            agreements += 1
        else:
            canh_bao.append(f"MA50/200 cross ({ma_cross}) không đồng thuận với nhãn {final_label}.")

    adx = layer3_indicators.get("adx")
    if adx is not None:
        total_checks += 1
        if final_label in ("uptrend", "downtrend") and adx > 25:
            agreements += 1
        elif final_label == "sideway" and adx < 20:
            agreements += 1
        else:
            canh_bao.append(f"ADX={adx:.1f} không xác nhận rõ nhãn {final_label}.")

    band_width_pctl = layer3_indicators.get("band_width_percentile")
    if band_width_pctl is not None:
        total_checks += 1
        if final_label == "sideway" and band_width_pctl < 20:
            agreements += 1
        elif final_label != "sideway" and band_width_pctl >= 20:
            agreements += 1
        else:
            canh_bao.append(
                f"Band Width percentile={band_width_pctl:.0f} không đồng thuận với nhãn {final_label}."
            )

    if total_checks == 0:
        do_tin_cay = "TRUNG_BINH"
        reasoning.append("Chưa có chỉ báo Lớp 3 để đối chiếu -> độ tin cậy mặc định TRUNG BÌNH.")
    else:
        ratio = agreements / total_checks
        if ratio == 1.0:
            do_tin_cay = "CAO"
        elif ratio >= 0.5:
            do_tin_cay = "TRUNG_BINH"
        else:
            do_tin_cay = "THAP"
        reasoning.append(
            f"Đối chiếu Lớp 3: {agreements}/{total_checks} chỉ báo đồng thuận -> độ tin cậy {do_tin_cay}."
        )

    return {
        "trang_thai": final_label.upper(),
        "do_tin_cay": do_tin_cay,
        "macro_score": round(macro_score, 3),
        "breadth_pct": round(breadth_pct, 2) if breadth_pct is not None else None,
        "breadth_theo_nhom": group_name,
        "canh_bao": canh_bao,
        "reasoning": reasoning,
    }
