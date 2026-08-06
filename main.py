"""
main.py
========
Điểm khởi chạy tổng của pm_ck — NỐI TOÀN BỘ LUỒNG CHẠY của hệ thống:

    data_collector -> indicators -> pattern_detector -> market_regime_detector
        -> capital_allocator -> paper_portfolio -> storage -> (dashboard đọc lại)

CHỈ ĐỌC DỮ LIỆU VÀ ĐƯA KHUYẾN NGHỊ THAM KHẢO — KHÔNG đặt lệnh giao dịch
thật dưới bất kỳ hình thức nào.

CÁCH CHẠY:
    python main.py

SAU KHI CHẠY XONG, xem kết quả trên dashboard:
    streamlit run dashboard/app.py

LƯU Ý VỀ PHẠM VI CỦA main.py Ở PHIÊN BẢN NÀY (các đơn giản hóa có chủ đích):
    1. Nguồn dữ liệu lấy theo cấu hình `data_source.adapter` trong
       config.yaml (mặc định: "vnstock" — dữ liệu THẬT). Có thể đổi lại
       thành "mock" trong config.yaml nếu muốn chạy thử không cần mạng.
    2. Ánh xạ mã -> ngành trong `config.yaml` (mục `watchlist.symbols`)
       là ánh xạ TẠM THỜI cho mục đích demo, cần thay bằng dữ liệu ngành
       thật khi có nguồn phù hợp.
    3. KHÔNG áp dụng độ trễ xác nhận đa phiên (confirmation lag) của
       `market_regime_detector` trong phiên bản chạy MỘT LẦN này — cơ chế
       đó cần lưu lại lịch sử phân loại thô qua NHIỀU LẦN CHẠY (ví dụ chạy
       định kỳ mỗi phiên giao dịch), nằm ngoài phạm vi một lần chạy đơn lẻ.
       Muốn bật đầy đủ, xem ghi chú trong `run_market_regime_step()`.
    4. Danh mục mô phỏng (`paper_portfolio`) được khởi tạo MỚI mỗi lần
       chạy (chưa có logic khôi phục vị thế đã lưu từ phiên trước) — phù
       hợp cho mục đích minh họa luồng chạy; khi dùng thực tế nhiều phiên
       liên tục, cần bổ sung logic nạp lại trạng thái danh mục từ storage.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import yaml
import pandas as pd

from backtest.backtest_engine import Trade  # noqa: F401 (tham chiếu cho phát triển sau)
from core.capital_allocator import get_allocation_recommendation
from core.data_collector import BinanceDataSource, DataCollector, MockDataSource, VnstockDataSource
from core.indicators import get_indicator_snapshot
from core.market_regime_detector import detect_market_regime
from core.notifier import Notifier, RealTelegramClient
from core.paper_portfolio import create_portfolio
from core.pattern_detector import detect_narrowing_pattern
from core.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_ck.main")


# ==============================================================================
# CẤU HÌNH
# ==============================================================================

def load_config(path: str = "config/config.yaml") -> dict:
    """Đọc file cấu hình YAML trung tâm của toàn hệ thống."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_storage_path(config: dict) -> str:
    """Lấy đường dẫn/connection string storage — ưu tiên biến môi trường
    `PM_CK_DB_PATH` (nếu có) TRƯỚC `config.yaml`, để cho phép giữ mật
    khẩu/connection string THẬT (vd. Supabase) NGOÀI file config.yaml khi
    cần đưa code lên GitHub (tránh lộ mật khẩu trong repo) — người dùng
    tự đặt biến môi trường này 1 lần trên máy, config.yaml có thể giữ giá
    trị AN TOÀN (vd. SQLite cục bộ) để đưa lên git thoải mái.
    """
    env_override = os.environ.get("PM_CK_DB_PATH")
    if env_override:
        return env_override
    return config.get("storage", {}).get("path", "./data/pm_ck.db")


# ==============================================================================
# BƯỚC 1 — data_collector
# ==============================================================================

def build_data_collector(config: dict) -> DataCollector:
    """Khởi tạo DataCollector với adapter TƯƠNG ỨNG cấu hình
    `data_source.adapter` trong config.yaml:
        - "mock"    -> dữ liệu giả lập (mặc định, an toàn, không cần mạng)
        - "vnstock" -> dữ liệu THẬT qua thư viện vnstock (cần: pip install vnstock)
        - "binance" -> dữ liệu THẬT qua Binance Public API (cho XAUUSD/BTCUSD,
          dùng bởi core/volatility_contraction_scanner.py — KHÔNG dùng cho
          cổ phiếu Việt Nam, xem core/data_collector.py::BinanceDataSource)

    Muốn chuyển sang dữ liệu thật, sửa trong config.yaml:
        data_source:
          adapter: "vnstock"

    Muốn lấy lịch sử DÀI HƠN mức mặc định (~800 phiên ≈ 3,2 năm gần nhất),
    thêm vào config.yaml (bổ sung 04/08/2026 — VD lấy từ đầu năm 2021):
        data_source:
          adapter: "vnstock"
          vnstock_start_date: "2021-01-01"
    """
    data_source_cfg = config.get("data_source", {})
    adapter_name = data_source_cfg.get("adapter", "mock")

    if adapter_name == "mock":
        source = MockDataSource()
    elif adapter_name == "vnstock":
        source = VnstockDataSource(start_date=data_source_cfg.get("vnstock_start_date"))
    elif adapter_name == "binance":
        source = BinanceDataSource()
    else:
        raise ValueError(
            f"adapter '{adapter_name}' không được hỗ trợ. Chỉ chấp nhận "
            f"'mock', 'vnstock' hoặc 'binance'. Xem core/data_collector.py "
            f"để thêm adapter mới."
        )

    return DataCollector(source, config=data_source_cfg)


# ==============================================================================
# BƯỚC 2 — indicators + pattern_detector (cho MỘT mã)
# ==============================================================================

def run_indicator_and_pattern_step(
    collector: DataCollector,
    storage: Storage,
    symbol: str,
    config: dict,
) -> tuple[dict, Optional[dict], pd.DataFrame]:
    """Lấy OHLCV, tính chỉ báo, phát hiện mô hình thu hẹp biên độ cho MỘT
    mã, lưu kết quả vào storage. Trả về (indicator_snapshot, pattern_result, df)
    — `df` được trả kèm để bước xác định giai đoạn thị trường định lượng
    (Lớp 3: MA cross/ADX/Band Width) dùng lại mà KHÔNG cần gọi lại API.
    """
    df = collector.get_ohlcv(symbol, timeframe="day")

    # Lưu lại lịch sử OHLCV (tối đa 750 phiên gần nhất) để dashboard vẽ
    # biểu đồ nến — không tốn thêm lệnh gọi API nào (dùng chung dữ liệu
    # vừa lấy ở trên).
    ohlcv_tail = df.tail(1500).copy()  # tăng từ 750 -> 1500 (~2021 tới nay + đệm)
    ohlcv_tail["date"] = ohlcv_tail["date"].astype(str)
    storage.save(
        "ohlcv_history", symbol,
        {"records": ohlcv_tail.to_dict(orient="records")},
    )

    snapshot = get_indicator_snapshot(df, config=config.get("indicators", {}))
    storage.save("indicator_snapshot", symbol, snapshot)
    logger.info(
        "[%s] Chỉ báo: close=%.2f, EMA200=%s, trên EMA200=%s",
        symbol, snapshot["close"], snapshot.get("ema200"),
        snapshot.get("price_above_ema200"),
    )

    # Lưu giá THỜI GIAN THỰC riêng — khác với "close" trong indicator
    # snapshot (vốn lấy từ phiên gần nhất của dữ liệu OHLCV, có thể trễ
    # hơn giá khớp lệnh thực tế tùy nguồn dữ liệu và thời điểm chạy).
    try:
        realtime = collector.get_realtime_price(symbol)
        storage.save("realtime_price", symbol, realtime)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Không lấy được giá thời gian thực: %s", symbol, exc)

    pattern_cfg = config.get("pattern_detector", {})
    pattern_result = detect_narrowing_pattern(
        df,
        scan_months_range=(
            pattern_cfg.get("scan_months_min", 10),
            pattern_cfg.get("scan_months_max", 30),
        ),
        n_segments=pattern_cfg.get("n_segments", 4),
        symbol=symbol,
    )
    if pattern_result is not None:
        storage.save("pattern_result", symbol, pattern_result)
        logger.info(
            "[%s] Phát hiện mô hình thu hẹp biên độ, confidence=%.2f",
            symbol, pattern_result["confidence"],
        )
    else:
        logger.info("[%s] Không phát hiện mô hình thu hẹp biên độ phù hợp.", symbol)

    return snapshot, pattern_result, df


# ==============================================================================
# BƯỚC 3a — market_regime_detector (cho MỘT "ngành"/nhóm — ở đây demo 1 mã/ngành)
# ==============================================================================

def compute_precomputed_macro_score(storage: Storage) -> Optional[float]:
    """Tổng hợp TOÀN BỘ dữ liệu vĩ mô đã NHẬP TAY qua dashboard (đủ 6
    nhóm: Fed Rate, tỷ giá USD/VND, CPI Mỹ, CPI Việt Nam, lãi suất liên
    ngân hàng, sự kiện địa chính trị) và tính Macro Score chi tiết qua
    `core.macro_score_engine.calculate_macro_score()`.

    Trả về None nếu CHƯA có bất kỳ dữ liệu vĩ mô nào được nhập (khi đó
    các bước gọi hàm này nên tự tính macro_score theo cách đơn giản cũ,
    xem `detect_market_regime_quant()`).
    """
    try:
        from core.macro_score_engine import calculate_macro_score as calc_macro_score_v2
        from core.manual_macro_data import build_full_macro_score_engine_input

        def _load_series(key: str) -> list[dict]:
            record = storage.get_latest("manual_macro_series", key)
            return record["data"]["entries"] if record else []

        fed_series = _load_series("fed_rate")
        fx_series = _load_series("usdvnd_rate")
        cpi_us_series = _load_series("cpi_us")
        cpi_vn_series = _load_series("cpi_vn")
        interbank_overnight_series = _load_series("interbank_overnight")
        interbank_3m_series = _load_series("interbank_3m")

        target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
        muc_tieu_cpi_vn = target_record["data"]["value"] if target_record else None

        event_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
        event_key = event_record["data"]["event_key"] if event_record else None

        has_any_data = any([
            fed_series, fx_series, cpi_us_series, cpi_vn_series,
            interbank_overnight_series, interbank_3m_series, event_record,
        ])
        if not has_any_data:
            return None

        macro_input = build_full_macro_score_engine_input(
            fed_series, fx_series,
            cpi_us_series=cpi_us_series, cpi_vn_series=cpi_vn_series,
            muc_tieu_cpi_vn=muc_tieu_cpi_vn,
            interbank_overnight_series=interbank_overnight_series,
            interbank_3m_series=interbank_3m_series,
            event_key=event_key,
        )
        macro_v2_result = calc_macro_score_v2(macro_input)
        logger.debug(
            "Điểm vĩ mô chi tiết (macro_score_engine, đủ 6 nhóm): %.2f (%s)",
            macro_v2_result["macro_score"], macro_v2_result["nhan"],
        )
        return macro_v2_result["macro_score"]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Không tính được điểm vĩ mô chi tiết: %s", exc)
        return None


def run_market_regime_step(
    collector: DataCollector,
    storage: Storage,
    sector: str,
    snapshot: dict,
    df: pd.DataFrame,
    config: dict,
) -> tuple[dict, dict]:
    """Xác định giai đoạn thị trường cho một ngành — chạy CẢ 2 mô hình:
    1. `detect_market_regime()` (định tính, đơn giản, có độ trễ xác nhận).
    2. `detect_market_regime_quant()` (định lượng 3 lớp đầy đủ: macro
       score, % Breadth EMA200, đối chiếu Lớp 3 — theo tài liệu kỹ thuật
       chi tiết).

    ĐƠN GIẢN HÓA: dùng raw_regime_history=None (bỏ qua độ trễ xác nhận đa
    phiên) vì đây là 1 lần chạy đơn lẻ. Để bật đầy đủ cơ chế chống nhiễu:
        1. Lưu "raw regime" của mỗi ngành vào storage sau MỖI LẦN CHẠY
           (ví dụ category="raw_regime_history", key=sector).
        2. Trước khi gọi detect_market_regime(), đọc lại N-1 giá trị gần
           nhất từ storage, truyền vào tham số `raw_regime_history`.
        3. Chạy main.py định kỳ (ví dụ mỗi phiên giao dịch) để lịch sử
           tích lũy đủ số phiên cấu hình trong
           `market_regime.confirmation_lag_sessions`.
    """
    macro_points = collector.get_macro_data()

    # Ở demo này, mỗi "ngành" chỉ có 1 mã đại diện — trong triển khai thật
    # nên gom snapshot của NHIỀU mã cùng ngành vào đây.
    sector_snapshots = [snapshot]

    regime_result = detect_market_regime(
        macro_context=macro_points,
        sector_price_data=sector_snapshots,
        sector_name=sector,
        raw_regime_history=None,  # xem ghi chú ở trên
        config=config.get("market_regime", {}),
    )

    storage.save("market_regime", sector, regime_result)
    logger.info(
        "[%s] Giai đoạn thị trường (định tính): %s (confidence=%.2f)",
        sector, regime_result["regime"], regime_result["confidence"],
    )

    # --- Mô hình định lượng 3 lớp (core/market_regime_detector.detect_market_regime_quant) ---
    from core.indicators import calculate_ma
    from core.market_breadth import (
        calculate_adx,
        calculate_bollinger_band_width,
        detect_ma_cross,
    )
    from core.market_regime_detector import detect_market_regime_quant

    layer3_indicators = {}
    try:
        ma50 = calculate_ma(df, 50)
        ma200 = calculate_ma(df, 200)
        layer3_indicators["ma_cross"] = detect_ma_cross(ma50, ma200)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] Không đủ dữ liệu tính MA50/200 cross: %s", sector, exc)

    try:
        adx_series = calculate_adx(df, period=14)
        if not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
            layer3_indicators["adx"] = float(adx_series.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] Không đủ dữ liệu tính ADX: %s", sector, exc)

    try:
        band_width_series = calculate_bollinger_band_width(df, period=20)
        band_width_valid = band_width_series.dropna()
        if len(band_width_valid) >= 20:
            percentile = (band_width_valid <= band_width_valid.iloc[-1]).mean() * 100.0
            layer3_indicators["band_width_percentile"] = float(percentile)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] Không đủ dữ liệu tính Band Width: %s", sector, exc)

    # --- Điểm vĩ mô CHI TIẾT (macro_score_engine), dùng dữ liệu Fed Rate
    #     + tỷ giá USD/VND đã NHẬP TAY qua dashboard (nếu có) ---
    precomputed_macro_score = compute_precomputed_macro_score(storage)

    regime_quant_result = detect_market_regime_quant(
        macro_context=macro_points,
        group_snapshots=sector_snapshots,
        layer3_indicators=layer3_indicators,
        group_name=sector,
        precomputed_macro_score=precomputed_macro_score,
    )
    storage.save("market_regime_quant", sector, regime_quant_result)
    logger.info(
        "[%s] Giai đoạn thị trường (định lượng 3 lớp): %s (độ tin cậy=%s, macro=%.2f, breadth=%s)",
        sector, regime_quant_result["trang_thai"], regime_quant_result["do_tin_cay"],
        regime_quant_result["macro_score"], regime_quant_result["breadth_pct"],
    )

    return regime_result, regime_quant_result


def run_capital_allocation_v2_step(
    storage: Storage,
    symbol: str,
    sector: str,
    snapshot: dict,
    df: pd.DataFrame,
    regime_quant_result: dict,
    nav: float,
    config: dict,
) -> Optional[dict]:
    """Chạy module khuyến nghị phân bổ vốn MỚI (`core/capital_allocation_engine`)
    cho MỘT mã, dựa trên:
        - Giai đoạn thị trường định lượng đã tính (`regime_quant_result`).
        - ATR14 + hỗ trợ/kháng cự tự động xác định TỪ DỮ LIỆU OHLCV THẬT
          (không hardcode bất kỳ mức giá nào).
        - Phân kỳ tăng (Bullish Divergence) tự động phát hiện qua
          `core.market_breadth.detect_bullish_divergence()` — dùng để mở
          khóa khuyến nghị mua trong giai đoạn DOWNTREND (thay vì luôn
          chặn cứng như phiên bản trước).

    Chọn chiến lược vào lệnh theo giai đoạn:
        UPTREND   -> "breakout" (xác nhận giá đã vượt kháng cự)
        DOWNTREND/SIDEWAY -> "support" (mua quanh vùng hỗ trợ)
    """
    from core.capital_allocation_engine import (
        InvalidCapitalAllocationError,
        calculate_capital_allocation,
        find_support_resistance,
    )
    from core.market_breadth import calculate_atr, detect_bullish_divergence

    trang_thai = regime_quant_result.get("trang_thai")
    if trang_thai not in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        logger.info(
            "[%s] Giai đoạn thị trường chưa xác định rõ ràng -> bỏ qua "
            "bước phân bổ vốn (module mới).", symbol,
        )
        return None

    try:
        atr14 = float(calculate_atr(df, period=14).iloc[-1])
        support, resistance = find_support_resistance(df, lookback=60)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] Không đủ dữ liệu tính ATR14/hỗ trợ-kháng cự -> bỏ qua "
            "bước phân bổ vốn (module mới): %s", symbol, exc,
        )
        return None

    strategy = "breakout" if trang_thai == "UPTREND" else "support"

    # --- Phát hiện phân kỳ tăng (chỉ thực sự CẦN THIẾT khi DOWNTREND,
    #     nhưng tính luôn để hiển thị tham khảo cho mọi giai đoạn) ---
    co_phan_ky_tang = False
    try:
        divergence_result = detect_bullish_divergence(df, rsi_period=14, lookback=90)
        co_phan_ky_tang = divergence_result.get("detected", False)
        if trang_thai == "DOWNTREND":
            if co_phan_ky_tang:
                logger.info(
                    "[%s] Phát hiện PHÂN KỲ TĂNG: đáy giá %.2f -> %.2f (thấp hơn), "
                    "RSI %.1f -> %.1f (cao hơn) -> MỞ KHÓA khuyến nghị mua.",
                    symbol, divergence_result["price_low_1"], divergence_result["price_low_2"],
                    divergence_result["rsi_low_1"], divergence_result["rsi_low_2"],
                )
            else:
                reason = divergence_result.get("reason", "chưa đủ điều kiện phân kỳ")
                logger.info(
                    "[%s] Chưa phát hiện phân kỳ tăng (%s) -> vẫn CHẶN khuyến nghị mua "
                    "trong giai đoạn DOWNTREND.", symbol, reason,
                )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] Không tính được phân kỳ tăng: %s", symbol, exc)

    watchlist_entry = {
        "ma": symbol,
        "nganh": sector,
        "atr14": atr14,
        "gia_tham_chieu": snapshot["close"],
        "chien_luoc": strategy,
        "ho_tro": support,
        "khang_cu": resistance,
        "co_phan_ky_tang": co_phan_ky_tang,
    }

    try:
        result = calculate_capital_allocation(
            trang_thai=trang_thai,
            do_tin_cay=regime_quant_result["do_tin_cay"],
            breadth_theo_nganh={sector: regime_quant_result.get("breadth_pct") or 0.0},
            nav=nav,
            watchlist=[watchlist_entry],
            risk_per_trade_pct=config.get("capital_allocator", {}).get(
                "risk_per_trade_pct", 2.0
            ) / 100.0,
            risk_total_pct=config.get("capital_allocator", {}).get(
                "risk_total_portfolio_pct", 20.0
            ) / 100.0,
        )
    except InvalidCapitalAllocationError as exc:
        logger.warning("[%s] Không tính được khuyến nghị phân bổ vốn (module mới): %s", symbol, exc)
        return None

    storage.save("capital_allocation_v2", symbol, result)

    if result["cac_dot_giai_ngan"]:
        first_symbol_info = result["cac_dot_giai_ngan"][0]["danh_sach_ma"]
        if first_symbol_info:
            info = first_symbol_info[0]
            logger.info(
                "[%s] Phân bổ vốn (module mới): entry=%s, KL=%d, cắt lỗ=%s, chốt lời=%s",
                symbol, info["khoang_gia_vao_lenh"], info["khoi_luong_du_kien"],
                info["khoang_cat_lo"], info["khoang_chot_loi_tham_khao"],
            )
    else:
        logger.info(
            "[%s] Phân bổ vốn (module mới): KHÔNG khuyến nghị giải ngân (%s).",
            symbol, "; ".join(result["canh_bao"]) if result["canh_bao"] else trang_thai,
        )

    return result


# ==============================================================================
# BƯỚC 3c — stock_signal_engine (tín hiệu MUA/GIỮ/BÁN cho MỘT mã)
# ==============================================================================

def run_stock_character_step(storage: Storage, symbol: str, df: pd.DataFrame) -> Optional[dict]:
    """Tính tính cách giao dịch (`core/stock_character_classifier.py`) cho
    MỘT mã, dùng dữ liệu OHLCV đã có sẵn — không gọi thêm API. Lưu kết
    quả vào storage để dashboard hiển thị, và trả về để dùng ĐIỀU CHỈNH
    tín hiệu Mua/Bán + khuyến nghị phân bổ vốn ngay trong cùng lượt chạy.

    Trả về `None` nếu không đủ dữ liệu (mã mới niêm yết, ít hơn 30 phiên)
    — không coi là lỗi, các bước sau vẫn chạy bình thường KHÔNG điều
    chỉnh theo tính cách (coi như trung tính).
    """
    from core.stock_character_classifier import InsufficientDataError, phan_loai_tinh_cach_co_phieu

    try:
        tinh_cach = phan_loai_tinh_cach_co_phieu(symbol, df)
    except (InsufficientDataError, ValueError) as exc:
        logger.debug("[%s] Không tính được tính cách giao dịch: %s", symbol, exc)
        return None

    storage.save("stock_character", symbol, tinh_cach)
    logger.info(
        "[%s] Tính cách giao dịch: %s (choppiness=%.2f)",
        symbol, tinh_cach["nhan_tinh_cach"], tinh_cach["choppiness_score"],
    )
    return tinh_cach


def run_stock_signal_step(
    storage: Storage,
    symbol: str,
    df: pd.DataFrame,
    regime_quant_result: dict,
    config: dict,
    tinh_cach: Optional[dict] = None,
) -> Optional[dict]:
    """Chạy `core/stock_signal_engine.evaluate_stock_signal()` cho MỘT
    mã, dùng dữ liệu OHLCV thật đã có sẵn (không gọi thêm API).

    ĐƠN GIẢN HÓA: `fundamentals` (dữ liệu tài chính EPS/ROE/D-E/CFO) hiện
    LUÔN là None — dự án CHƯA có nguồn dữ liệu báo cáo tài chính. Module
    vẫn hoạt động đầy đủ ở lớp KỸ THUẬT, lớp cơ bản coi như trung tính
    (xem docstring `core/stock_signal_engine.py`).

    `position_info` (dùng cho kiểm tra cắt lỗ) lấy từ `trade_journal` nếu
    mã này đang có vị thế MUA còn mở.

    `tinh_cach`: nếu truyền vào (từ `run_stock_character_step()`), kết
    quả sẽ được ĐIỀU CHỈNH qua `core.character_integration.dieu_chinh_tin_hieu_theo_tinh_cach()`
    trước khi lưu (chiết khấu độ tin cậy Breakout nếu mã đang "lình xình",
    gộp thêm cảnh báo SQUAT/CHURNING).
    """
    from core.capital_allocation_engine import find_support_resistance
    from core.stock_signal_engine import InvalidStockSignalError, evaluate_stock_signal

    trang_thai = regime_quant_result.get("trang_thai")

    position_info = None
    trade_ids = storage.query_all_keys("trade_journal")
    for trade_id in trade_ids:
        record = storage.get_latest("trade_journal", trade_id)
        if record is None:
            continue
        trade = record["data"]
        if trade["symbol"] == symbol and not trade.get("is_closed"):
            position_info = {
                "gia_cat_lo": None,  # trade_journal hiện chưa lưu mức cắt lỗ riêng biệt
                "day_gan_nhat": df["low"].tail(20).min() if len(df) >= 20 else None,
                "loi_lo_hien_tai_pct_nav": None,
            }
            break

    try:
        support, resistance = find_support_resistance(df, lookback=60)
    except Exception:  # noqa: BLE001
        support, resistance = None, None

    try:
        result = evaluate_stock_signal(
            symbol=symbol, df=df,
            macro_score=regime_quant_result.get("macro_score"),
            market_regime=trang_thai,
            resistance_level=resistance, support_level=support,
            fundamentals=None,  # xem ghi chú ở docstring
            position_info=position_info,
            strategy=config.get("stock_signal", {}).get("strategy", "dau_tu"),
        )
    except InvalidStockSignalError as exc:
        logger.warning("[%s] Không tính được tín hiệu mua/bán: %s", symbol, exc)
        return None

    if tinh_cach is not None:
        from core.character_integration import dieu_chinh_tin_hieu_theo_tinh_cach
        result = dieu_chinh_tin_hieu_theo_tinh_cach(result, tinh_cach)

    storage.save("stock_signal", symbol, result)
    logger.info(
        "[%s] Tín hiệu: %s%s",
        symbol, result["khuyen_nghi"],
        f" ({result['loai_ban']}, ưu tiên {result['uu_tien']})" if result.get("loai_ban") else "",
    )
    return result


# ==============================================================================
# BƯỚC 3b — capital_allocator (cho MỘT mã)
# ==============================================================================

def run_allocation_step(
    storage: Storage,
    symbol: str,
    sector: str,
    snapshot: dict,
    pattern_result: Optional[dict],
    regime_result: dict,
    nav: float,
    config: dict,
    tinh_cach: Optional[dict] = None,
) -> Optional[dict]:
    """Đưa khuyến nghị phân bổ vốn cho MỘT mã, dựa trên vùng entry lấy từ
    kết quả `pattern_detector` (đoạn tích lũy gần nhất). Nếu chưa phát
    hiện mô hình nào (chưa có vùng entry rõ ràng), BỎ QUA bước này cho mã
    đó — vì `capital_allocator` YÊU CẦU BẮT BUỘC phải có entry_price_range.

    `tinh_cach`: nếu truyền vào (từ `run_stock_character_step()`), kết
    quả sẽ được ĐIỀU CHỈNH qua `core.character_integration.dieu_chinh_phan_bo_theo_tinh_cach()`
    trước khi lưu (giảm tỷ trọng nếu mã đang "bùng nổ ngắn" hoặc có cảnh
    báo CHURNING — ưu tiên bảo toàn vốn).
    """
    if pattern_result is None:
        logger.info(
            "[%s] Chưa có vùng entry rõ ràng từ pattern_detector -> bỏ qua "
            "bước khuyến nghị phân bổ vốn cho mã này.",
            symbol,
        )
        return None

    last_segment = pattern_result["segments"][-1]
    signal_price_context = {
        "current_price": snapshot["close"],
        "entry_low": last_segment["low"],
        "entry_high": last_segment["high"],
        "sector": sector,
    }

    allocation_result = get_allocation_recommendation(
        regime_result=regime_result,
        nav=nav,
        signal_price_context=signal_price_context,
        config=config.get("capital_allocator", {}),
    )

    if tinh_cach is not None:
        from core.character_integration import dieu_chinh_phan_bo_theo_tinh_cach
        allocation_result = dieu_chinh_phan_bo_theo_tinh_cach(allocation_result, tinh_cach)

    storage.save("allocation_recommendation", symbol, allocation_result)
    logger.info(
        "[%s] Khuyến nghị phân bổ vốn: %.1f%%, entry=[%.2f, %.2f], stop_loss=%s",
        symbol, allocation_result["target_pct"],
        signal_price_context["entry_low"], signal_price_context["entry_high"],
        allocation_result["stop_loss"],
    )
    return allocation_result


# ==============================================================================
# BỔ SUNG — CHỈ SỐ THỊ TRƯỜNG CHUNG (VNINDEX, VN30, VN100...)
# ==============================================================================

def run_index_step(
    collector: DataCollector,
    storage: Storage,
    index_symbol: str,
    config: dict,
) -> Optional[dict]:
    """Lấy dữ liệu + tính chỉ báo cho MỘT CHỈ SỐ thị trường chung (VNINDEX,
    VN30, VN100...) — dùng đúng endpoint chỉ số riêng của nguồn dữ liệu
    (`DataCollector.get_index_ohlcv`), KHÁC với cổ phiếu thường.

    CHỈ tính chỉ báo (MA20/EMA50/100/200, RSI qua biểu đồ) + lưu lịch sử
    OHLCV để xem trên dashboard — KHÔNG chạy pattern_detector/
    market_regime_detector/capital_allocation cho chỉ số, vì không thể
    "mua" một chỉ số như một cổ phiếu cụ thể.
    """
    try:
        df = collector.get_index_ohlcv(index_symbol, timeframe="day")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Không lấy được dữ liệu chỉ số: %s", index_symbol, exc)
        return None

    ohlcv_tail = df.tail(1500).copy()  # tăng từ 750 -> 1500 (~2021 tới nay + đệm)
    ohlcv_tail["date"] = ohlcv_tail["date"].astype(str)
    storage.save(
        "ohlcv_history", index_symbol,
        {"records": ohlcv_tail.to_dict(orient="records")},
    )

    snapshot = get_indicator_snapshot(df, config=config.get("indicators", {}))
    storage.save("indicator_snapshot", index_symbol, snapshot)
    logger.info(
        "[%s] Chỉ báo chỉ số: close=%.2f, EMA200=%s, trên EMA200=%s",
        index_symbol, snapshot["close"], snapshot.get("ema200"),
        snapshot.get("price_above_ema200"),
    )

    # LƯU Ý: KHÔNG gọi collector.get_realtime_price() cho chỉ số — đã xác
    # nhận qua kiểm tra trực tiếp (test_index_quote.py, 27/07/2026) rằng
    # vnstock.Market().quote() với mã CHỈ SỐ (VNINDEX/VN30/VN100...) chỉ
    # trả về đúng 1 cột 'symbol', KHÔNG có dữ liệu giá thật — khác hẳn cổ
    # phiếu thường (trả về đủ 30 cột). Đây là giới hạn của chính vnstock,
    # không phải lỗi code — gọi hàm này cho chỉ số sẽ LUÔN thất bại sau 3
    # lần thử lại vô ích (tốn ~14 giây/chỉ số + làm rối log). Giá đóng cửa
    # từ `indicator_snapshot` (snapshot["close"]) đã đủ dùng cho chỉ số.

    return snapshot


# ==============================================================================
# BƯỚC 4 — paper_portfolio + notifier
# ==============================================================================

def run_portfolio_step(
    storage: Storage,
    collector: DataCollector,
    symbols: list[str],
    config: dict,
) -> dict:
    """Khởi tạo danh mục mô phỏng (mới mỗi lần chạy — xem ghi chú ở đầu
    file), lấy snapshot NAV/PnL hiện tại và lưu vào storage để dashboard
    hiển thị.
    """
    portfolio_cfg = config.get("paper_portfolio", {})
    portfolio = create_portfolio(portfolio_cfg.get("initial_cash", 100_000_000))

    current_prices = {}
    for symbol in symbols:
        try:
            price_info = collector.get_realtime_price(symbol)
            current_prices[symbol] = price_info["price"]
        except Exception as exc:  # noqa: BLE001 — không để 1 mã lỗi làm hỏng cả pipeline
            logger.warning("[%s] Không lấy được giá hiện tại: %s", symbol, exc)

    snapshot = portfolio.get_portfolio_snapshot(current_prices)
    storage.save("portfolio_snapshot", "default", snapshot)

    logger.info(
        "Danh mục mô phỏng: NAV=%.0f, tỷ trọng cổ phiếu=%.1f%%",
        snapshot["nav"], snapshot["total_stock_weight_pct"],
    )
    return snapshot


def build_notifier(config: dict) -> Optional[Notifier]:
    """Khởi tạo Notifier NẾU Telegram được bật trong cấu hình. Trả về None
    nếu chưa bật (mặc định `notifier.telegram.enabled: false` trong
    config.yaml) — pipeline vẫn chạy bình thường, chỉ là không gửi cảnh
    báo ra ngoài.
    """
    telegram_cfg = config.get("notifier", {}).get("telegram", {})
    if not telegram_cfg.get("enabled", False):
        logger.info("Notifier Telegram đang TẮT (notifier.telegram.enabled=false).")
        return None

    client = RealTelegramClient(
        token_env_var=telegram_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")
    )
    return Notifier(
        client=client,
        whitelist_chat_ids=telegram_cfg.get("whitelist_chat_ids", []),
    )


# ==============================================================================
# LUỒNG CHẠY TỔNG
# ==============================================================================

def run_pipeline(config: dict) -> None:
    """Chạy toàn bộ luồng MỘT LẦN cho tất cả mã trong watchlist."""
    storage = Storage(db_path=resolve_storage_path(config))
    collector = build_data_collector(config)
    notifier = build_notifier(config)

    symbol_sector_map: dict[str, str] = config.get("watchlist", {}).get("symbols", {})
    if not symbol_sector_map:
        logger.warning(
            "Chưa cấu hình watchlist.symbols trong config.yaml -> không có mã "
            "nào để chạy pipeline."
        )
        return

    nav_for_allocation = config.get("paper_portfolio", {}).get("initial_cash", 100_000_000)

    for symbol, sector in symbol_sector_map.items():
        logger.info("=== Xử lý mã %s (ngành: %s) ===", symbol, sector)
        try:
            snapshot, pattern_result, df = run_indicator_and_pattern_step(
                collector, storage, symbol, config
            )

            regime_result, regime_quant_result = run_market_regime_step(
                collector, storage, sector, snapshot, df, config
            )

            # --- Tính cách giao dịch (core/stock_character_classifier.py)
            #     — tính MỘT LẦN, dùng để điều chỉnh CẢ tín hiệu mua/bán
            #     lẫn khuyến nghị phân bổ vốn bên dưới. ---
            tinh_cach = run_stock_character_step(storage, symbol, df)

            run_capital_allocation_v2_step(
                storage, symbol, sector, snapshot, df,
                regime_quant_result, nav_for_allocation, config,
            )

            allocation_result = run_allocation_step(
                storage, symbol, sector, snapshot, pattern_result,
                regime_result, nav_for_allocation, config,
                tinh_cach=tinh_cach,
            )

            run_stock_signal_step(
                storage, symbol, df, regime_quant_result, config,
                tinh_cach=tinh_cach,
            )

            if notifier is not None and pattern_result is not None:
                if pattern_result["confidence"] >= 0.7:
                    notifier.send_pattern_alert(
                        symbol=symbol,
                        confidence=pattern_result["confidence"],
                        accumulation_high=pattern_result["accumulation_high"],
                    )

        except Exception:
            logger.exception("Lỗi khi xử lý mã %s — bỏ qua, tiếp tục mã tiếp theo.", symbol)
            continue

    # --- Chỉ số thị trường chung (VNINDEX, VN30, VN100...) ---
    index_symbols = config.get("watchlist", {}).get("indices", [])
    for index_symbol in index_symbols:
        logger.info("=== Xử lý chỉ số %s ===", index_symbol)
        try:
            run_index_step(collector, storage, index_symbol, config)
        except Exception:
            logger.exception(
                "Lỗi khi xử lý chỉ số %s — bỏ qua, tiếp tục chỉ số tiếp theo.", index_symbol
            )
            continue

    run_portfolio_step(storage, collector, list(symbol_sector_map.keys()), config)

    # --- Báo cáo tổng hợp danh sách mã đủ điều kiện MUA/BÁN ---
    from core.stock_signal_engine import build_signal_summary_report

    signal_ids = storage.query_all_keys("stock_signal")
    evaluations = []
    for sid in signal_ids:
        record = storage.get_latest("stock_signal", sid)
        if record is not None:
            evaluations.append(record["data"])

    if evaluations:
        summary_report = build_signal_summary_report(evaluations)
        storage.save("signal_summary_report", "latest", summary_report)
        logger.info(
            "Báo cáo tổng hợp tín hiệu: %d MUA, %d BÁN CẮT LỖ, %d BÁN CHỐT LỜI, %d GIỮ/THEO DÕI.",
            len(summary_report["mua"]), len(summary_report["ban_cat_lo"]),
            len(summary_report["ban_chot_loi"]), len(summary_report["giu_theo_doi"]),
        )

    # --- Báo cáo tiêu chí ngắn hạn (VN-Index quá mua, bắt cá hồi, cổ phiếu quá mua) ---
    run_short_term_signal_step(storage, list(symbol_sector_map.keys()))
    run_entry_screener_step(storage, list(symbol_sector_map.keys()))

    storage.close()
    logger.info("Hoàn tất pipeline. Chạy 'streamlit run dashboard/app.py' để xem kết quả.")


def run_entry_screener_step(storage: Storage, watchlist_symbols: list[str]) -> Optional[dict]:
    """Chạy `core.entry_screener.quet_danh_sach_cho()` cho TOÀN BỘ
    watchlist, dùng dữ liệu ĐÃ CÓ SẴN trong storage (ohlcv_history,
    indicator_snapshot, pattern_result) — không gọi thêm API.

    Chạy với ĐỦ CẢ 4 tiêu chí — dashboard sẽ lọc lại theo lựa chọn của
    người dùng trên kết quả đã lưu (không cần recompute).
    """
    from core.capital_allocation_engine import find_support_resistance
    from core.entry_screener import TIEU_CHI_KHA_DUNG, quet_danh_sach_cho

    danh_sach_ma_info = []
    for symbol in watchlist_symbols:
        ohlcv_record = storage.get_latest("ohlcv_history", symbol)
        if ohlcv_record is None:
            continue
        records = ohlcv_record["data"].get("records", [])
        if len(records) < 30:
            continue

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        snapshot_record = storage.get_latest("indicator_snapshot", symbol)
        ema200 = snapshot_record["data"].get("ema200") if snapshot_record else None

        pattern_record = storage.get_latest("pattern_result", symbol)
        pattern_result = pattern_record["data"] if pattern_record else None

        try:
            support, resistance = find_support_resistance(df, lookback=60)
        except Exception:  # noqa: BLE001
            resistance = None

        volume_ma20 = df["volume"].tail(20).mean() if len(df) >= 20 else None

        danh_sach_ma_info.append({
            "symbol": symbol, "df": df, "ema200": ema200,
            "pattern_result": pattern_result,
            "resistance_level": resistance, "volume_ma20": volume_ma20,
        })

    if not danh_sach_ma_info:
        logger.info("Chưa có đủ dữ liệu -> bỏ qua bước rà soát danh sách vào lệnh.")
        return None

    report = quet_danh_sach_cho(
        danh_sach_ma_info, tieu_chi_da_chon=list(TIEU_CHI_KHA_DUNG.keys()),
    )
    storage.save("entry_screener_report", "latest", report)
    logger.info(
        "Báo cáo rà soát danh sách vào lệnh: %d/%d mã đạt ít nhất 1 tiêu chí.",
        report["tong_so_ma_dat"], report["tong_so_ma_da_quet"],
    )
    return report


def run_short_term_signal_step(storage: Storage, watchlist_symbols: list[str]) -> Optional[dict]:
    """Tính báo cáo tiêu chí ngắn hạn (`core.short_term_signal`) — dùng
    dữ liệu VNINDEX + watchlist ĐÃ CÓ SẴN trong storage, không gọi thêm
    API. Cần VNINDEX đã được xử lý qua `run_index_step()` trước đó
    (xem `watchlist.indices` trong config.yaml).
    """
    from core.short_term_signal import build_short_term_signal_report

    vnindex_snapshot_record = storage.get_latest("indicator_snapshot", "VNINDEX")
    vnindex_ohlcv_record = storage.get_latest("ohlcv_history", "VNINDEX")
    if vnindex_snapshot_record is None or vnindex_ohlcv_record is None:
        logger.info(
            "Chưa có dữ liệu VNINDEX (cần thêm 'VNINDEX' vào watchlist.indices "
            "trong config.yaml) -> bỏ qua báo cáo tiêu chí ngắn hạn."
        )
        return None

    vnindex_snapshot = vnindex_snapshot_record["data"]
    records = vnindex_ohlcv_record["data"].get("records", [])
    if len(records) < 40:
        logger.info("Chưa đủ 40 phiên dữ liệu VNINDEX -> bỏ qua báo cáo tiêu chí ngắn hạn.")
        return None

    df_vnindex = pd.DataFrame(records)
    df_vnindex["date"] = pd.to_datetime(df_vnindex["date"])

    from core.indicators import calculate_ma
    ma20_series = calculate_ma(df_vnindex, 20)

    macro_score = compute_precomputed_macro_score(storage)

    stock_snapshots = []
    for symbol in watchlist_symbols:
        record = storage.get_latest("indicator_snapshot", symbol)
        if record is not None and record["data"].get("ma20") is not None:
            stock_snapshots.append({
                "ma": symbol, "close": record["data"]["close"], "ma20": record["data"]["ma20"],
            })

    report = build_short_term_signal_report(
        vnindex_close=vnindex_snapshot["close"],
        vnindex_ma20=vnindex_snapshot["ma20"],
        vnindex_history_close=df_vnindex["close"],
        vnindex_history_ma20=ma20_series,
        vnindex_history_40d=df_vnindex["close"].tail(40).tolist(),
        stock_snapshots=stock_snapshots,
        macro_score=macro_score,
        danh_gia_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    storage.save("short_term_signal_report", "latest", report)
    logger.info(
        "Báo cáo tiêu chí ngắn hạn: VNIndex %s (%.2f%%), bắt cá hồi=%s, %d mã quá mua.",
        report["vnindex"]["muc_canh_bao"], report["vnindex"]["do_lech_ma20_pct"],
        report["tin_hieu_bat_ca_hoi"]["kich_hoat"], len(report["co_phieu_qua_mua"]),
    )
    return report


def _luu_chuoi_giai_doan(storage: Storage, key: str, chuoi) -> None:
    """Lưu 1 chuỗi giai đoạn (pd.Series index=ngày) vào storage dưới dạng
    JSON-hóa được (list các bản ghi {"date": ..., "giai_doan": ...})."""
    records = [
        {"date": str(ngay.date()) if hasattr(ngay, "date") else str(ngay), "giai_doan": gia_tri}
        for ngay, gia_tri in chuoi.items()
    ]
    storage.save("chuoi_giai_doan_lich_su", key, {"records": records})


def run_market_regime_history_step(storage: Storage) -> None:
    """Tính và LƯU LẠI (bổ sung 05/08/2026) chuỗi giai đoạn Uptrend/
    Sideway/Downtrend THEO TỪNG NGÀY trong lịch sử — cho TOÀN THỊ TRƯỜNG
    (key="thi_truong") và TỪNG NGÀNH (key=tên ngành) — chạy 1 LẦN/NGÀY
    như 1 bước trong pipeline chính (`run_full_market.py`), để dashboard
    CHỈ CẦN ĐỌC (nhanh, 1 lượt truy vấn) thay vì phải TÍNH LẠI mỗi lần
    người dùng mở mục "Lọc theo giai đoạn thị trường/ngành" (chậm, tốn
    nhiều lượt gọi Supabase, và mất khi Streamlit khởi động lại vì trước
    đây chỉ cache tạm trong bộ nhớ).

    CHỈ 1 LƯỢT TRUY VẤN LỚN duy nhất để lấy OHLCV của TẤT CẢ mã (dùng
    `get_latest_many`) — toàn bộ tính toán còn lại (vector hóa, tính cho
    từng ngành) đều làm TRONG BỘ NHỚ, không tốn thêm request nào.
    """
    from core.market_regime_detector import tinh_chuoi_giai_doan_theo_ngay

    all_symbol_keys = storage.query_all_keys("ohlcv_history")
    if not all_symbol_keys:
        logger.warning("Không có dữ liệu OHLCV nào để tính chuỗi giai đoạn lịch sử — bỏ qua bước này.")
        return

    ohlcv_map_raw = storage.get_latest_many("ohlcv_history", all_symbol_keys)
    du_lieu_theo_ma: dict[str, pd.DataFrame] = {}
    for ma, record in ohlcv_map_raw.items():
        records = record["data"].get("records", [])
        if not records:
            continue
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        du_lieu_theo_ma[ma] = df.sort_values("date").reset_index(drop=True)

    if not du_lieu_theo_ma:
        logger.warning("Không có mã nào đủ dữ liệu OHLCV hợp lệ — bỏ qua bước tính chuỗi giai đoạn lịch sử.")
        return

    # --- Toàn thị trường ---
    chuoi_thi_truong = tinh_chuoi_giai_doan_theo_ngay(du_lieu_theo_ma)
    _luu_chuoi_giai_doan(storage, "thi_truong", chuoi_thi_truong)
    logger.info("Đã lưu chuỗi giai đoạn lịch sử TOÀN THỊ TRƯỜNG (%d ngày).", len(chuoi_thi_truong))

    # --- Theo từng ngành ---
    sector_keys = storage.query_all_keys("symbol_sector")
    sector_map = storage.get_latest_many("symbol_sector", sector_keys)
    nganh_theo_ma = {ma: rec["data"].get("sector") for ma, rec in sector_map.items()}

    tat_ca_nganh = sorted({v for v in nganh_theo_ma.values() if v})
    so_nganh_da_luu = 0
    for nganh in tat_ca_nganh:
        ma_trong_nganh = [
            ma for ma, ng in nganh_theo_ma.items()
            if ng == nganh and ma in du_lieu_theo_ma
        ]
        if not ma_trong_nganh:
            continue
        du_lieu_nganh = {ma: du_lieu_theo_ma[ma] for ma in ma_trong_nganh}
        chuoi_nganh = tinh_chuoi_giai_doan_theo_ngay(du_lieu_nganh)
        if len(chuoi_nganh) > 0:
            _luu_chuoi_giai_doan(storage, nganh, chuoi_nganh)
            so_nganh_da_luu += 1

    logger.info(
        "Đã lưu chuỗi giai đoạn lịch sử cho %d/%d ngành.", so_nganh_da_luu, len(tat_ca_nganh),
    )


def main() -> None:
    print("pm_ck — Phần mềm theo dõi & mô phỏng giao dịch CK Việt Nam")
    print("⚠️  Đây là công cụ THEO DÕI VÀ MÔ PHỎNG — không đặt lệnh giao dịch thật.\n")

    config = load_config()
    run_pipeline(config)


if __name__ == "__main__":
    main()
