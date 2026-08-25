"""
run_full_market.py
=====================
Script BATCH xử lý TOÀN BỘ thị trường (~1.500+ mã) — khác với `main.py`
(chỉ xử lý danh sách mã đã chọn lọc trong config.yaml).

CHỈ DÙNG ĐƯỢC VỚI ADAPTER "vnstock" — vì cần API liệt kê toàn bộ mã theo
ngành (`fetch_symbol_sector_map`), MockDataSource không hỗ trợ việc này.

ĐẶC ĐIỂM QUAN TRỌNG:
    - GIÃN CÁCH THỜI GIAN giữa các mã (mặc định 3 giây) để tránh vượt giới
      hạn request/phút của vnstock (20-60 request/phút tùy cấp độ).
    - LƯU CHECKPOINT sau MỖI mã xử lý xong — nếu bạn dừng giữa chừng
      (Ctrl+C, mất mạng...), lần chạy sau sẽ TỰ ĐỘNG bỏ qua các mã đã
      xong, không phải chạy lại từ đầu.
    - 1 mã lỗi không làm dừng cả batch — log lỗi và tiếp tục mã kế tiếp.

CÁCH CHẠY:
    python run_full_market.py                  # chạy toàn bộ thị trường
    python run_full_market.py --limit 50        # chỉ chạy thử 50 mã đầu
    python run_full_market.py --delay 5          # giãn cách 5 giây/mã
    python run_full_market.py --reset            # bỏ qua checkpoint cũ, chạy lại từ đầu

LƯU Ý: script này CHỈ chạy indicators + pattern_detector cho từng mã
(phần cần gọi API nhiều lần, tốn thời gian). Bước xác định giai đoạn thị
trường theo ngành (market_regime_detector) + khuyến nghị phân bổ vốn
(capital_allocator) nên chạy RIÊNG sau đó bằng một script tổng hợp khác
đọc lại từ storage (không cần gọi mạng nữa, nên rất nhanh) — có thể bổ
sung ở bước tiếp theo nếu cần.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from core.data_collector import DataCollector
from core.indicators import get_indicator_snapshot
from core.pattern_detector import detect_narrowing_pattern
from core.storage import Storage
from main import build_data_collector, compute_precomputed_macro_score, load_config, resolve_storage_path, run_market_regime_history_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_ck.run_full_market")

CHECKPOINT_CATEGORY = "batch_checkpoint"
CHECKPOINT_KEY = "full_market"


def load_checkpoint(storage: Storage) -> set[str]:
    """Đọc danh sách mã ĐÃ xử lý xong từ lần chạy trước (nếu có)."""
    record = storage.get_latest(CHECKPOINT_CATEGORY, CHECKPOINT_KEY)
    if record is None:
        return set()
    return set(record["data"].get("completed_symbols", []))


def save_checkpoint(storage: Storage, completed: set[str]) -> None:
    """Ghi lại danh sách mã đã xử lý xong — gọi sau MỖI mã để đảm bảo
    không mất tiến độ nếu chương trình bị ngắt giữa chừng.
    """
    storage.save(
        CHECKPOINT_CATEGORY, CHECKPOINT_KEY,
        {"completed_symbols": sorted(completed)},
    )


RATE_LIMIT_KEYWORDS = ("rate limit", "giới hạn", "request limit")

# Nhận diện lỗi MẤT KẾT NỐI Supabase (bổ sung 04/08/2026) — hay gặp hơn
# hẳn từ khi tăng lượng dữ liệu tải mỗi mã (từ 2021 tới nay), khiến các
# phiên chạy dài hơn nhiều, dễ bị Supabase tự đóng kết nối nhàn rỗi giữa
# chừng. Dò theo từ khóa trong thông điệp lỗi (giống cách nhận diện rate
# limit ở trên) — không import trực tiếp psycopg2 để không bắt buộc cài
# đặt thư viện đó nếu ai đó chạy ở chế độ SQLite thuần túy.
CONNECTION_ERROR_KEYWORDS = (
    "connection already closed", "connection is closed",
    "server closed the connection", "could not connect", "connection not open",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Nhận diện lỗi giới hạn API của vnstock qua nội dung thông báo lỗi
    (thư viện không cung cấp một class ngoại lệ riêng biệt để bắt theo
    kiểu, nên phải kiểm tra theo từ khóa trong thông điệp lỗi).
    """
    text = str(exc).lower()
    return any(keyword in text for keyword in RATE_LIMIT_KEYWORDS)


def _is_connection_error(exc: BaseException) -> bool:
    """Nhận diện lỗi MẤT KẾT NỐI Supabase (khác với rate limit của vnstock)."""
    text = str(exc).lower()
    return any(keyword in text for keyword in CONNECTION_ERROR_KEYWORDS)


def _save_checkpoint_voi_ket_noi_lai(storage: Storage, completed: set[str], config: dict) -> Storage:
    """Gọi `save_checkpoint()`; nếu bị lỗi mất kết nối, TỰ ĐỘNG mở lại kết
    nối mới rồi thử lại đúng 1 lần — tránh làm sập toàn bộ chương trình
    chỉ vì 1 lần rớt kết nối tạm thời (bổ sung 04/08/2026).

    Trả về đối tượng `Storage` ĐANG DÙNG (có thể là bản mới nếu vừa phải
    kết nối lại) — LUÔN gán ngược lại biến `storage` ở nơi gọi.
    """
    try:
        save_checkpoint(storage, completed)
        return storage
    except Exception as exc:  # noqa: BLE001
        if not _is_connection_error(exc):
            raise
        logger.warning("Kết nối Supabase bị ngắt khi lưu checkpoint -> đang kết nối lại...")
        try:
            storage.close()
        except Exception:  # noqa: BLE001
            pass
        storage_moi = Storage(db_path=resolve_storage_path(config))
        save_checkpoint(storage_moi, completed)
        return storage_moi


def run_full_market(
    config: dict,
    delay_seconds: float = 3.0,
    limit: int | None = None,
    reset_checkpoint: bool = False,
    max_rate_limit_retries: int = 5,
    rate_limit_cooldown_seconds: float = 65.0,
) -> None:
    storage = Storage(db_path=resolve_storage_path(config))
    # SỬA LỖI (05/08/2026): trước đây tự khởi tạo VnstockDataSource() TRỰC
    # TIẾP, KHÔNG tham số — bỏ qua hoàn toàn cấu hình `vnstock_start_date`
    # trong config.yaml (dù build_data_collector() đã đọc đúng cấu hình
    # đó), khiến script LUÔN lấy mặc định ~800 phiên gần nhất bất kể cấu
    # hình. Dùng đúng build_data_collector() để tôn trọng cấu hình.
    collector = build_data_collector(config)

    # --- Ưu tiên danh sách mã + ngành TÙY CHỈNH của người dùng (config.yaml
    #     -> watchlist.symbols) nếu có — thay vì luôn quét TOÀN BỘ ~1.500+
    #     mã theo phân loại ngành mặc định của vnstock. Vẫn giữ nguyên toàn
    #     bộ cơ chế an toàn (checkpoint, chống rate-limit) bên dưới, dù quy
    #     mô danh sách nhỏ hơn nhiều.
    custom_symbols = config.get("watchlist", {}).get("symbols", {})
    if custom_symbols:
        symbol_sector_map = dict(custom_symbols)
        logger.info(
            "Dùng danh sách mã TÙY CHỈNH từ config.yaml (watchlist.symbols): %d mã.",
            len(symbol_sector_map),
        )
    else:
        logger.info("Đang lấy danh sách toàn bộ mã + ngành từ vnstock...")
        symbol_sector_map = collector.source.fetch_symbol_sector_map()
        logger.info("Tổng số mã lấy được: %d", len(symbol_sector_map))

    completed = set() if reset_checkpoint else load_checkpoint(storage)
    if completed:
        logger.info(
            "Tìm thấy checkpoint từ lần chạy trước: %d mã đã xử lý xong -> "
            "sẽ bỏ qua các mã này.",
            len(completed),
        )

    remaining_symbols = [s for s in symbol_sector_map if s not in completed]
    if limit is not None:
        remaining_symbols = remaining_symbols[:limit]

    total_remaining = len(remaining_symbols)
    logger.info("Số mã cần xử lý trong lần chạy này: %d", total_remaining)

    pattern_cfg = config.get("pattern_detector", {})

    for idx, symbol in enumerate(remaining_symbols, start=1):
        sector = symbol_sector_map[symbol]
        success = False
        rate_limit_attempts = 0

        # QUAN TRỌNG: bắt cả `SystemExit` chứ không chỉ `Exception` — vì
        # vnstock ở gói Khách (Guest) đã ghi nhận thực tế là gọi thẳng
        # sys.exit()/tương đương khi vượt giới hạn API, thay vì raise một
        # exception thông thường. Nếu chỉ bắt `Exception`, lỗi này sẽ làm
        # SẬP TOÀN BỘ chương trình (đã xảy ra trong thực tế khi kiểm thử).
        while not success and rate_limit_attempts <= max_rate_limit_retries:
            try:
                df = collector.get_ohlcv(symbol, timeframe="day")

                # Lưu lại lịch sử OHLCV (tối đa 750 phiên gần nhất) để
                # dashboard vẽ biểu đồ nến — không tốn thêm request nào.
                ohlcv_tail = df.tail(1500).copy()  # tăng từ 750 -> 1500 (~2021 tới nay + đệm)
                ohlcv_tail["date"] = ohlcv_tail["date"].astype(str)
                storage.save(
                    "ohlcv_history", symbol,
                    {"records": ohlcv_tail.to_dict(orient="records")},
                )

                snapshot = get_indicator_snapshot(df, config=config.get("indicators", {}))
                storage.save("indicator_snapshot", symbol, snapshot)

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

                storage.save("symbol_sector", symbol, {"sector": sector})
                success = True

            except (Exception, SystemExit) as exc:  # noqa: BLE001
                if _is_connection_error(exc):
                    # Mất kết nối Supabase giữa chừng — KHÔNG phải lỗi dữ
                    # liệu của mã này, nên KHÔNG bỏ qua vĩnh viễn: mở lại
                    # kết nối mới rồi thử lại đúng mã này (bổ sung 04/08/2026).
                    logger.warning(
                        "[%d/%d] %s — Kết nối Supabase bị ngắt giữa chừng "
                        "-> đang kết nối lại rồi thử lại mã này...",
                        idx, total_remaining, symbol,
                    )
                    try:
                        storage.close()
                    except Exception:  # noqa: BLE001
                        pass
                    storage = Storage(db_path=resolve_storage_path(config))
                    rate_limit_attempts += 1  # dùng chung bộ đếm để tránh lặp vô hạn
                    continue
                if _is_rate_limit_error(exc):
                    rate_limit_attempts += 1
                    logger.warning(
                        "[%d/%d] %s — Bị giới hạn API (rate limit). Chờ %.0f "
                        "giây rồi thử lại (lần %d/%d)...",
                        idx, total_remaining, symbol,
                        rate_limit_cooldown_seconds, rate_limit_attempts,
                        max_rate_limit_retries,
                    )
                    time.sleep(rate_limit_cooldown_seconds)
                    continue
                else:
                    logger.exception(
                        "[%d/%d] %s — LỖI (không phải rate limit), bỏ qua "
                        "mã này, tiếp tục mã kế tiếp.",
                        idx, total_remaining, symbol,
                    )
                    break

        if success:
            completed.add(symbol)
            logger.info("[%d/%d] %s (%s) — OK", idx, total_remaining, symbol, sector)
        elif rate_limit_attempts > max_rate_limit_retries:
            logger.error(
                "[%d/%d] %s — Vẫn bị giới hạn API/mất kết nối sau %d lần thử lại. "
                "TẠM BỎ QUA (sẽ tự động thử lại ở lần chạy kế tiếp nhờ checkpoint, "
                "KHÔNG đánh dấu là đã xử lý).",
                idx, total_remaining, symbol, max_rate_limit_retries,
            )
            # CỐ Ý không thêm vào `completed` — để lần resume sau tự thử lại.
        else:
            # Lỗi khác (không phải rate limit) -> coi như đã xử lý xong,
            # bỏ qua vĩnh viễn trong batch này (xem giải thích ở docstring).
            completed.add(symbol)

        storage = _save_checkpoint_voi_ket_noi_lai(storage, completed, config)
        if idx < total_remaining:
            time.sleep(delay_seconds)

    logger.info(
        "Hoàn tất batch dữ liệu. Đã xử lý tổng cộng %d/%d mã trên toàn thị trường.",
        len(completed), len(symbol_sector_map),
    )

    # --- Bước tổng hợp: xác định giai đoạn thị trường THEO NGÀNH THẬT ---
    # Đọc lại dữ liệu ĐÃ LƯU trong storage (indicator_snapshot + ánh xạ
    # ngành từ symbol_sector) — KHÔNG gọi thêm bất kỳ request nào tới
    # vnstock, nên chạy rất nhanh dù có hàng trăm mã.
    aggregate_market_regime_by_sector(storage, collector, symbol_sector_map)

    # --- Bước tính tín hiệu MUA/GIỮ/BÁN cho TỪNG MÃ + báo cáo tổng hợp ---
    # Chạy SAU khi đã có giai đoạn thị trường theo ngành (cần cho điều
    # kiện phủ quyết mua) — đọc lại OHLCV đã lưu, KHÔNG gọi thêm API.
    compute_stock_signals_for_all_symbols(storage, symbol_sector_map)

    # --- Khuyến nghị phân bổ vốn (CẢ 2 LOẠI: đơn giản + ATR14 chi tiết)
    #     cho TỪNG MÃ — trước đây CHỈ chạy qua main.py (watchlist nhỏ),
    #     CHƯA từng chạy cho danh sách quét qua run_full_market.py, khiến
    #     2 mục "Khuyến nghị phân bổ vốn" trên dashboard trống rỗng với
    #     watchlist tùy chỉnh lớn (sự cố thực tế phát hiện 27/07/2026). ---
    compute_capital_allocations_for_all_symbols(storage, symbol_sector_map, config)

    # --- Báo cáo tiêu chí ngắn hạn (VN-Index) + rà soát danh sách vào lệnh
    #     (dùng lại đúng 2 hàm đã có trong main.py, không viết lại) ---
    from main import run_entry_screener_step, run_short_term_signal_step
    run_short_term_signal_step(storage, list(symbol_sector_map.keys()))
    run_entry_screener_step(storage, list(symbol_sector_map.keys()))

    # --- Tính và LƯU LẠI chuỗi giai đoạn Uptrend/Sideway/Downtrend theo
    #     TỪNG NGÀY (toàn thị trường + từng ngành) — bổ sung 05/08/2026,
    #     để dashboard đọc nhanh thay vì phải tính lại mỗi lần (xem
    #     module "Tổng hợp" -> "Lọc theo giai đoạn thị trường/ngành"). ---
    run_market_regime_history_step(storage)

    # --- Tính và LƯU LẠI bảng ensemble 3 phương pháp (Breadth/Peak-
    #     Trough/Markov) — bổ sung 06/08/2026, để dashboard đọc nhanh
    #     thay vì phải fit lại mô hình Markov (tốn tài nguyên) mỗi lần
    #     (xem module "🧭 Ensemble 3 phương pháp"). ---
    from main import run_market_regime_ensemble_step
    run_market_regime_ensemble_step(storage)

    # --- Bộ lọc "📈 Cổ phiếu dài hạn" (backtest 8 bộ chỉ số theo giai đoạn)
    #     — bước RẤT NẶNG (Ensemble cần fit Markov cho từng mã, ước tính
    #     ~26 giây/mã) nhưng TỰ CHECKPOINT theo mã qua chính storage (mã đã
    #     có kết quả sẽ được bỏ qua), nên an toàn để chạy lại/resume. ---
    from main import run_long_term_screener_step
    run_long_term_screener_step(storage, symbol_sector_map)

    storage.close()
    logger.info("Hoàn tất toàn bộ pipeline (dữ liệu + giai đoạn thị trường theo ngành).")


CONFIDENCE_TO_FLOAT = {"CAO": 0.9, "TRUNG_BINH": 0.6, "THAP": 0.3}


def compute_capital_allocations_for_all_symbols(
    storage: Storage, symbol_sector_map: dict[str, str], config: dict,
) -> None:
    """Chạy CẢ 2 module khuyến nghị phân bổ vốn (đơn giản `capital_allocator.py`
    + ATR14 chi tiết `capital_allocation_engine.py`) cho TỪNG MÃ trong
    `symbol_sector_map` — dùng lại dữ liệu ĐÃ LƯU (OHLCV, indicator_snapshot,
    pattern_result, market_regime_quant theo ngành), KHÔNG gọi thêm API.

    Tái sử dụng NGUYÊN 2 hàm đã có trong `main.py`
    (`run_allocation_step`, `run_capital_allocation_v2_step`) — không viết
    lại logic tính toán, chỉ chuyển đổi định dạng `regime_quant` (định
    lượng, "trang_thai" viết hoa) sang định dạng tương thích với
    `capital_allocator.py` cũ (định tính, "regime" viết thường).
    """
    from main import run_allocation_step, run_capital_allocation_v2_step

    nav = config.get("paper_portfolio", {}).get("initial_cash", 100_000_000)

    logger.info("Đang tính khuyến nghị phân bổ vốn (2 loại) cho từng mã...")
    count_simple = 0
    count_v2 = 0

    for symbol, sector in symbol_sector_map.items():
        ohlcv_record = storage.get_latest("ohlcv_history", symbol)
        snapshot_record = storage.get_latest("indicator_snapshot", symbol)
        if ohlcv_record is None or snapshot_record is None:
            continue

        records = ohlcv_record["data"].get("records", [])
        if len(records) < 30:
            continue

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        snapshot = snapshot_record["data"]

        pattern_record = storage.get_latest("pattern_result", symbol)
        pattern_result = pattern_record["data"] if pattern_record else None

        regime_record = storage.get_latest("market_regime_quant", sector) if sector else None
        regime_quant_data = regime_record["data"] if regime_record else {}

        # --- Bản tương thích cho capital_allocator.py (đơn giản) ---
        trang_thai = regime_quant_data.get("trang_thai")
        regime_result_compat = {
            "regime": trang_thai.lower() if trang_thai else None,
            "confidence": CONFIDENCE_TO_FLOAT.get(regime_quant_data.get("do_tin_cay"), 0.5),
            "affected_sectors": [],
        }

        # --- Đọc lại tính cách giao dịch ĐÃ TÍNH SẴN ở bước tín hiệu
        #     mua/bán (compute_stock_signals_for_all_symbols chạy TRƯỚC
        #     hàm này) — không tính lại lần 2 cho cùng 1 mã. ---
        character_record = storage.get_latest("stock_character", symbol)
        tinh_cach = character_record["data"] if character_record else None

        try:
            result_simple = run_allocation_step(
                storage, symbol, sector, snapshot, pattern_result,
                regime_result_compat, nav, config,
                tinh_cach=tinh_cach,
            )
            if result_simple is not None:
                count_simple += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] Lỗi tính phân bổ vốn đơn giản: %s", symbol, exc)

        if regime_record is not None:
            try:
                result_v2 = run_capital_allocation_v2_step(
                    storage, symbol, sector, snapshot, df,
                    regime_quant_data, nav, config,
                )
                if result_v2 is not None:
                    count_v2 += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("[%s] Lỗi tính phân bổ vốn ATR14: %s", symbol, exc)

    logger.info(
        "Đã tính phân bổ vốn: %d/%d mã (đơn giản), %d/%d mã (ATR14 chi tiết).",
        count_simple, len(symbol_sector_map), count_v2, len(symbol_sector_map),
    )


def compute_stock_signals_for_all_symbols(
    storage: Storage, symbol_sector_map: dict[str, str]
) -> None:
    """Tính tín hiệu MUA/GIỮ/BÁN (`core.stock_signal_engine`) cho TỪNG MÃ
    trong `symbol_sector_map` — dùng OHLCV + giai đoạn thị trường của
    ngành đã tính sẵn trong storage, không gọi thêm API.

    QUAN TRỌNG: PHẢI nhận `symbol_sector_map` của ĐÚNG lần chạy hiện tại
    làm tham số — TUYỆT ĐỐI không tự suy ra danh sách mã bằng cách đọc
    `storage.query_all_keys("symbol_sector")` (đọc TOÀN BỘ lịch sử từng
    lưu), vì storage là APPEND-ONLY và tích lũy dữ liệu qua nhiều lần
    chạy khác nhau (vd. lần trước quét toàn thị trường ~1500 mã theo
    ngành vnstock, lần này quét danh sách tùy chỉnh nhỏ hơn) — đọc toàn
    bộ lịch sử sẽ TRỘN LẪN dữ liệu cũ/mới, gây sai lệch nghiêm trọng số
    lượng mã và ngành trong báo cáo (sự cố thực tế đã xảy ra ngày
    27/07/2026 — 718 mã thay vì ~210, 42 ngành thay vì 17).
    """
    from core.capital_allocation_engine import find_support_resistance
    from core.stock_signal_engine import (
        InvalidStockSignalError,
        build_signal_summary_report,
        evaluate_stock_signal,
    )

    logger.info("Đang tính tín hiệu mua/bán cho từng mã...")

    evaluations = []

    for symbol, sector in symbol_sector_map.items():
        ohlcv_record = storage.get_latest("ohlcv_history", symbol)
        if ohlcv_record is None:
            continue

        records = ohlcv_record["data"].get("records", [])
        if len(records) < 30:  # cần tối thiểu để tính chỉ báo
            continue

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        regime_record = storage.get_latest("market_regime_quant", sector) if sector else None
        regime_data = regime_record["data"] if regime_record else {}

        try:
            support, resistance = find_support_resistance(df, lookback=60)
        except Exception:  # noqa: BLE001
            support, resistance = None, None

        try:
            result = evaluate_stock_signal(
                symbol=symbol, df=df,
                macro_score=regime_data.get("macro_score"),
                market_regime=regime_data.get("trang_thai"),
                resistance_level=resistance, support_level=support,
                fundamentals=None,  # chưa có nguồn dữ liệu tài chính, xem docstring module
            )
        except InvalidStockSignalError:
            continue

        # --- Điều chỉnh theo tính cách giao dịch (dùng lại đúng hàm đã có
        #     trong main.py, không viết lại) ---
        from main import run_stock_character_step

        tinh_cach = run_stock_character_step(storage, symbol, df)
        if tinh_cach is not None:
            from core.character_integration import dieu_chinh_tin_hieu_theo_tinh_cach
            result = dieu_chinh_tin_hieu_theo_tinh_cach(result, tinh_cach)

        storage.save("stock_signal", symbol, result)
        evaluations.append(result)

    if evaluations:
        summary_report = build_signal_summary_report(evaluations)
        storage.save("signal_summary_report", "latest", summary_report)
        logger.info(
            "Báo cáo tổng hợp tín hiệu: %d MUA, %d BÁN CẮT LỖ, %d BÁN CHỐT LỜI, %d GIỮ/THEO DÕI (/%d mã).",
            len(summary_report["mua"]), len(summary_report["ban_cat_lo"]),
            len(summary_report["ban_chot_loi"]), len(summary_report["giu_theo_doi"]),
            summary_report["tong_so_ma"],
        )


def aggregate_market_regime_by_sector(
    storage: Storage, collector: DataCollector, symbol_sector_map: dict[str, str]
) -> None:
    """Tổng hợp % Breadth EMA200 + trạng thái thị trường (mô hình 3 lớp
    định lượng) cho TỪNG NGÀNH, dựa trên `symbol_sector_map` của ĐÚNG lần
    chạy hiện tại — không gọi thêm request nào tới nguồn dữ liệu.

    QUAN TRỌNG: xem cảnh báo tương tự ở docstring
    `compute_stock_signals_for_all_symbols()` — PHẢI nhận `symbol_sector_map`
    làm tham số, KHÔNG tự đọc `storage.query_all_keys("symbol_sector")`
    (sẽ trộn lẫn dữ liệu từ các lần chạy khác nhau trên cùng 1 database).

    Tính CẢ Lớp 3 (MA50/200 cross, ADX, Band Width) ở mức ngành, bằng
    cách tổng hợp (biểu quyết đa số / trung bình cộng) chỉ báo Lớp 3 của
    TỪNG MÃ trong ngành — dùng lại OHLCV đã lưu (`ohlcv_history`), không
    cần gọi thêm API nào. Xem `core.market_breadth.aggregate_layer3_indicators_for_group`.
    """
    from core.market_regime_detector import detect_market_regime, detect_market_regime_quant

    logger.info("Đang tổng hợp giai đoạn thị trường theo ngành từ dữ liệu đã lưu...")

    sector_to_symbols: dict[str, list[str]] = {}
    for symbol, sector in symbol_sector_map.items():
        if sector:
            sector_to_symbols.setdefault(sector, []).append(symbol)

    # --- Dọn dẹp bản ghi NGÀNH CŨ không còn thuộc lần chạy hiện tại ---
    # `market_regime_quant` được keyed theo TÊN NGÀNH — nếu không dọn, các
    # ngành từ lần chạy TRƯỚC (vd. quét toàn thị trường theo ngành vnstock)
    # sẽ tồn tại VĨNH VIỄN song song với ngành của lần chạy hiện tại (vd.
    # watchlist tùy chỉnh), khiến dashboard hiển thị LẪN LỘN ngành cũ + mới
    # (sự cố thực tế 27/07/2026: 42 ngành thay vì 17).
    existing_sectors = set(storage.query_all_keys("market_regime_quant"))
    stale_sectors = existing_sectors - set(sector_to_symbols.keys())
    for stale_sector in stale_sectors:
        storage.delete_key("market_regime_quant", stale_sector)
    if stale_sectors:
        logger.info(
            "Đã dọn dẹp %d ngành cũ không còn thuộc lần chạy hiện tại: %s",
            len(stale_sectors), sorted(stale_sectors),
        )

    # Tương tự cho market_regime (mô hình định tính cũ) — xem mục 4b CLAUDE.md
    existing_sectors_qualitative = set(storage.query_all_keys("market_regime"))
    stale_sectors_qualitative = existing_sectors_qualitative - set(sector_to_symbols.keys())
    for stale_sector in stale_sectors_qualitative:
        storage.delete_key("market_regime", stale_sector)

    macro_points = collector.get_macro_data()

    # --- Điểm vĩ mô CHI TIẾT (macro_score_engine) — tính MỘT LẦN, dùng
    #     chung cho mọi ngành (vì đây là chỉ số vĩ mô toàn thị trường,
    #     không phụ thuộc ngành cụ thể) ---
    precomputed_macro_score = compute_precomputed_macro_score(storage)
    if precomputed_macro_score is not None:
        logger.info("Điểm vĩ mô chi tiết (macro_score_engine): %.2f", precomputed_macro_score)

    for sector, symbols_in_sector in sector_to_symbols.items():
        group_snapshots = []
        for symbol in symbols_in_sector:
            record = storage.get_latest("indicator_snapshot", symbol)
            if record is not None:
                group_snapshots.append(record["data"])

        if not group_snapshots:
            continue

        # --- Tải OHLCV đã lưu của từng mã trong ngành để tính Lớp 3 thật ---
        layer3_indicators = None
        try:
            from core.market_breadth import aggregate_layer3_indicators_for_group

            ohlcv_by_symbol = {}
            for symbol in symbols_in_sector:
                record = storage.get_latest("ohlcv_history", symbol)
                if record is None:
                    continue
                records = record["data"].get("records", [])
                if len(records) < 210:  # cần đủ cho MA200/ADX/Band Width
                    continue
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                ohlcv_by_symbol[symbol] = df

            if ohlcv_by_symbol:
                layer3_indicators = aggregate_layer3_indicators_for_group(ohlcv_by_symbol)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] Không tính được Lớp 3 theo ngành: %s", sector, exc)

        regime_quant_result = detect_market_regime_quant(
            macro_context=macro_points,
            group_snapshots=group_snapshots,
            layer3_indicators=layer3_indicators,
            group_name=sector,
            precomputed_macro_score=precomputed_macro_score,
        )
        storage.save("market_regime_quant", sector, regime_quant_result)
        logger.info(
            "[%s] (%d mã) Giai đoạn thị trường: %s (độ tin cậy=%s, breadth=%s%%)",
            sector, len(group_snapshots), regime_quant_result["trang_thai"],
            regime_quant_result["do_tin_cay"], regime_quant_result["breadth_pct"],
        )

        # --- Mô hình định tính (cũ, đơn giản hơn) — tính SONG SONG để
        #     mục "Giai đoạn thị trường (định tính)" trên dashboard cũng
        #     có dữ liệu khi chạy qua run_full_market.py, không chỉ khi
        #     chạy qua main.py (sự cố thực tế 28/07/2026 — trước đây
        #     KHÔNG ghi vào category "market_regime" ở đây). ---
        try:
            regime_result = detect_market_regime(
                macro_context=macro_points,
                sector_price_data=group_snapshots,
                sector_name=sector,
            )
            storage.save("market_regime", sector, regime_result)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] Không tính được mô hình định tính: %s", sector, exc)

    logger.info("Hoàn tất tổng hợp giai đoạn thị trường cho %d ngành.", len(sector_to_symbols))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy batch lấy dữ liệu + chỉ báo + mô hình cho toàn bộ thị trường."
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Số giây giãn cách giữa các mã (mặc định 3.0 — an toàn với giới hạn request của vnstock).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Chỉ chạy thử N mã đầu tiên (dùng để test trước khi chạy toàn bộ).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Bỏ qua checkpoint cũ, chạy lại từ đầu toàn bộ danh sách.",
    )
    parser.add_argument(
        "--rate-limit-cooldown", type=float, default=65.0,
        help="Số giây chờ khi gặp lỗi giới hạn API trước khi thử lại (mặc định 65s).",
    )
    parser.add_argument(
        "--max-rate-limit-retries", type=int, default=5,
        help="Số lần thử lại tối đa khi liên tục gặp lỗi giới hạn API cho 1 mã.",
    )
    args = parser.parse_args()

    print("pm_ck — Chạy batch toàn bộ thị trường")
    print("⚠️  Đây là công cụ THEO DÕI VÀ MÔ PHỎNG — không đặt lệnh giao dịch thật.\n")

    config = load_config()
    run_full_market(
        config,
        delay_seconds=args.delay,
        limit=args.limit,
        reset_checkpoint=args.reset,
        max_rate_limit_retries=args.max_rate_limit_retries,
        rate_limit_cooldown_seconds=args.rate_limit_cooldown,
    )


if __name__ == "__main__":
    main()
