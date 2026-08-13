"""
dashboard/app.py
==================
[Giai đoạn 5 — Giao diện]

Dashboard Streamlit hiển thị toàn cảnh hệ thống theo dõi & mô phỏng giao
dịch chứng khoán Việt Nam. CHỈ ĐỌC dữ liệu qua `core/storage.py` — KHÔNG
gọi trực tiếp `data_collector` để tránh trùng lặp việc gọi API.

Chạy dashboard:
    streamlit run dashboard/app.py

CÁC PHẦN HIỂN THỊ:
    1. Bảng giá theo dõi (watchlist) kèm chỉ báo chính + giá thời gian thực.
    2. Biểu đồ nến Nhật + đường MA/EMA + khối lượng giao dịch cho 1 mã.
    3. Giai đoạn thị trường hiện tại theo từng ngành, kèm lý do và các
       ngành đang bị gắn cờ thận trọng vì yếu tố vĩ mô.
    4. Khuyến nghị phân bổ vốn hiện tại.
    5. Danh sách mã đang có mô hình thu hẹp biên độ, sắp xếp theo độ tin
       cậy giảm dần.
    6. Hiệu suất danh mục mô phỏng (PnL theo thời gian, tỷ trọng thực tế
       vs khuyến nghị).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Khi chạy bằng `streamlit run dashboard/app.py`, Python KHÔNG tự thêm
# thư mục gốc dự án (pm_ck/) vào đường dẫn tìm module — chỉ có thư mục
# chứa file này (dashboard/) mới được thêm mặc định. Vì vậy cần tự thêm
# thư mục gốc vào sys.path để import được package `core`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.storage import Storage


_MAU_XANH_TANG = "color: #16a34a; font-weight: 600;"
_MAU_DO_GIAM = "color: #dc2626; font-weight: 600;"
_MAU_DEN_MAC_DINH = "color: #000000;"


def style_tang_giam(
    df: pd.DataFrame,
    cot_theo_ten: bool = True,
    cot_theo_dau: Optional[list[str]] = None,
    dinh_dang_so: Optional[dict[str, str]] = None,
):
    """Trả về `pandas.Styler` (dùng trực tiếp cho `st.dataframe`) tô màu:
        - XANH LÁ cho các ô ở cột có chữ "tăng" trong tên (ví dụ cột
          "Xác suất tăng sau 3 phiên (%)").
        - ĐỎ cho các ô ở cột có chữ "giảm" trong tên (ví dụ cột "Xác suất
          giảm sau 3 phiên (%)", "% giảm từ pivot").
        - ĐEN (mặc định) cho MỌI ô còn lại — tránh màu xám nhạt do theme,
          chỉ ô tăng/giảm mới có màu riêng.

    `cot_theo_dau`: danh sách thêm các cột % không có chữ "tăng"/"giảm"
    trong tên (ví dụ "% thay đổi TB", "Tỷ lệ phục hồi (%)") — tô màu THEO
    DẤU giá trị: dương -> xanh, âm -> đỏ, bằng 0/NaN -> giữ đen.

    `dinh_dang_so`: dict {tên_cột: chuỗi_định_dạng} để RÚT GỌN số thập
    phân hiển thị (ví dụ {"Giá hiện tại": "{:.1f}"}) — mặc định pandas
    Styler hiện đủ 6 chữ số thập phân nếu không chỉ định, rất rối mắt.

    An toàn nếu cột không tồn tại trong `df` (tự động bỏ qua, không lỗi).
    """
    styler = df.style

    # Màu đen mặc định cho TOÀN BỘ bảng trước — các bước tô xanh/đỏ bên
    # dưới sẽ ghi đè lên đúng những ô cần thiết.
    styler = styler.set_properties(**{"color": "#000000"})

    if dinh_dang_so:
        cot_hop_le_dinh_dang = {k: v for k, v in dinh_dang_so.items() if k in df.columns}
        if cot_hop_le_dinh_dang:
            styler = styler.format(cot_hop_le_dinh_dang, na_rep="—")

    if cot_theo_ten:
        cot_xanh = [c for c in df.columns if "tăng" in c.lower()]
        cot_do = [c for c in df.columns if "giảm" in c.lower()]
        if cot_xanh:
            styler = styler.map(lambda v: _MAU_XANH_TANG, subset=cot_xanh)
        if cot_do:
            styler = styler.map(lambda v: _MAU_DO_GIAM, subset=cot_do)

    if cot_theo_dau:
        cot_hop_le = [c for c in cot_theo_dau if c in df.columns]

        def _theo_dau(v):
            if pd.isna(v):
                return _MAU_DEN_MAC_DINH
            if v > 0:
                return _MAU_XANH_TANG
            if v < 0:
                return _MAU_DO_GIAM
            return _MAU_DEN_MAC_DINH

        if cot_hop_le:
            styler = styler.map(_theo_dau, subset=cot_hop_le)

    return styler


def filter_symbols_by_search(symbols: list[str], search_text: str) -> list[str]:
    """Lọc danh sách mã theo từ khóa tìm kiếm (không phân biệt hoa/thường,
    khớp một phần chuỗi con — ví dụ gõ "HP" khớp "HPG"). Trả về nguyên
    danh sách nếu `search_text` rỗng.
    """
    if not search_text or not search_text.strip():
        return symbols
    keyword = search_text.strip().upper()
    return [s for s in symbols if keyword in s.upper()]


def render_search_box_if_needed(
    symbols: list[str], key: str, threshold: int = 5, label: str = "🔍 Tìm mã"
) -> list[str]:
    """Hiện ô tìm kiếm CHỈ KHI danh sách có nhiều hơn `threshold` mã (mặc
    định 5) — tránh làm phiền khi danh sách ngắn, nhưng giúp tìm nhanh khi
    danh sách dài (vd. watchlist 212 mã). Trả về danh sách ĐÃ LỌC theo từ
    khóa người dùng gõ (hoặc nguyên danh sách nếu chưa gõ gì / danh sách
    ngắn không cần tìm kiếm).
    """
    if len(symbols) <= threshold:
        return symbols

    search_text = st.text_input(label, key=key, placeholder="Gõ để lọc, vd: HPG")
    filtered = filter_symbols_by_search(symbols, search_text)

    if search_text and not filtered:
        st.warning(f"Không tìm thấy mã nào khớp với '{search_text}'.")
        return symbols

    return filtered

# Cho phép override qua biến môi trường PM_CK_DB_PATH — dùng để test có
# thể trỏ tới 1 file HOÀN TOÀN TÁCH BIỆT, không bao giờ đụng tới database
# THẬT của người dùng (tránh lặp lại sự cố dữ liệu test bị lẫn vào dữ liệu
# thật do dọn dẹp sau test không đáng tin cậy trên Windows).
#
# THỨ TỰ ƯU TIÊN lấy đường dẫn/kết nối storage (từ cao xuống thấp):
#   1. Biến môi trường PM_CK_DB_PATH — CHỈ dùng cho test, luôn ưu tiên cao nhất.
#   2. st.secrets["SUPABASE_CONNECTION_STRING"] — dùng khi deploy lên
#      Streamlit Community Cloud (nhập qua mục Settings -> Secrets trên
#      web Streamlit Cloud, KHÔNG lưu trong code/GitHub).
#   3. config/config.yaml -> storage.path — dùng khi chạy trên máy cục bộ.
#   4. "./data/pm_ck.db" — phương án dự phòng cuối cùng nếu không đọc
#      được config.yaml (vd. thiếu file, sai định dạng).
#
# SỬA 28/07/2026: trước đây hàm này CHỈ hard-code "./data/pm_ck.db",
# KHÔNG hề đọc config.yaml — nghĩa là đổi config.yaml sang Supabase
# TRƯỚC ĐÂY KHÔNG có tác dụng gì với dashboard (dù main.py/run_full_market.py
# vẫn đọc đúng config.yaml bình thường). Đã sửa để dashboard đọc ĐÚNG
# cùng 1 nguồn với 2 entry point kia.
def _resolve_db_path() -> str:
    env_override = os.environ.get("PM_CK_DB_PATH")
    if env_override:
        return env_override

    try:
        if "SUPABASE_CONNECTION_STRING" in st.secrets:
            return st.secrets["SUPABASE_CONNECTION_STRING"]
    except Exception:  # noqa: BLE001
        pass  # Không có secrets.toml (bình thường khi chạy cục bộ) -> bỏ qua

    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        path = config.get("storage", {}).get("path")
        if path:
            return path
    except Exception:  # noqa: BLE001
        pass  # config.yaml thiếu/lỗi -> dùng phương án dự phòng bên dưới

    return "./data/pm_ck.db"


DB_PATH = _resolve_db_path()


# ==============================================================================
# KẾT NỐI STORAGE (cache để không mở lại kết nối mỗi lần rerun)
# ==============================================================================

@st.cache_resource
def load_storage(db_path: str = DB_PATH) -> Storage:
    return Storage(db_path=db_path)


# ==============================================================================
# PHẦN 1 — BẢNG GIÁ THEO DÕI (WATCHLIST) + CHỈ BÁO CHÍNH
# ==============================================================================

def _fmt_price(value) -> Optional[str]:
    """Định dạng giá cổ phiếu: 2 chữ số thập phân, có dấu phẩy phân cách
    hàng nghìn (ví dụ: 23.20, 1,234.56)."""
    if value is None:
        return None
    return f"{value:,.2f}"


def _fmt_number(value) -> Optional[str]:
    """Định dạng số nguyên (MA/EMA/khối lượng): làm tròn, có dấu phẩy
    phân cách hàng nghìn/triệu (ví dụ: 21,065,067)."""
    if value is None:
        return None
    return f"{value:,.0f}"


def render_watchlist_section(storage: Storage, symbols: list[str]) -> None:
    st.subheader("📈 Bảng giá theo dõi (Watchlist)")

    if not symbols:
        st.info("Chưa có mã nào trong danh sách theo dõi.")
        return

    # Gộp truy vấn (1 lượt gọi cho TẤT CẢ mã) thay vì hỏi từng mã một —
    # xem giải thích chi tiết trong docstring `get_latest_many()`.
    snapshot_map = storage.get_latest_many("indicator_snapshot", symbols)
    realtime_map = storage.get_latest_many("realtime_price", symbols)

    rows = []
    for symbol in symbols:
        record = snapshot_map.get(symbol)
        realtime_record = realtime_map.get(symbol)

        if record is None:
            rows.append({
                "Mã": symbol, "Giá hiện tại": None, "Giá đóng cửa (OHLCV)": None,
                "MA20": None, "EMA50": None, "EMA100": None, "EMA200": None,
                "Volume MA15": None, "Volume MA20": None, "Trên EMA200?": None,
            })
            continue

        data = record["data"]
        realtime_price = (
            realtime_record["data"].get("price") if realtime_record else None
        )
        rows.append({
            "Mã": symbol,
            "Giá hiện tại": _fmt_price(realtime_price),
            "Giá đóng cửa (OHLCV)": _fmt_price(data.get("close")),
            "MA20": _fmt_number(data.get("ma20")),
            "EMA50": _fmt_number(data.get("ema50")),
            "EMA100": _fmt_number(data.get("ema100")),
            "EMA200": _fmt_number(data.get("ema200")),
            "Volume MA15": _fmt_number(data.get("volume_ma_15")),
            "Volume MA20": _fmt_number(data.get("volume_ma_20")),
            "Trên EMA200?": "✅" if data.get("price_above_ema200") else (
                "❌" if data.get("price_above_ema200") is False else "—"
            ),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)
    st.caption(
        "💡 'Giá hiện tại' lấy từ giá khớp lệnh gần nhất (realtime_price). "
        "'Giá đóng cửa (OHLCV)' là giá của phiên gần nhất trong dữ liệu lịch "
        "sử dùng để tính chỉ báo — 2 giá trị có thể khác nhau tùy thời điểm "
        "và độ trễ của nguồn dữ liệu. Mã xử lý qua `run_full_market.py` sẽ "
        "chưa có 'Giá hiện tại' (chỉ `main.py` mới lấy giá thời gian thực, "
        "để tránh tốn thêm request khi quét toàn thị trường)."
    )


# ==============================================================================
# PHẦN 2 — BIỂU ĐỒ NẾN NHẬT + MA/EMA + KHỐI LƯỢNG
# ==============================================================================

def _load_ohlcv_history_df(storage: Storage, symbol: str) -> Optional[pd.DataFrame]:
    """Đọc lại lịch sử OHLCV đã lưu (từ main.py / run_full_market.py) và
    dựng lại thành DataFrame để vẽ biểu đồ.
    """
    record = storage.get_latest("ohlcv_history", symbol)
    if record is None:
        return None

    records = record["data"].get("records", [])
    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def render_chart_section(storage: Storage, symbols: list[str]) -> None:
    st.subheader("🕯️ Biểu đồ nến — Giá, đường trung bình & khối lượng")

    # Cho phép chọn bất kỳ mã nào ĐÃ CÓ lịch sử OHLCV trong storage —
    # không giới hạn theo watchlist gõ tay, vì có thể đã quét toàn bộ thị
    # trường qua run_full_market.py.
    available_symbols = storage.query_all_keys("ohlcv_history")
    if not available_symbols:
        st.info(
            "Chưa có dữ liệu lịch sử giá nào được lưu. Chạy `main.py` hoặc "
            "`run_full_market.py` trước để có dữ liệu vẽ biểu đồ."
        )
        return

    # Ưu tiên đưa các mã trong watchlist lên đầu danh sách chọn cho tiện.
    ordered_symbols = [s for s in symbols if s in available_symbols] + [
        s for s in available_symbols if s not in symbols
    ]
    ordered_symbols = render_search_box_if_needed(
        ordered_symbols, key="chart_symbol_search", label="🔍 Tìm mã để xem biểu đồ"
    )

    col_symbol, col_timeframe, col_theme = st.columns([3, 1, 1])
    with col_symbol:
        selected_symbol = st.selectbox("Chọn mã để xem biểu đồ", ordered_symbols)
    with col_timeframe:
        timeframe_label = st.selectbox(
            "Khung thời gian", ["Ngày", "Tuần", "Tháng"], index=0,
        )
    with col_theme:
        theme_label = st.selectbox("Giao diện", ["Tối", "Sáng"], index=0, key="chart_theme")
    timeframe_map = {"Ngày": "day", "Tuần": "week", "Tháng": "month"}
    timeframe = timeframe_map[timeframe_label]

    # --- Quản lý chú thích sự kiện trên biểu đồ ---
    from datetime import date as date_cls_ann

    from core.chart_annotations import create_annotation

    with st.expander("📝 Chú thích sự kiện trên biểu đồ"):
        with st.form("annotation_form", clear_on_submit=True):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                ann_date = st.date_input("Ngày sự kiện", value=date_cls_ann.today(), key="ann_date")
                ann_scope = st.radio(
                    "Áp dụng cho", [f"Chỉ mã {selected_symbol}", "Toàn thị trường (mọi mã)"],
                    key="ann_scope",
                )
            with col_b:
                ann_text = st.text_area(
                    "Nội dung chú thích", key="ann_text",
                    placeholder="Ví dụ: Mỹ tấn công Iran, ảnh hưởng tâm lý thị trường",
                )
            ann_submitted = st.form_submit_button("Lưu chú thích")
            if ann_submitted:
                if not ann_text or not ann_text.strip():
                    st.error("Nội dung chú thích không được để trống.")
                else:
                    scope_symbol = "" if "Toàn thị trường" in ann_scope else selected_symbol
                    entry = create_annotation(scope_symbol, ann_date, ann_text)
                    storage.save("chart_annotation", entry["annotation_id"], entry)
                    st.success("Đã lưu chú thích.")
                    st.rerun()

        # --- Danh sách chú thích hiện có (áp dụng cho mã đang xem, kể cả loại "toàn thị trường") ---
        all_ann_ids = storage.query_all_keys("chart_annotation")
        relevant_annotations = []
        for ann_id in all_ann_ids:
            record = storage.get_latest("chart_annotation", ann_id)
            if record is None:
                continue
            data = record["data"]
            if data.get("symbol") in ("", selected_symbol):
                relevant_annotations.append(data)

        if relevant_annotations:
            st.caption(f"Chú thích hiện có cho {selected_symbol} (kể cả chú thích chung):")
            for ann in sorted(relevant_annotations, key=lambda a: a["date"], reverse=True):
                col_txt, col_del = st.columns([5, 1])
                scope_note = "🌐 " if not ann.get("symbol") else ""
                col_txt.write(f"**{ann['date']}** {scope_note}— {ann['text']}")
                if col_del.button("🗑️", key=f"del_ann_{ann['annotation_id']}"):
                    storage.delete_key("chart_annotation", ann["annotation_id"])
                    st.rerun()
        else:
            st.info("Chưa có chú thích nào cho mã này.")

    df_daily = _load_ohlcv_history_df(storage, selected_symbol)
    if df_daily is None or df_daily.empty:
        st.info(f"Không có dữ liệu OHLCV cho mã '{selected_symbol}'.")
        return

    from core.indicators import calculate_ema, calculate_ma, calculate_rsi, resample_ohlcv

    df = resample_ohlcv(df_daily, timeframe=timeframe)
    if df.empty:
        st.info(f"Không đủ dữ liệu để gộp theo khung '{timeframe_label}'.")
        return

    ma20 = calculate_ma(df, 20)
    ema50 = calculate_ema(df, 50)
    ema100 = calculate_ema(df, 100)
    ema200 = calculate_ema(df, 200)
    if timeframe != "day" and ema200.isna().all():
        st.caption(
            f"ℹ️ Chưa đủ dữ liệu lịch sử để tính EMA200 ở khung '{timeframe_label}' "
            f"(cần ít nhất 200 {timeframe_label.lower()} — dữ liệu hiện lưu tối đa "
            f"~750 phiên ngày). Các đường MA20/EMA50/EMA100 vẫn hiển thị bình thường."
        )
    rsi14 = calculate_rsi(df, period=14) if len(df) > 14 else None

    # --- Bảng màu theo giao diện đã chọn (Tối kiểu TradingView/fireant, hoặc Sáng nền trắng) ---
    if theme_label == "Tối":
        BG_COLOR = "#131722"
        GRID_COLOR = "#2a2e39"
        TEXT_COLOR = "#ffffff"      # trắng sáng, tương phản rõ trên nền tối
        LEGEND_BG = "rgba(19,23,34,0.6)"
    else:
        BG_COLOR = "#ffffff"
        GRID_COLOR = "#e0e0e0"
        TEXT_COLOR = "#000000"      # đen, tương phản rõ trên nền trắng
        LEGEND_BG = "rgba(255,255,255,0.7)"

    UP_COLOR = "#26a69a"
    DOWN_COLOR = "#ef5350"
    RSI_COLOR = "#9575cd" if theme_label == "Sáng" else "#b39ddb"

    # --- Khung thông tin O/H/L/C + % thay đổi so với phiên trước (kiểu fireant/TradingView) ---
    last = df.iloc[-1]
    prev_close = df.iloc[-2]["close"] if len(df) >= 2 else last["open"]
    change = last["close"] - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    change_color = "#26a69a" if change >= 0 else "#ef5350"
    change_sign = "+" if change >= 0 else ""

    rsi_info = ""
    if rsi14 is not None and not pd.isna(rsi14.iloc[-1]):
        rsi_info = f"&nbsp;&nbsp;RSI(14) <b>{rsi14.iloc[-1]:.1f}</b>"

    st.markdown(
        f"""
        <div style="background-color:{BG_COLOR}; padding:10px 16px; border-radius:6px 6px 0 0;
                    font-family:monospace; font-size:14px; color:{TEXT_COLOR};
                    border:1px solid {GRID_COLOR};">
            <b style="font-size:16px;">{selected_symbol}</b> · {timeframe_label}&nbsp;&nbsp;
            O <b>{last['open']:.2f}</b>&nbsp;
            H <b>{last['high']:.2f}</b>&nbsp;
            L <b>{last['low']:.2f}</b>&nbsp;
            C <b>{last['close']:.2f}</b>&nbsp;&nbsp;
            <span style="color:{change_color};">
                {change_sign}{change:.2f} ({change_sign}{change_pct:.2f}%)
            </span>
            {rsi_info}
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_rsi = rsi14 is not None
    n_rows = 3 if has_rsi else 2
    row_heights = [0.6, 0.2, 0.2] if has_rsi else [0.75, 0.25]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.02,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=selected_symbol,
            increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=ma20, name="MA20",
                   line=dict(width=1.2, color="#e0e0e0" if theme_label == "Tối" else "#616161")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=ema50, name="EMA50",
                   line=dict(width=1.2, color="#ffd54f")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=ema100, name="EMA100",
                   line=dict(width=1.2, color="#42a5f5")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=ema200, name="EMA200",
                   line=dict(width=1.5, color="#ef5350")),
        row=1, col=1,
    )

    # --- Hàng 2: khối lượng giao dịch theo cột, màu theo tăng/giảm phiên đó ---
    volume_up_color = UP_COLOR if theme_label == "Tối" else "#00695c"
    volume_down_color = DOWN_COLOR if theme_label == "Tối" else "#b71c1c"
    volume_colors = [
        volume_up_color if c >= o else volume_down_color
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["volume"], name="Khối lượng",
            marker=dict(color=volume_colors, opacity=1.0),
        ),
        row=2, col=1,
    )

    # --- Đường MA20 cho khối lượng giao dịch ---
    from core.indicators import calculate_volume_ma

    volume_ma20 = calculate_volume_ma(df, 20)
    volume_ma_color = "#ff9800" if theme_label == "Tối" else "#1565c0"
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=volume_ma20, name="Volume MA20",
            line=dict(width=1.3, color=volume_ma_color),
        ),
        row=2, col=1,
    )

    # --- Hàng 3 (nếu có đủ dữ liệu): RSI(14) kèm 2 mốc tham chiếu 30/70 ---
    if has_rsi:
        fig.add_trace(
            go.Scatter(x=df["date"], y=rsi14, name="RSI(14)",
                       line=dict(width=1.3, color=RSI_COLOR)),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#787b86",
                      line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#787b86",
                      line_width=1, row=3, col=1)

    fig.update_layout(
        height=870 if has_rsi else 720,
        template="plotly_dark" if theme_label == "Tối" else "plotly_white",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        xaxis_rangeslider_visible=False,  # tắt rangeslider mặc định (dính vào nến) — sẽ bật riêng ở hàng cuối
        dragmode="pan",  # giữ chuột kéo ngang để di chuyển qua các mốc thời gian (thay vì zoom mặc định)
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            bgcolor=LEGEND_BG,
            font=dict(color=TEXT_COLOR, size=12),  # màu chữ tương phản rõ — đây là chỗ trước đây bị mờ
            bordercolor=GRID_COLOR, borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR, showgrid=True, rangeslider_visible=False,
        color=TEXT_COLOR,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, showgrid=True, title_text="Giá", color=TEXT_COLOR,
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, showgrid=True, title_text="Khối lượng", color=TEXT_COLOR,
        row=2, col=1,
    )
    if has_rsi:
        fig.update_yaxes(
            gridcolor=GRID_COLOR, showgrid=True, title_text="RSI", color=TEXT_COLOR,
            range=[0, 100], row=3, col=1,
        )

    # LƯU Ý: đã BỎ thanh kéo thu nhỏ (rangeslider) — Plotly tự vẽ dải xem
    # trước này ở dạng MỜ/NHỎ theo mặc định (không sửa được rõ hơn), dễ
    # gây hiểu nhầm là biểu đồ khối lượng/RSI bị lỗi mờ. Việc di chuyển
    # xem theo thời gian giờ chỉ dùng: kéo chuột (pan), cuộn chuột
    # (zoom), và các nút bấm chọn nhanh (1T/3T/6T/1N/Tất cả) bên dưới.

    # --- Nút bấm chọn nhanh khoảng thời gian (kiểu TradingView/fireant) ---
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1T", step="month", stepmode="backward"),
                dict(count=3, label="3T", step="month", stepmode="backward"),
                dict(count=6, label="6T", step="month", stepmode="backward"),
                dict(count=1, label="1N", step="year", stepmode="backward"),
                dict(step="all", label="Tất cả"),
            ],
            bgcolor=GRID_COLOR, activecolor="#42a5f5", font=dict(color=TEXT_COLOR, size=11),
            y=1.16, yanchor="bottom", x=0, xanchor="left",  # đặt CAO HƠN legend để không bị đè
        ),
        row=1, col=1,
    )

    # --- Vẽ chú thích sự kiện lên biểu đồ (đường thẳng đứng + điểm đánh dấu) ---
    ANNOTATION_COLOR = "#ffa726"
    df_dates_only = df["date"].dt.normalize()

    for ann in relevant_annotations:
        ann_ts = pd.Timestamp(ann["date"])
        # Tìm phiên gần nhất với ngày chú thích trong dữ liệu đang hiển thị
        # (vì ngày chú thích có thể rơi vào cuối tuần, hoặc không khớp
        # đúng ranh giới tuần/tháng sau khi resample).
        diffs = (df_dates_only - ann_ts).abs()
        nearest_idx = diffs.idxmin()
        nearest_date = df.loc[nearest_idx, "date"]

        # Chỉ vẽ nếu ngày chú thích nằm trong phạm vi dữ liệu đang có (sai
        # số cho phép tối đa ~15 ngày để tránh chú thích "trôi" quá xa khi
        # xem ở khung Tháng).
        if abs((nearest_date - ann_ts).days) > 15:
            continue

        fig.add_shape(
            type="line", x0=nearest_date, x1=nearest_date, y0=0, y1=1,
            yref="paper", xref="x",
            line=dict(color=ANNOTATION_COLOR, dash="dot", width=1.5),
        )
        fig.add_trace(
            go.Scatter(
                x=[nearest_date], y=[df.loc[nearest_idx, "high"] * 1.03],
                mode="markers", marker=dict(symbol="triangle-down", size=11, color=ANNOTATION_COLOR),
                name="Chú thích", hovertext=f"{ann['date']}: {ann['text']}", hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=1,
        )

    st.plotly_chart(
        fig, width='stretch',
        config={
            "displayModeBar": True,   # luôn hiện thanh công cụ (thay vì chỉ hiện khi rê chuột)
            "scrollZoom": True,        # cuộn chuột để phóng to/thu nhỏ trực tiếp, không cần giữ phím
            "displaylogo": False,      # ẩn logo Plotly cho gọn
            "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        },
    )
    st.caption(
        "💡 **Phóng to/thu nhỏ:** cuộn chuột giữa biểu đồ, hoặc dùng nút 🔍+/🔍− ở "
        "thanh công cụ góc trên bên phải. **Di chuyển:** giữ chuột trái và kéo ngang. "
        "**Reset về ban đầu:** double-click vào biểu đồ, hoặc nút 🏠 ở thanh công cụ."
    )


# ==============================================================================
# PHẦN 2 — GIAI ĐOẠN THỊ TRƯỜNG HIỆN TẠI
# ==============================================================================

# Tên ngành song ngữ — khớp ĐÚNG các khóa tiếng Anh trong config.yaml
# (watchlist.symbols). Dùng để hiển thị dạng "Tiếng Việt (english_key)"
# thay vì chỉ hiện khóa tiếng Anh trần trụi.
SECTOR_LABELS = {
    "agriculture": "Nông nghiệp",
    "banking": "Ngân hàng",
    "energy": "Năng lượng",
    "fertilizer_chemical": "Phân bón - Hóa chất",
    "industrial_real_estate": "Bất động sản khu công nghiệp",
    "insurance": "Bảo hiểm",
    "oil_gas": "Dầu khí",
    "pharma": "Dược phẩm",
    "public_investment": "Đầu tư công",
    "real_estate": "Bất động sản",
    "retail_consumer": "Bán lẻ - Tiêu dùng",
    "seafood_textile": "Thủy sản - Dệt may",
    "securities": "Chứng khoán",
    "shipping_port": "Cảng biển - Vận tải biển",
    "steel": "Thép",
    "technology": "Công nghệ",
    "vn30_other": "VN30 khác",
}


def _nhan_nganh(sector_key: str) -> str:
    """Trả về nhãn hiển thị song ngữ "Tiếng Việt (khóa_tiếng_anh)" cho 1
    khóa ngành — dùng SECTOR_LABELS ở trên. Nếu khóa không nằm trong bảng
    (thường là do dữ liệu CŨ còn sót lại trong Supabase từ giai đoạn phát
    triển trước, lưu trực tiếp tên tiếng Việt thay vì khóa tiếng Anh chuẩn
    hóa theo config.yaml hiện tại) -> hiện nguyên văn kèm cảnh báo, KHÔNG
    tự đoán ghép cặp vì có thể gán sai ngành.
    """
    if sector_key in SECTOR_LABELS:
        return f"{SECTOR_LABELS[sector_key]} ({sector_key})"
    return f"{sector_key} ⚠️ (dữ liệu cũ, chưa chuẩn hóa)"


def render_market_regime_section(storage: Storage) -> None:
    """Hiển thị giai đoạn thị trường (định tính) — ĐÃ CẬP NHẬT (31/07/2026):
        1. Thêm chỉ số VNINDEX (vị trí so với EMA200) hiển thị riêng ở đầu.
        2. Tự động lấy TẤT CẢ ngành hiện có dữ liệu (không cần người dùng
           gõ tay danh sách ngành ở sidebar như thiết kế cũ).
        3. Với mỗi ngành, hiện rõ DANH SÁCH MÃ đang ở TRÊN đường EMA200
           (và danh sách mã đang dưới, để đối chiếu).
    """
    st.subheader("📊 Giai đoạn thị trường hiện tại")

    # --- VNINDEX — hiển thị riêng, luôn ở đầu mục ---
    vnindex_record = storage.get_latest("indicator_snapshot", "VNINDEX")
    if vnindex_record:
        vdata = vnindex_record["data"]
        tren_ema = vdata.get("price_above_ema200")
        trang_thai = "✅ Trên EMA200" if tren_ema is True else (
            "❌ Dưới EMA200" if tren_ema is False else "— Chưa đủ dữ liệu"
        )
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("VNINDEX", _fmt_number(vdata.get("close")) or "—")
        col_v2.metric("EMA200", _fmt_number(vdata.get("ema200")) or "—")
        col_v3.metric("Vị trí so với EMA200", trang_thai)

        # Hiện rõ thời điểm cập nhật lần cuối — để tự kiểm tra dữ liệu có
        # bị cũ hay không (VD: pipeline tự động chưa chạy kịp hôm nay),
        # không cần đoán hoặc hỏi lại.
        # LƯU Ý: timestamp lưu theo giờ MÁY CHẠY SCRIPT lúc ghi dữ liệu —
        # GitHub Actions chạy theo giờ UTC, còn chạy tay trên máy cá nhân
        # thường theo giờ hệ thống (thường là giờ VN). Vì không biết chắc
        # nguồn nào ghi lần cuối, chỉ hiện số phút đã trôi qua (tính theo
        # giờ UTC hiện tại, đúng với nguồn tự động chính là GitHub
        # Actions) kèm giờ gốc y nguyên để đối chiếu thủ công nếu cần.
        raw_ts = vnindex_record.get("timestamp")
        if raw_ts:
            try:
                ts = pd.Timestamp(raw_ts)
                so_phut_truoc = int((pd.Timestamp.now(tz=None) - ts).total_seconds() // 60)

                # Nếu ra số ÂM (vô lý — nghĩa là thời điểm lưu "muộn hơn"
                # hiện tại) -> rất có thể do script ghi bằng giờ ĐỊA
                # PHƯƠNG máy chạy (VD giờ VN, UTC+7) trong khi so sánh ở
                # đây theo giờ UTC của máy chủ Streamlit Cloud, khiến giờ
                # ghi "trông như" đang ở tương lai. Tự động hiệu chỉnh
                # +7 giờ (giả định phổ biến nhất: script chạy trên máy cá
                # nhân đặt tại Việt Nam) và ghi rõ đây là số ĐÃ HIỆU CHỈNH,
                # không phải số gốc, để không gây hiểu lầm dữ liệu tương lai.
                da_hieu_chinh = False
                if so_phut_truoc < 0:
                    so_phut_truoc += 7 * 60
                    da_hieu_chinh = True

                canh_bao_cu = (
                    " ⚠️ **Đã lâu chưa cập nhật — kiểm tra lại pipeline tự động.**"
                    if so_phut_truoc > 90 else ""
                )
                ghi_chu_hieu_chinh = (
                    " (đã tự hiệu chỉnh +7 giờ, giả định script chạy bằng giờ VN)"
                    if da_hieu_chinh else ""
                )
                st.caption(
                    f"🕒 Cập nhật lần cuối (giờ ghi nhận trên máy chạy script): "
                    f"{ts.strftime('%d/%m/%Y %H:%M')} — khoảng {so_phut_truoc} phút trước"
                    f"{ghi_chu_hieu_chinh}.{canh_bao_cu}"
                )
            except Exception:  # noqa: BLE001
                st.caption(f"🕒 Cập nhật lần cuối: {raw_ts}")
    else:
        st.info("Chưa có dữ liệu VNINDEX. Chạy `update_indices.py` hoặc `main.py` trước.")

    st.divider()

    # --- Tự động lấy TOÀN BỘ ngành hiện có dữ liệu (không phụ thuộc danh
    #     sách gõ tay) — dựa trên bảng ánh xạ mã -> ngành đã có sẵn. ---
    all_symbol_sector_keys = storage.query_all_keys("symbol_sector")
    if not all_symbol_sector_keys:
        st.info("Chưa có dữ liệu ngành nào. Chạy `main.py` hoặc `run_full_market.py` trước.")
        return

    sector_map = storage.get_latest_many("symbol_sector", all_symbol_sector_keys)
    symbols_by_sector: dict[str, list[str]] = {}
    for sym, record in sector_map.items():
        sector = record["data"].get("sector")
        if sector:
            symbols_by_sector.setdefault(sector, []).append(sym)

    all_sectors = sorted(symbols_by_sector.keys())
    if not all_sectors:
        st.info("Chưa có dữ liệu ngành nào.")
        return

    nganh_chua_chuan_hoa = [s for s in all_sectors if s not in SECTOR_LABELS]
    if nganh_chua_chuan_hoa:
        st.warning(
            "⚠️ Phát hiện "
            + str(len(nganh_chua_chuan_hoa))
            + " ngành CHƯA khớp với danh sách chuẩn trong `config.yaml` hiện tại: "
            + ", ".join(nganh_chua_chuan_hoa)
            + " — đây thường là dữ liệu ngành CŨ còn sót lại trong Supabase từ trước "
            "khi chuẩn hóa theo `config.yaml`. Chạy lại `run_full_market.py` cho các mã "
            "liên quan sẽ ghi đè bằng dữ liệu ngành mới, khắc phục dứt điểm."
        )

    regime_emoji = {"uptrend": "🟢", "downtrend": "🔴", "sideway": "🟡"}
    all_affected_sectors: set[str] = set()

    for sector in all_sectors:
        symbols_trong_nganh = sorted(symbols_by_sector.get(sector, []))
        snapshot_map = storage.get_latest_many("indicator_snapshot", symbols_trong_nganh)

        ma_tren_ema200 = sorted(
            sym for sym in symbols_trong_nganh
            if snapshot_map.get(sym) and snapshot_map[sym]["data"].get("price_above_ema200") is True
        )
        ma_duoi_ema200 = sorted(
            sym for sym in symbols_trong_nganh
            if snapshot_map.get(sym) and snapshot_map[sym]["data"].get("price_above_ema200") is False
        )

        record = storage.get_latest("market_regime", sector)
        regime = record["data"].get("regime") if record else None
        confidence = record["data"].get("confidence", 0.0) if record else 0.0
        emoji = regime_emoji.get(regime, "⚪")

        with st.expander(
            f"{emoji} {_nhan_nganh(sector)}: {regime or 'chưa xác định'} "
            f"(độ tin cậy {confidence * 100:.0f}%) — {len(ma_tren_ema200)}/{len(symbols_trong_nganh)} mã trên EMA200"
        ):
            if record:
                for reason in record["data"].get("reasoning", []):
                    st.write(f"- {reason}")
                all_affected_sectors.update(record["data"].get("affected_sectors", []))
            else:
                st.write("Chưa có dữ liệu giai đoạn thị trường cho ngành này.")

            st.markdown(
                f"**✅ Mã đang TRÊN EMA200 ({len(ma_tren_ema200)} mã):** "
                + (", ".join(ma_tren_ema200) if ma_tren_ema200 else "không có mã nào")
            )
            st.markdown(
                f"**❌ Mã đang DƯỚI EMA200 ({len(ma_duoi_ema200)} mã):** "
                + (", ".join(ma_duoi_ema200) if ma_duoi_ema200 else "không có mã nào")
            )

    if all_affected_sectors:
        st.warning(
            "⚠️ Các ngành đang bị gắn cờ THẬN TRỌNG do yếu tố vĩ mô: "
            + ", ".join(sorted(all_affected_sectors))
        )


# ==============================================================================
# NHẬP DỮ LIỆU VĨ MÔ THỦ CÔNG (Fed Funds Rate / Tỷ giá USD-VND)
# ==============================================================================

MACRO_SERIES_LABELS = {
    "fed_rate": "1. Lãi suất Fed Funds Rate (%)",
    "usdvnd_rate": "2. Tỷ giá USD/VND (trung tâm)",
    "cpi_vn": "3. CPI Việt Nam (YoY %)",
    "interbank_overnight": "4. Lãi suất liên ngân hàng — qua đêm (%)",
    "interbank_3m": "4. Lãi suất liên ngân hàng — kỳ hạn 3 tháng (%)",
}

EVENT_OPTIONS = {
    "none": "Không có sự kiện rủi ro nổi bật",
    "escalating_tension": "Căng thẳng leo thang (giai đoạn đầu)",
    "conflict_outbreak": "Xung đột/chiến sự nổ ra hoặc leo thang mạnh",
    "de_escalation_signal": "Có tín hiệu hạ nhiệt/đàm phán tiến triển",
    "positive_resolution": "Sự kiện tích cực xác nhận, rủi ro giải tỏa",
}

GEOPOLITICAL_EVENT_LOG_CATEGORY = "geopolitical_event_entry"

# ==============================================================================
# CÁC SỰ KIỆN NGUY CƠ CAO (bổ sung 06/08/2026) — KHÁC HẲN "Sự kiện địa
# chính trị" ở trên: đây CHỈ LÀ CẢNH BÁO HIỂN THỊ, KHÔNG được tính vào
# công thức Macro Score (core/macro_score_engine.py) — có chủ đích, vì
# đây là các sự kiện CỤ THỂ, TỰ DO KHAI BÁO (tên tự đặt, không theo
# danh mục cố định như EVENT_OPTIONS), phạm vi ảnh hưởng có thể CHỈ 1
# NGÀNH (không hợp lý để gộp vào 1 điểm vĩ mô chung cho toàn thị trường).
# ==============================================================================

HIGH_RISK_EVENT_LOG_CATEGORY = "high_risk_event_entry"

LOAI_ANH_HUONG_OPTIONS = {"ngay": "Ảnh hưởng NGAY", "tuong_lai": "Ảnh hưởng TƯƠNG LAI (có thời điểm bắt đầu)"}
PHAM_VI_ANH_HUONG_OPTIONS = {"nhieu_nganh": "Nhiều ngành / toàn thị trường", "1_nganh": "Chỉ 1 ngành cụ thể"}


def _sync_current_geopolitical_event(storage: Storage) -> None:
    """Đồng bộ trạng thái "sự kiện ĐANG ÁP DỤNG" (đọc bởi
    `core/macro_score_engine.py` qua khóa cũ `manual_macro_setting` /
    `geopolitical_event` — dùng chung bởi `main.py`, `check_macro_score.py`
    và dashboard) — LUÔN lấy theo sự kiện có NGÀY BẮT ĐẦU GẦN NHẤT trong
    danh sách log `geopolitical_event_entry`.

    Gọi hàm này ngay sau khi thêm/sửa/xóa bất kỳ sự kiện nào trong danh
    sách, để khóa "trạng thái hiện tại" luôn khớp với sự kiện mới nhất —
    KHÔNG cần sửa main.py/check_macro_score.py, 2 script đó vẫn đọc đúng
    1 khóa cũ như trước, chỉ khác là giờ khóa đó được cập nhật tự động
    thay vì nhập tay trực tiếp.

    Nếu danh sách rỗng (đã xóa hết sự kiện) -> xóa luôn trạng thái hiện
    tại, macro engine sẽ tự dùng mặc định "none" (không có sự kiện).
    """
    event_ids = storage.query_all_keys(GEOPOLITICAL_EVENT_LOG_CATEGORY)
    if not event_ids:
        storage.delete_key("manual_macro_setting", "geopolitical_event")
        return

    entries = []
    for eid in event_ids:
        record = storage.get_latest(GEOPOLITICAL_EVENT_LOG_CATEGORY, eid)
        if record:
            entries.append({"id": eid, **record["data"]})

    if not entries:
        storage.delete_key("manual_macro_setting", "geopolitical_event")
        return

    # Sự kiện có NGÀY BẮT ĐẦU (start_date, định dạng YYYY-MM-DD) MỚI NHẤT
    # được coi là đang chi phối bối cảnh hiện tại.
    moi_nhat = max(entries, key=lambda e: e["start_date"])
    storage.save("manual_macro_setting", "geopolitical_event", {
        "event_key": moi_nhat["event_key"],
        "note": moi_nhat.get("note", ""),
        "updated_date": moi_nhat["start_date"],
        "source_entry_id": moi_nhat["id"],
    })


def load_macro_series(storage: Storage, series_key: str) -> list[dict]:
    record = storage.get_latest("manual_macro_series", series_key)
    if record is None:
        return []
    return record["data"].get("entries", [])


def save_macro_series(storage: Storage, series_key: str, entries: list[dict]) -> None:
    storage.save("manual_macro_series", series_key, {"entries": entries})


def render_manual_macro_data_section(storage: Storage) -> None:
    st.subheader("🌍 Nhập dữ liệu vĩ mô thủ công")
    st.caption(
        "Đủ 6 nhóm chỉ số cho `core/macro_score_engine.py`. Nhóm nào chưa "
        "nhập sẽ tự dùng giá trị TRUNG TÍNH (không kéo điểm lên/xuống) — "
        "không bắt buộc phải nhập đủ cả 6 mới có kết quả."
    )

    from datetime import date as date_cls
    import uuid

    from core.manual_macro_data import (
        add_cpi_us_entry,
        add_entry,
        compute_consecutive_increases,
        compute_delta_last,
        compute_distance_from_peak_pct,
        compute_ytd_change_pct,
        get_latest_cpi_us_yoy,
        get_recent_cpi_us_mom,
        remove_entry,
    )

    tab_series, tab_cpi_us, tab_target, tab_event, tab_high_risk = st.tabs([
        "📈 Chuỗi số liệu (Fed/FX/CPI VN/Liên NH)",
        "🇺🇸 CPI Mỹ",
        "🎯 Mục tiêu CPI VN",
        "⚠️ Sự kiện địa chính trị",
        "🚨 Các sự kiện nguy cơ cao",
    ])

    # --- TAB 1: các chuỗi đơn giản (date + value) ---
    with tab_series:
        series_choice = st.selectbox(
            "Chọn chỉ số để nhập", list(MACRO_SERIES_LABELS.keys()),
            format_func=lambda k: MACRO_SERIES_LABELS[k], key="macro_series_choice",
        )

        with st.form("macro_entry_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                entry_date = st.date_input("Ngày", value=date_cls.today(), key="macro_entry_date")
            with col2:
                entry_value = st.number_input(
                    "Giá trị", step=0.01, format="%.4f", key="macro_entry_value"
                )
            submitted = st.form_submit_button("Lưu điểm dữ liệu")
            if submitted:
                series = load_macro_series(storage, series_choice)
                updated = add_entry(series, entry_date, entry_value)
                save_macro_series(storage, series_choice, updated)
                st.success(f"Đã lưu {MACRO_SERIES_LABELS[series_choice]} ngày {entry_date}.")
                st.rerun()

        series = load_macro_series(storage, series_choice)
        if not series:
            st.info("Chưa có dữ liệu nào được nhập cho chỉ số này.")
        else:
            chart_df = pd.DataFrame(series)
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            st.line_chart(chart_df.set_index("date")["value"])

            with st.expander("Xem toàn bộ lịch sử đã nhập"):
                st.dataframe(
                    pd.DataFrame(series).sort_values("date", ascending=False),
                    width='stretch', hide_index=True,
                )

            with st.expander("🗑️ Xóa điểm dữ liệu nhập nhầm"):
                for entry in sorted(series, key=lambda e: e["date"], reverse=True):
                    col_txt, col_del = st.columns([4, 1])
                    col_txt.write(f"{entry['date']}: {entry['value']}")
                    if col_del.button("🗑️", key=f"del_{series_choice}_{entry['date']}"):
                        entry_date_obj = date_cls.fromisoformat(entry["date"])
                        updated = remove_entry(series, entry_date_obj)
                        save_macro_series(storage, series_choice, updated)
                        st.rerun()

            st.markdown("**Đại lượng tự tính:**")
            col1, col2, col3 = st.columns(3)
            delta_last = compute_delta_last(series)
            col1.metric(
                "Delta so với lần trước",
                f"{delta_last:+.4f}" if delta_last is not None else "—",
            )
            if series_choice == "usdvnd_rate":
                ytd_change = compute_ytd_change_pct(series)
                weeks_up = compute_consecutive_increases(series)
                distance_peak = compute_distance_from_peak_pct(series)
                col2.metric("% thay đổi từ đầu năm (YTD)", f"{ytd_change:+.2f}%" if ytd_change is not None else "—")
                col3.metric("Số kỳ tăng liên tiếp", weeks_up)
                st.metric(
                    "Khoảng cách tới đỉnh lịch sử",
                    f"{distance_peak:.2f}%" if distance_peak is not None else "—",
                )
            if series_choice in ("interbank_overnight", "interbank_3m"):
                st.caption(
                    "💡 Độ dốc đường cong (spread 3 tháng - qua đêm) sẽ được tự "
                    "động tính khi cả 2 chuỗi (qua đêm + 3 tháng) đều có dữ liệu."
                )

    # --- TAB 2: CPI Mỹ (2 giá trị/điểm: YoY + MoM) ---
    with tab_cpi_us:
        st.caption("Cần CẢ 2 giá trị mỗi lần nhập: CPI YoY% và CPI MoM% (theo tháng công bố).")
        cpi_us_series = load_macro_series(storage, "cpi_us")

        with st.form("cpi_us_entry_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                cpi_us_date = st.date_input("Tháng công bố", value=date_cls.today(), key="cpi_us_date")
            with col2:
                cpi_us_yoy = st.number_input("CPI YoY (%)", step=0.01, format="%.2f", key="cpi_us_yoy")
            with col3:
                cpi_us_mom = st.number_input("CPI MoM (%)", step=0.01, format="%.2f", key="cpi_us_mom")
            submitted_cpi = st.form_submit_button("Lưu điểm dữ liệu")
            if submitted_cpi:
                updated = add_cpi_us_entry(cpi_us_series, cpi_us_date, cpi_us_yoy, cpi_us_mom)
                save_macro_series(storage, "cpi_us", updated)
                st.success(f"Đã lưu CPI Mỹ tháng {cpi_us_date}.")
                st.rerun()

        cpi_us_series = load_macro_series(storage, "cpi_us")
        if not cpi_us_series:
            st.info("Chưa có dữ liệu CPI Mỹ nào được nhập.")
        else:
            with st.expander("Xem toàn bộ lịch sử CPI Mỹ đã nhập"):
                st.dataframe(
                    pd.DataFrame(cpi_us_series).sort_values("date", ascending=False),
                    width='stretch', hide_index=True,
                )
            with st.expander("🗑️ Xóa điểm dữ liệu CPI Mỹ nhập nhầm"):
                for entry in sorted(cpi_us_series, key=lambda e: e["date"], reverse=True):
                    col_txt, col_del = st.columns([4, 1])
                    col_txt.write(f"{entry['date']}: YoY={entry['yoy']}%, MoM={entry['mom']}%")
                    if col_del.button("🗑️", key=f"del_cpi_us_{entry['date']}"):
                        entry_date_obj = date_cls.fromisoformat(entry["date"])
                        updated = remove_entry(cpi_us_series, entry_date_obj)
                        save_macro_series(storage, "cpi_us", updated)
                        st.rerun()
            col1, col2 = st.columns(2)
            col1.metric("CPI YoY mới nhất", f"{get_latest_cpi_us_yoy(cpi_us_series):.2f}%")
            mom_recent = get_recent_cpi_us_mom(cpi_us_series, n=3)
            col2.metric("Số tháng MoM gần nhất có sẵn", len(mom_recent))

    # --- TAB 3: Mục tiêu lạm phát VN (giá trị đơn, không phải chuỗi) ---
    with tab_target:
        st.caption(
            "Mục tiêu kiểm soát lạm phát của Quốc hội/NHNN — cập nhật theo "
            "Nghị quyết hàng năm, KHÔNG hardcode vĩnh viễn."
        )
        current_target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
        current_target = (
            current_target_record["data"]["value"] if current_target_record else 4.0
        )
        new_target = st.number_input(
            "Mục tiêu CPI VN (%/năm)", value=float(current_target),
            step=0.1, format="%.1f", key="cpi_vn_target_input",
        )
        if st.button("Cập nhật mục tiêu", key="update_cpi_target_btn"):
            storage.save("manual_macro_setting", "cpi_vn_target", {"value": new_target})
            st.success(f"Đã cập nhật mục tiêu CPI VN: {new_target}%/năm.")
            st.rerun()
        st.metric("Mục tiêu hiện tại", f"{current_target}%/năm")

    # --- TAB 4: Sự kiện địa chính trị — DANH SÁCH nhiều sự kiện, mỗi sự
    #     kiện có ngày bắt đầu riêng để tính "mốc ảnh hưởng" (số ngày đã
    #     trôi qua), sửa/xóa được TỪNG sự kiện riêng lẻ (không xóa cả
    #     danh sách như thiết kế cũ) ---
    with tab_event:
        st.caption(
            "Đây là điểm DUY NHẤT có thể ghi đè (override) toàn bộ Macro Score "
            "về mức rất âm bất kể các chỉ số khác — cập nhật ngay khi có tin tức "
            "quan trọng, không chờ dữ liệu kinh tế phản ánh (luôn trễ hơn thị trường). "
            "Hệ thống tự động lấy sự kiện có **ngày bắt đầu gần nhất** trong danh "
            "sách dưới đây làm trạng thái áp dụng cho tính điểm vĩ mô."
        )

        # --- Đọc toàn bộ danh sách sự kiện đã nhập ---
        event_ids = storage.query_all_keys(GEOPOLITICAL_EVENT_LOG_CATEGORY)
        events = []
        for eid in event_ids:
            record = storage.get_latest(GEOPOLITICAL_EVENT_LOG_CATEGORY, eid)
            if record:
                events.append({"id": eid, **record["data"]})
        events.sort(key=lambda e: e["start_date"], reverse=True)

        # --- Trạng thái ĐANG ÁP DỤNG (tự động lấy theo sự kiện mới nhất) ---
        current_event_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
        if current_event_record:
            current_data = current_event_record["data"]
            st.info(
                f"📌 Đang áp dụng cho tính điểm vĩ mô: "
                f"**{EVENT_OPTIONS.get(current_data['event_key'], current_data['event_key'])}** "
                f"(bắt đầu {current_data.get('updated_date', '—')}) — "
                f"tự động lấy theo sự kiện có ngày bắt đầu gần nhất bên dưới."
            )
        else:
            st.info("📌 Chưa có sự kiện nào được nhập — đang dùng mặc định: Không có sự kiện rủi ro nổi bật.")

        st.markdown("#### ➕ Thêm sự kiện mới")
        col_new1, col_new2 = st.columns(2)
        with col_new1:
            new_event_key = st.selectbox(
                "Mức độ sự kiện", list(EVENT_OPTIONS.keys()),
                format_func=lambda k: EVENT_OPTIONS[k], key="new_event_key",
            )
        with col_new2:
            new_event_start_date = st.date_input(
                "Ngày bắt đầu sự kiện", value=date_cls.today(), key="new_event_start_date",
            )
        new_event_note = st.text_area("Ghi chú (tùy chọn)", key="new_event_note")

        if st.button("➕ Thêm sự kiện mới", key="add_event_btn"):
            new_id = f"evt_{uuid.uuid4().hex[:12]}"
            storage.save(GEOPOLITICAL_EVENT_LOG_CATEGORY, new_id, {
                "event_key": new_event_key,
                "start_date": new_event_start_date.isoformat(),
                "note": new_event_note,
                "created_at": date_cls.today().isoformat(),
            })
            _sync_current_geopolitical_event(storage)
            st.success(f"Đã thêm sự kiện: {EVENT_OPTIONS[new_event_key]} (bắt đầu {new_event_start_date.isoformat()}).")
            st.rerun()

        st.divider()
        st.markdown("#### 📋 Danh sách sự kiện đã nhập")

        if not events:
            st.info("Chưa có sự kiện nào trong danh sách.")
        else:
            today = date_cls.today()
            rows = []
            for e in events:
                try:
                    so_ngay = (today - date_cls.fromisoformat(e["start_date"])).days
                except ValueError:
                    so_ngay = None
                rows.append({
                    "Ngày bắt đầu": e["start_date"],
                    "Mức độ": EVENT_OPTIONS.get(e["event_key"], e["event_key"]),
                    "Ghi chú": e.get("note") or "—",
                    "Số ngày đã trôi qua (mốc ảnh hưởng)": so_ngay,
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            st.markdown("#### ✏️ Sửa / 🗑️ Xóa 1 sự kiện cụ thể")
            options_nhan = {
                e["id"]: f"{e['start_date']} — {EVENT_OPTIONS.get(e['event_key'], e['event_key'])}"
                for e in events
            }
            selected_id = st.selectbox(
                "Chọn sự kiện cần sửa/xóa", list(options_nhan.keys()),
                format_func=lambda eid: options_nhan[eid], key="edit_event_select",
            )
            selected_event_data = next(e for e in events if e["id"] == selected_id)

            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                edit_event_key = st.selectbox(
                    "Mức độ sự kiện", list(EVENT_OPTIONS.keys()),
                    index=list(EVENT_OPTIONS.keys()).index(selected_event_data["event_key"])
                    if selected_event_data["event_key"] in EVENT_OPTIONS else 0,
                    format_func=lambda k: EVENT_OPTIONS[k], key=f"edit_event_key_{selected_id}",
                )
            with col_edit2:
                edit_start_date = st.date_input(
                    "Ngày bắt đầu sự kiện",
                    value=date_cls.fromisoformat(selected_event_data["start_date"]),
                    key=f"edit_event_date_{selected_id}",
                )
            edit_note = st.text_area(
                "Ghi chú (tùy chọn)", value=selected_event_data.get("note", ""),
                key=f"edit_event_note_{selected_id}",
            )

            col_save, col_del = st.columns(2)
            with col_save:
                if st.button("💾 Lưu thay đổi cho sự kiện này", key=f"save_event_btn_{selected_id}"):
                    storage.save(GEOPOLITICAL_EVENT_LOG_CATEGORY, selected_id, {
                        "event_key": edit_event_key,
                        "start_date": edit_start_date.isoformat(),
                        "note": edit_note,
                        "created_at": selected_event_data.get("created_at", date_cls.today().isoformat()),
                    })
                    _sync_current_geopolitical_event(storage)
                    st.success("Đã lưu thay đổi.")
                    st.rerun()
            with col_del:
                if st.button("🗑️ Xóa sự kiện này", key=f"delete_event_btn_{selected_id}"):
                    storage.delete_key(GEOPOLITICAL_EVENT_LOG_CATEGORY, selected_id)
                    _sync_current_geopolitical_event(storage)
                    st.success("Đã xóa sự kiện này.")
                    st.rerun()

    # --- Rà soát: hiển thị chi tiết công thức tính điểm vĩ mô ---
    with st.expander("🔍 Rà soát chi tiết công thức tính điểm vĩ mô"):
        from core.macro_score_engine import DEFAULT_WEIGHTS
        from core.macro_score_engine import calculate_macro_score as calc_macro_v2
        from core.manual_macro_data import build_full_macro_score_engine_input

        def _load(key):
            record = storage.get_latest("manual_macro_series", key)
            return record["data"]["entries"] if record else []

        target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
        muc_tieu = target_record["data"]["value"] if target_record else None
        event_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
        event_key_current = event_record["data"]["event_key"] if event_record else None

        macro_input = build_full_macro_score_engine_input(
            _load("fed_rate"), _load("usdvnd_rate"),
            cpi_us_series=_load("cpi_us"), cpi_vn_series=_load("cpi_vn"),
            muc_tieu_cpi_vn=muc_tieu,
            interbank_overnight_series=_load("interbank_overnight"),
            interbank_3m_series=_load("interbank_3m"),
            event_key=event_key_current,
        )
        result = calc_macro_v2(macro_input)

        st.markdown("**Input đã tổng hợp** (nhóm chưa nhập dùng giá trị trung tính mặc định):")
        st.json(macro_input, expanded=False)

        st.markdown("**Chi tiết từng thành phần (đã nhân trọng số):**")
        rows = []
        group_names = {
            "fed": "Fed Funds Rate", "cpi_us": "CPI Mỹ", "cpi_vn": "CPI Việt Nam",
            "fx": "Tỷ giá USD/VND", "interbank": "Lãi suất liên ngân hàng", "event": "Sự kiện địa chính trị",
        }
        for key, sub_score in result["chi_tiet_sub_scores"].items():
            weight = DEFAULT_WEIGHTS[key]
            rows.append({
                "Nhóm": group_names.get(key, key),
                "Điểm thô": f"{sub_score:+.3f}",
                "Trọng số": f"{weight:.0%}",
                "Đóng góp": f"{sub_score * weight:+.3f}",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.metric("Macro Score tổng", f"{result['macro_score']:+.3f}", result["nhan"])

    # --- TAB 5: Các sự kiện nguy cơ cao (bổ sung 06/08/2026) — CHỈ LÀ
    #     CẢNH BÁO, KHÔNG tính vào Macro Score. Tự do khai báo tên sự
    #     kiện, thời điểm ảnh hưởng (ngay/tương lai), phạm vi (nhiều
    #     ngành/1 ngành cụ thể). ---
    with tab_high_risk:
        st.caption(
            "🚨 Khai báo các sự kiện nguy cơ cao TỰ DO (không theo danh mục cố định) — "
            "VD: bão lớn, sự cố nhà máy, thay đổi chính sách ngành, sự kiện pháp lý "
            "riêng lẻ... **CHỈ LÀ CẢNH BÁO HIỂN THỊ**, KHÔNG được tính vào công thức "
            "Macro Score — vì phạm vi có thể chỉ ảnh hưởng 1 ngành, không hợp lý để "
            "gộp vào điểm vĩ mô chung của toàn thị trường."
        )

        high_risk_ids = storage.query_all_keys(HIGH_RISK_EVENT_LOG_CATEGORY)
        high_risk_events = []
        for eid in high_risk_ids:
            record = storage.get_latest(HIGH_RISK_EVENT_LOG_CATEGORY, eid)
            if record:
                high_risk_events.append({"id": eid, **record["data"]})

        today_hr = date_cls.today()

        def _trang_thai_su_kien(e: dict) -> str:
            if e["loai_anh_huong"] == "ngay":
                return "🔴 ĐANG ẢNH HƯỞNG"
            ngay_bd = date_cls.fromisoformat(e["ngay_bat_dau_anh_huong"])
            if ngay_bd <= today_hr:
                return "🔴 ĐANG ẢNH HƯỞNG"
            return f"🟡 SẮP ẢNH HƯỞNG (còn {(ngay_bd - today_hr).days} ngày)"

        high_risk_events.sort(
            key=lambda e: e["ngay_bat_dau_anh_huong"] if e["loai_anh_huong"] == "tuong_lai" else e.get("created_at", ""),
            reverse=True,
        )

        st.markdown("#### ➕ Khai báo sự kiện nguy cơ cao mới")
        ten_su_kien_moi = st.text_input("Tên sự kiện (tự đặt)", key="new_high_risk_ten")

        col_hr1, col_hr2 = st.columns(2)
        with col_hr1:
            loai_anh_huong_moi = st.radio(
                "Thời điểm ảnh hưởng", list(LOAI_ANH_HUONG_OPTIONS.keys()),
                format_func=lambda k: LOAI_ANH_HUONG_OPTIONS[k], key="new_high_risk_loai",
            )
            ngay_bat_dau_moi = None
            if loai_anh_huong_moi == "tuong_lai":
                ngay_bat_dau_moi = st.date_input(
                    "Ngày dự kiến bắt đầu ảnh hưởng", value=today_hr, key="new_high_risk_ngay_bd",
                )
        with col_hr2:
            pham_vi_moi = st.radio(
                "Phạm vi ảnh hưởng", list(PHAM_VI_ANH_HUONG_OPTIONS.keys()),
                format_func=lambda k: PHAM_VI_ANH_HUONG_OPTIONS[k], key="new_high_risk_pham_vi",
            )
            nganh_cu_the_moi = None
            if pham_vi_moi == "1_nganh":
                nganh_cu_the_moi = st.selectbox(
                    "Chọn ngành cụ thể", list(SECTOR_LABELS.keys()),
                    format_func=lambda k: SECTOR_LABELS[k], key="new_high_risk_nganh",
                )

        ghi_chu_moi = st.text_area("Ghi chú (tùy chọn)", key="new_high_risk_note")

        if st.button("➕ Thêm sự kiện nguy cơ cao", key="add_high_risk_btn"):
            if not ten_su_kien_moi.strip():
                st.warning("Cần nhập tên sự kiện.")
            else:
                new_id = f"hrisk_{uuid.uuid4().hex[:12]}"
                storage.save(HIGH_RISK_EVENT_LOG_CATEGORY, new_id, {
                    "ten_su_kien": ten_su_kien_moi.strip(),
                    "loai_anh_huong": loai_anh_huong_moi,
                    "ngay_bat_dau_anh_huong": ngay_bat_dau_moi.isoformat() if ngay_bat_dau_moi else today_hr.isoformat(),
                    "pham_vi": pham_vi_moi,
                    "nganh_cu_the": nganh_cu_the_moi,
                    "ghi_chu": ghi_chu_moi,
                    "created_at": today_hr.isoformat(),
                })
                st.success(f"Đã thêm sự kiện nguy cơ cao: {ten_su_kien_moi.strip()}.")
                st.rerun()

        st.divider()
        st.markdown("#### 📋 Danh sách sự kiện nguy cơ cao đã khai báo")

        if not high_risk_events:
            st.info("Chưa có sự kiện nguy cơ cao nào được khai báo.")
        else:
            rows_hr = []
            for e in high_risk_events:
                pham_vi_hien_thi = (
                    SECTOR_LABELS.get(e.get("nganh_cu_the"), e.get("nganh_cu_the"))
                    if e["pham_vi"] == "1_nganh" else "Nhiều ngành / toàn thị trường"
                )
                rows_hr.append({
                    "Tên sự kiện": e["ten_su_kien"],
                    "Trạng thái": _trang_thai_su_kien(e),
                    "Thời điểm": e["ngay_bat_dau_anh_huong"] if e["loai_anh_huong"] == "tuong_lai" else "Ngay lập tức",
                    "Phạm vi": pham_vi_hien_thi,
                    "Ghi chú": e.get("ghi_chu") or "—",
                })
            st.dataframe(pd.DataFrame(rows_hr), width='stretch', hide_index=True)

            st.markdown("#### ✏️ Sửa / 🗑️ Xóa 1 sự kiện cụ thể")
            options_nhan_hr = {e["id"]: e["ten_su_kien"] for e in high_risk_events}
            selected_hr_id = st.selectbox(
                "Chọn sự kiện cần sửa/xóa", list(options_nhan_hr.keys()),
                format_func=lambda eid: options_nhan_hr[eid], key="edit_high_risk_select",
            )
            sel_hr = next(e for e in high_risk_events if e["id"] == selected_hr_id)

            edit_ten = st.text_input("Tên sự kiện", value=sel_hr["ten_su_kien"], key=f"edit_hr_ten_{selected_hr_id}")
            col_ehr1, col_ehr2 = st.columns(2)
            with col_ehr1:
                edit_loai = st.radio(
                    "Thời điểm ảnh hưởng", list(LOAI_ANH_HUONG_OPTIONS.keys()),
                    index=list(LOAI_ANH_HUONG_OPTIONS.keys()).index(sel_hr["loai_anh_huong"]),
                    format_func=lambda k: LOAI_ANH_HUONG_OPTIONS[k], key=f"edit_hr_loai_{selected_hr_id}",
                )
                edit_ngay_bd = None
                if edit_loai == "tuong_lai":
                    edit_ngay_bd = st.date_input(
                        "Ngày dự kiến bắt đầu ảnh hưởng",
                        value=date_cls.fromisoformat(sel_hr["ngay_bat_dau_anh_huong"]),
                        key=f"edit_hr_ngay_{selected_hr_id}",
                    )
            with col_ehr2:
                edit_pham_vi = st.radio(
                    "Phạm vi ảnh hưởng", list(PHAM_VI_ANH_HUONG_OPTIONS.keys()),
                    index=list(PHAM_VI_ANH_HUONG_OPTIONS.keys()).index(sel_hr["pham_vi"]),
                    format_func=lambda k: PHAM_VI_ANH_HUONG_OPTIONS[k], key=f"edit_hr_phamvi_{selected_hr_id}",
                )
                edit_nganh = None
                if edit_pham_vi == "1_nganh":
                    sector_keys_list = list(SECTOR_LABELS.keys())
                    default_idx = (
                        sector_keys_list.index(sel_hr["nganh_cu_the"])
                        if sel_hr.get("nganh_cu_the") in sector_keys_list else 0
                    )
                    edit_nganh = st.selectbox(
                        "Chọn ngành cụ thể", sector_keys_list, index=default_idx,
                        format_func=lambda k: SECTOR_LABELS[k], key=f"edit_hr_nganh_{selected_hr_id}",
                    )
            edit_ghi_chu = st.text_area(
                "Ghi chú", value=sel_hr.get("ghi_chu", ""), key=f"edit_hr_note_{selected_hr_id}",
            )

            col_save_hr, col_del_hr = st.columns(2)
            with col_save_hr:
                if st.button("💾 Lưu thay đổi", key=f"save_hr_btn_{selected_hr_id}"):
                    storage.save(HIGH_RISK_EVENT_LOG_CATEGORY, selected_hr_id, {
                        "ten_su_kien": edit_ten.strip(),
                        "loai_anh_huong": edit_loai,
                        "ngay_bat_dau_anh_huong": edit_ngay_bd.isoformat() if edit_ngay_bd else today_hr.isoformat(),
                        "pham_vi": edit_pham_vi,
                        "nganh_cu_the": edit_nganh,
                        "ghi_chu": edit_ghi_chu,
                        "created_at": sel_hr.get("created_at", today_hr.isoformat()),
                    })
                    st.success("Đã lưu thay đổi.")
                    st.rerun()
            with col_del_hr:
                if st.button("🗑️ Xóa sự kiện này", key=f"delete_hr_btn_{selected_hr_id}"):
                    storage.delete_key(HIGH_RISK_EVENT_LOG_CATEGORY, selected_hr_id)
                    st.success("Đã xóa sự kiện này.")
                    st.rerun()



def render_vcp_scan_section(storage: Storage) -> None:
    """Hiển thị kết quả rà soát mô hình co hẹp biên độ (Volatility
    Contraction Pattern — VCP) cho XAUUSD (qua PAXGUSDT) và BTC/USD (qua
    BTCUSDT) — `core.volatility_contraction_scanner`, dữ liệu do
    `update_vcp.py` tính và lưu định kỳ.

    CHỈ RÀ SOÁT MẪU HÌNH KỸ THUẬT THAM KHẢO — KHÔNG phải tín hiệu giao
    dịch hay khuyến nghị đầu tư.
    """
    st.subheader("📉 Rà soát mô hình co hẹp (XAUUSD/BTC)")
    st.caption(
        "⚠️ Rà soát mẫu hình kỹ thuật THAM KHẢO cho vàng thế giới (qua PAXGUSDT — "
        "token bảo chứng 1:1 bằng vàng vật chất trên Binance) và Bitcoin (qua "
        "BTCUSDT) — KHÔNG phải tín hiệu giao dịch hay khuyến nghị đầu tư. Vàng và "
        "Bitcoin đều là tài sản biến động mạnh, mô hình co hẹp không đảm bảo "
        "hướng breakout sẽ xảy ra theo chiều nào."
    )

    for symbol, ten_hien_thi in [("XAUUSD", "🥇 Vàng thế giới (XAUUSD)"), ("BTCUSD", "₿ Bitcoin (BTC/USD)")]:
        record = storage.get_latest("vcp_scan_result", symbol)
        if record is None:
            st.info(f"{ten_hien_thi}: chưa có dữ liệu. Chạy `update_vcp.py` trước.")
            continue

        data = record["data"]
        xac_nhan = data.get("xac_nhan_co_hep")
        icon = "🟢" if xac_nhan else "⚪"

        with st.expander(
            f"{icon} {ten_hien_thi} — khung {data.get('khung_thoi_gian_da_chon', '—')} "
            f"— {'XÁC NHẬN co hẹp' if xac_nhan else 'chưa xác nhận co hẹp'}"
        ):
            if data.get("canh_bao"):
                st.warning(data["canh_bao"])

            chuoi_bac = data.get("chuoi_bac_bien_do") or []
            if chuoi_bac:
                st.markdown("**Chuỗi biên độ (cũ → mới):** " + " → ".join(chuoi_bac))

            col1, col2 = st.columns(2)
            with col1:
                ty_le_giam = data.get("ty_le_giam_tong_the_pct")
                st.metric("Tỷ lệ giảm biên độ tổng thể", f"{ty_le_giam:.1f}%" if ty_le_giam is not None else "—")
            with col2:
                ma20 = data.get("doi_chieu_ma20", {}) or {}
                xu_huong = ma20.get("ma20_xu_huong") or "—"
                st.metric("Xu hướng MA20", xu_huong)

            # --- Biểu đồ nến kèm đánh dấu các đỉnh/đáy đã phát hiện ---
            # Dùng dữ liệu giá ĐÃ LƯU SẴN từ lần chạy update_vcp.py (KHÔNG
            # gọi lại Binance API trực tiếp từ dashboard) — Streamlit
            # Cloud có thể ĐẶT MÁY CHỦ TẠI MỸ như GitHub Actions, cũng có
            # nguy cơ bị Binance chặn (lỗi 451) nếu gọi API sống ở đây.
            du_lieu_gia = data.get("du_lieu_gia") or []
            if du_lieu_gia:
                df_gia = pd.DataFrame(du_lieu_gia)
                df_gia["date"] = pd.to_datetime(df_gia["date"])

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_gia["date"], open=df_gia["open"], high=df_gia["high"],
                    low=df_gia["low"], close=df_gia["close"], name=symbol,
                    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                    increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
                ))

                diem_dinh_day = data.get("diem_dinh_day") or []
                if diem_dinh_day:
                    df_diem = pd.DataFrame(diem_dinh_day)
                    df_diem["date"] = pd.to_datetime(df_diem["date"])
                    dinh = df_diem[df_diem["loai"] == "dinh"]
                    day = df_diem[df_diem["loai"] == "day"]

                    fig.add_trace(go.Scatter(
                        x=dinh["date"], y=dinh["gia"], mode="markers+text",
                        marker=dict(symbol="triangle-down", size=11, color="#ef5350"),
                        text=["Đỉnh"] * len(dinh), textposition="top center",
                        name="Đỉnh cục bộ",
                    ))
                    fig.add_trace(go.Scatter(
                        x=day["date"], y=day["gia"], mode="markers+text",
                        marker=dict(symbol="triangle-up", size=11, color="#26a69a"),
                        text=["Đáy"] * len(day), textposition="bottom center",
                        name="Đáy cục bộ",
                    ))

                # --- Chú thích CHÍNH XÁC các chu kỳ dùng để tính "Tỷ lệ
                #     giảm biên độ tổng thể" — để người xem thấy TẬN MẮT
                #     đâu là chu kỳ ĐẦU và chu kỳ CUỐI đang được so sánh,
                #     không chỉ đọc 1 con số trừu tượng. ---
                chuoi_bien_do_pct = data.get("chuoi_bien_do_pct") or []
                chi_tiet_chu_ky = data.get("chi_tiet_chu_ky") or []
                if chuoi_bien_do_pct and chi_tiet_chu_ky and diem_dinh_day:
                    so_chu_ky = len(chuoi_bien_do_pct)
                    chu_ky_da_xet = chi_tiet_chu_ky[-so_chu_ky:]

                    # Tra giá theo ngày (chuỗi ISO) từ danh sách điểm đỉnh/đáy.
                    gia_theo_ngay = {
                        pd.Timestamp(p["date"]): p["gia"] for p in diem_dinh_day
                    }

                    mau_chu_ky = ["#7e57c2", "#26a69a", "#ffa726", "#42a5f5", "#ec407a", "#8d6e63"]
                    for i, ck in enumerate(chu_ky_da_xet):
                        tu_ngay = pd.Timestamp(ck["tu_ngay"])
                        den_ngay = pd.Timestamp(ck["den_ngay"])
                        gia_tu = gia_theo_ngay.get(tu_ngay)
                        gia_den = gia_theo_ngay.get(den_ngay)
                        if gia_tu is None or gia_den is None:
                            continue

                        la_dau = i == 0
                        la_cuoi = i == len(chu_ky_da_xet) - 1
                        mau = "#d32f2f" if la_dau else ("#2e7d32" if la_cuoi else mau_chu_ky[i % len(mau_chu_ky)])
                        do_day = 3 if (la_dau or la_cuoi) else 1.5

                        fig.add_trace(go.Scatter(
                            x=[tu_ngay, den_ngay], y=[gia_tu, gia_den],
                            mode="lines+text",
                            line=dict(color=mau, width=do_day, dash="dot"),
                            text=[f"Chu kỳ {i + 1}: {ck['bien_do_pct']:.2f}%", ""],
                            textposition="top center",
                            name=(
                                f"Chu kỳ {i + 1} (ĐẦU — {ck['bien_do_pct']:.2f}%)" if la_dau else
                                f"Chu kỳ {i + 1} (CUỐI — {ck['bien_do_pct']:.2f}%)" if la_cuoi else
                                f"Chu kỳ {i + 1} ({ck['bien_do_pct']:.2f}%)"
                            ),
                            showlegend=True,
                        ))

                    # Vùng bao quanh toàn bộ N chu kỳ đang xét.
                    fig.add_vrect(
                        x0=pd.Timestamp(chu_ky_da_xet[0]["tu_ngay"]),
                        x1=pd.Timestamp(chu_ky_da_xet[-1]["den_ngay"]),
                        fillcolor="#9575cd", opacity=0.08, line_width=0,
                        annotation_text=f"Vùng xét {so_chu_ky} chu kỳ gần nhất",
                        annotation_position="top left",
                    )

                    ty_le_giam = data.get("ty_le_giam_tong_the_pct")
                    if ty_le_giam is not None:
                        st.caption(
                            f"🔴 Đường ĐỎ = chu kỳ ĐẦU tiên trong nhóm {so_chu_ky} chu kỳ gần nhất "
                            f"(biên độ {chu_ky_da_xet[0]['bien_do_pct']:.2f}%) · "
                            f"🟢 Đường XANH LÁ = chu kỳ CUỐI/gần nhất "
                            f"(biên độ {chu_ky_da_xet[-1]['bien_do_pct']:.2f}%) · "
                            f"Tỷ lệ giảm = ({chu_ky_da_xet[0]['bien_do_pct']:.2f} − "
                            f"{chu_ky_da_xet[-1]['bien_do_pct']:.2f}) / {chu_ky_da_xet[0]['bien_do_pct']:.2f} "
                            f"× 100 = **{ty_le_giam:.1f}%**"
                        )

                fig.update_layout(
                    height=420, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, width='stretch', key=f"vcp_chart_{symbol}")
            else:
                st.caption("Chưa có dữ liệu giá chi tiết để vẽ biểu đồ (cần chạy lại `update_vcp.py` bản mới nhất).")

            ngay_danh_gia = data.get("ngay_danh_gia", "—")
            st.caption(f"Ngày đánh giá: {ngay_danh_gia}")
            if data.get("canh_bao_phap_ly"):
                st.caption(data["canh_bao_phap_ly"])


def render_hdtl_vn30_section(storage: Storage) -> None:
    """Công cụ tính toán HĐTL VN30 (bổ sung 04/08/2026) — entry range,
    phân bổ vốn (ràng buộc kép: rủi ro 2% NAV + trần ký quỹ), và R:R.
    Toàn bộ input NHẬP TAY (không phụ thuộc dữ liệu tự động) — vì HĐTL
    VN30 cần nguồn dữ liệu phái sinh riêng (HNX/công ty chứng khoán),
    KHÔNG lấy được qua vnstock/Binance đang dùng cho phần còn lại của
    hệ thống.
    """
    st.subheader("📐 HĐTL VN30 — Entry / Phân bổ vốn / R:R")
    st.caption(
        "⚠️ Công cụ TÍNH TOÁN THAM KHẢO — KHÔNG tự động đặt lệnh, không phải khuyến "
        "nghị đầu tư hay đảm bảo lợi nhuận. Phái sinh có đòn bẩy cao, cơ chế thanh "
        "toán bù trừ hàng ngày (mark-to-market) có thể gây lỗ nhanh hơn nhiều so với "
        "cổ phiếu thường. Toàn bộ input bên dưới NHẬP TAY theo dữ liệu bạn tự theo dõi."
    )

    from core.derivatives_trading_engine import (
        HUONG_THEO_TIN_HIEU, InvalidDerivativesError, phan_tich_lenh_hdtl_vn30,
    )

    col1, col2 = st.columns(2)
    with col1:
        nav = st.number_input("NAV (VNĐ)", value=1_500_000_000, min_value=0, step=10_000_000, key="hdtl_nav")
        phong_cach_nhan = st.radio(
            "Phong cách giao dịch", ["Lướt trong ngày", "Giữ theo kịch bản"],
            key="hdtl_phong_cach",
        )
        phong_cach = "luot_trong_ngay" if phong_cach_nhan == "Lướt trong ngày" else "giu_theo_kich_ban"
        kieu_tin_hieu = st.selectbox(
            "Kiểu tín hiệu", list(HUONG_THEO_TIN_HIEU.keys()),
            format_func=lambda k: {
                "BREAKOUT_TANG": "Breakout tăng (LONG)", "BREAKOUT_GIAM": "Breakout giảm (SHORT)",
                "MUA_HO_TRO": "Mua hỗ trợ (LONG)", "BAN_KHANG_CU": "Bán kháng cự (SHORT)",
            }[k],
            key="hdtl_kieu_tin_hieu",
        )
    with col2:
        gia_tham_chieu = st.number_input("Giá tham chiếu (điểm chỉ số)", value=1830.0, min_value=0.1, step=0.5, key="hdtl_gia_tham_chieu")
        atr14 = st.number_input("ATR14 (điểm chỉ số)", value=25.0, min_value=0.1, step=0.5, key="hdtl_atr14")
        gia_cat_lo = st.number_input("Giá cắt lỗ (điểm chỉ số)", value=1812.0, step=0.5, key="hdtl_gia_cat_lo")
        gia_chot_loi_nhap = st.number_input(
            "Giá chốt lời dự kiến (điểm chỉ số) — để 0 nếu chưa xác định",
            value=1899.0, step=0.5, key="hdtl_gia_chot_loi",
        )

    with st.expander("⚙️ Tham số nâng cao (ký quỹ / rủi ro)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            ty_le_ky_quy_pct = st.number_input("Tỷ lệ ký quỹ yêu cầu (%)", value=15.0, min_value=0.1, max_value=99.0, step=0.5, key="hdtl_ty_le_ky_quy")
        with col_b:
            ty_le_ky_quy_toi_da_pct = st.number_input("Trần ký quỹ tối đa (% NAV)", value=50.0, min_value=1.0, max_value=100.0, step=1.0, key="hdtl_tran_ky_quy")
        with col_c:
            rui_ro_moi_lenh_pct = st.number_input("Rủi ro tối đa/lệnh (% NAV)", value=2.0, min_value=0.1, max_value=100.0, step=0.5, key="hdtl_rui_ro_lenh")

    if st.button("📐 Tính toán", key="hdtl_tinh_toan_btn"):
        try:
            ket_qua = phan_tich_lenh_hdtl_vn30(
                nav=nav, phong_cach=phong_cach, kieu_tin_hieu=kieu_tin_hieu,
                gia_tham_chieu=gia_tham_chieu, atr14=atr14, gia_cat_lo=gia_cat_lo,
                gia_chot_loi_du_kien=(gia_chot_loi_nhap if gia_chot_loi_nhap > 0 else None),
                ty_le_ky_quy=ty_le_ky_quy_pct / 100, ty_le_ky_quy_toi_da_nav=ty_le_ky_quy_toi_da_pct / 100,
                rui_ro_moi_lenh_pct=rui_ro_moi_lenh_pct / 100,
            )
        except InvalidDerivativesError as exc:
            st.error(f"⚠️ {exc}")
            return

        st.divider()
        huong_mau = "🟢 LONG" if ket_qua["huong"] == "LONG" else "🔴 SHORT"
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Hướng lệnh", huong_mau)
        col_r2.metric("Vùng vào lệnh", f"{ket_qua['khoang_gia_vao_lenh'][0]:,.1f} — {ket_qua['khoang_gia_vao_lenh'][1]:,.1f}")
        col_r3.metric("Số hợp đồng tối ưu", f"{ket_qua['so_hop_dong']:,}")

        col_r4, col_r5, col_r6 = st.columns(3)
        col_r4.metric("Rủi ro/1 HĐ", f"{ket_qua['rui_ro_tren_1_hd_vnd']:,} đ")
        col_r5.metric("Ký quỹ yêu cầu/1 HĐ", f"{ket_qua['ky_quy_yeu_cau_1_hd_vnd']:,} đ")
        col_r6.metric("Tổng ký quỹ sử dụng", f"{ket_qua['tong_ky_quy_su_dung_vnd']:,} đ ({ket_qua['tong_ky_quy_pct_nav']:.1f}% NAV)")

        nhan_nut_that = {
            "theo_rui_ro_2pct": "Rủi ro tối đa/lệnh",
            "theo_tran_ky_quy": "Trần ký quỹ",
            "theo_gioi_han_quy_dinh": "Giới hạn quy định (500 HĐ/lệnh)",
        }
        st.info(
            f"📌 Số hợp đồng đang bị giới hạn bởi: **{nhan_nut_that.get(ket_qua['nut_that_gioi_han'])}** "
            f"— chi tiết từng ràng buộc: "
            + ", ".join(
                f"{nhan_nut_that.get(k, k)}={v if v is not None else '—'}"
                for k, v in ket_qua["chi_tiet_cac_rang_buoc"].items()
            )
        )

        if ket_qua["ty_le_rr"]:
            danh_gia_mau = {
                "TOT": "🟢 TỐT", "CHAP_NHAN_DUOC": "🟡 CHẤP NHẬN ĐƯỢC",
                "THAP_CAN_XEM_XET_LAI": "🟠 THẤP — CẦN XEM XÉT LẠI", "KHONG_NEN_VAO_LENH": "🔴 KHÔNG NÊN VÀO LỆNH",
            }
            st.metric("Tỷ lệ R:R", f"{ket_qua['ty_le_rr']['rr']:.2f}", danh_gia_mau.get(ket_qua["ty_le_rr"]["danh_gia"]))

        for cb in ket_qua["canh_bao"]:
            st.warning(cb)

        st.caption(ket_qua["canh_bao_phap_ly"])


def _assess_sub_score(sub_score: float) -> tuple[str, str]:
    """Gán icon + nhãn đánh giá định tính cho 1 điểm thành phần vĩ mô."""
    if sub_score >= 1.0:
        return "🟢", "Rất tích cực"
    if sub_score >= 0.3:
        return "🟢", "Tích cực"
    if sub_score >= -0.3:
        return "🟡", "Trung tính"
    if sub_score >= -1.0:
        return "🔴", "Tiêu cực"
    return "🔴", "Tiêu cực mạnh"


def render_market_summary_report_section(storage: Storage) -> None:
    """Báo cáo tổng hợp thị trường chung — gộp cả 3 lớp (vĩ mô, breadth kỹ
    thuật, khuyến nghị phân bổ vốn) thành MỘT báo cáo dễ đọc, thay vì phải
    xem rải rác nhiều bảng riêng lẻ.
    """
    st.subheader("📋 Báo cáo tổng hợp thị trường chung")

    # --- CẢNH BÁO SỰ KIỆN NGUY CƠ CAO (bổ sung 06/08/2026) — hiển thị
    #     NGAY ĐẦU báo cáo cho dễ thấy, CHỈ LÀ CẢNH BÁO, KHÔNG được tính
    #     vào bất kỳ điểm số nào bên dưới (Macro Score/breadth/phân bổ vốn). ---
    from datetime import date as _date_hr
    _today_hr = _date_hr.today()
    _high_risk_ids = storage.query_all_keys(HIGH_RISK_EVENT_LOG_CATEGORY)
    _high_risk_active = []
    for _eid in _high_risk_ids:
        _rec = storage.get_latest(HIGH_RISK_EVENT_LOG_CATEGORY, _eid)
        if not _rec:
            continue
        _e = _rec["data"]
        _dang_hoat_dong = _e["loai_anh_huong"] == "ngay" or _date_hr.fromisoformat(_e["ngay_bat_dau_anh_huong"]) <= _today_hr
        if _dang_hoat_dong:
            _high_risk_active.append(_e)

    if _high_risk_active:
        _dong_canh_bao = []
        for _e in _high_risk_active:
            _pham_vi_txt = (
                SECTOR_LABELS.get(_e.get("nganh_cu_the"), _e.get("nganh_cu_the"))
                if _e["pham_vi"] == "1_nganh" else "nhiều ngành/toàn thị trường"
            )
            _dong_canh_bao.append(f"**{_e['ten_su_kien']}** (phạm vi: {_pham_vi_txt})")
        st.error(
            "🚨 **" + str(len(_high_risk_active)) + " sự kiện nguy cơ cao đang hoạt động** — "
            "CHỈ LÀ CẢNH BÁO, KHÔNG được tính vào điểm số bên dưới:\n\n"
            + "\n\n".join(f"- {d}" for d in _dong_canh_bao)
            + "\n\nXem/quản lý chi tiết tại mục \"🌐 Nhập dữ liệu vĩ mô thủ công\" → tab \"🚨 Các sự kiện nguy cơ cao\"."
        )

    from core.capital_allocation_engine import ALLOCATION_TABLE, calculate_stock_allocation_pct
    from core.macro_score_engine import DEFAULT_WEIGHTS
    from core.macro_score_engine import calculate_macro_score as calc_macro_v2
    from core.manual_macro_data import build_full_macro_score_engine_input
    from core.market_breadth import aggregate_layer3_indicators_for_group, calculate_ema200_breadth
    from core.market_regime_detector import detect_market_regime_quant

    # --- LỚP 1: Điểm vĩ mô ---
    def _load(key):
        record = storage.get_latest("manual_macro_series", key)
        return record["data"]["entries"] if record else []

    target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
    muc_tieu = target_record["data"]["value"] if target_record else None
    event_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
    event_key_current = event_record["data"]["event_key"] if event_record else "none"
    event_note = event_record["data"].get("note", "") if event_record else ""

    macro_input = build_full_macro_score_engine_input(
        _load("fed_rate"), _load("usdvnd_rate"),
        cpi_us_series=_load("cpi_us"), cpi_vn_series=_load("cpi_vn"),
        muc_tieu_cpi_vn=muc_tieu,
        interbank_overnight_series=_load("interbank_overnight"),
        interbank_3m_series=_load("interbank_3m"),
        event_key=event_key_current,
    )
    macro_result = calc_macro_v2(macro_input)

    st.markdown("### Lớp 1 — Điểm vĩ mô (Macro Score)")
    rows = []
    descriptions = {
        "fed": f"Fed Rate delta={macro_input['fed_rate_delta_last_meeting']:+.2f}, dot-plot delta={macro_input['fed_dotplot_delta']:+.2f}",
        "cpi_us": f"CPI Mỹ YoY={macro_input['cpi_us_yoy']:.1f}%, MoM 3T={macro_input['cpi_us_mom_3thang']}",
        "cpi_vn": f"CPI VN YoY={macro_input['cpi_vn_yoy']:.1f}% (mục tiêu {macro_input['muc_tieu_cpi_vn']:.1f}%)",
        "fx": f"YTD={macro_input['fx_ytd_change_pct']:+.1f}%, {macro_input['fx_so_tuan_tang_lien_tiep']} kỳ tăng liên tiếp, cách đỉnh {macro_input['fx_khoang_cach_dinh_pct']:.1f}%",
        "interbank": f"Spread 3T-qua đêm={macro_input['interbank_do_doc_duong_cong']:+.2f}%, thay đổi tuần={macro_input['interbank_thay_doi_tuan_3m']:+.2f}%",
        "event": event_note or "Không có ghi chú",
    }
    group_names = {
        "fed": "Fed Funds Rate", "cpi_us": "CPI Mỹ", "cpi_vn": "CPI Việt Nam",
        "fx": "Tỷ giá USD/VND", "interbank": "Lãi suất liên NH", "event": "Sự kiện địa chính trị",
    }
    for key, sub_score in macro_result["chi_tiet_sub_scores"].items():
        icon, label = _assess_sub_score(sub_score)
        rows.append({
            "Chỉ số": group_names[key],
            "Diễn biến hiện tại": descriptions[key],
            "Đánh giá": f"{icon} {label}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    override_active = macro_result["chi_tiet_sub_scores"]["event"] <= -1.5
    override_note = " (⚠️ CƠ CHẾ OVERRIDE sự kiện đang kích hoạt)" if override_active else ""
    st.info(
        f"**Macro Score hiện tại: {macro_result['macro_score']:+.2f} "
        f"({macro_result['nhan']}){override_note}**"
    )

    # --- LỚP 2 & 3: Xác nhận kỹ thuật TOÀN THỊ TRƯỜNG ---
    st.markdown("### Lớp 2 & 3 — Xác nhận kỹ thuật (toàn thị trường)")

    all_symbols = storage.query_all_keys("indicator_snapshot")
    all_snapshots = []
    ohlcv_by_symbol = {}
    for symbol in all_symbols:
        record = storage.get_latest("indicator_snapshot", symbol)
        if record is not None:
            all_snapshots.append(record["data"])
        ohlcv_record = storage.get_latest("ohlcv_history", symbol)
        if ohlcv_record is not None:
            records = ohlcv_record["data"].get("records", [])
            if len(records) >= 210:
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                ohlcv_by_symbol[symbol] = df

    if not all_snapshots:
        st.info(
            "Chưa có dữ liệu chỉ báo nào trong storage. Chạy `main.py` hoặc "
            "`run_full_market.py` trước để có dữ liệu cho báo cáo này."
        )
        return

    breadth_result = calculate_ema200_breadth(all_snapshots)
    layer3 = (
        aggregate_layer3_indicators_for_group(ohlcv_by_symbol) if ohlcv_by_symbol else {}
    )

    regime_result = detect_market_regime_quant(
        macro_context=[], group_snapshots=all_snapshots,
        layer3_indicators=layer3, group_name="Toàn thị trường",
        precomputed_macro_score=macro_result["macro_score"],
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("% Breadth EMA200", f"{breadth_result['breadth_pct']:.1f}%" if breadth_result["breadth_pct"] else "—")
    col2.metric("MA50/200 Cross", layer3.get("ma_cross", "—"))
    col3.metric("ADX trung bình", f"{layer3['adx']:.1f}" if "adx" in layer3 else "—")

    regime_emoji = {"UPTREND": "🟢", "DOWNTREND": "🔴", "SIDEWAY": "🟡"}
    st.markdown(
        f"**→ Trạng thái: {regime_emoji.get(regime_result['trang_thai'], '⚪')} "
        f"{regime_result['trang_thai']}, độ tin cậy {regime_result['do_tin_cay']}**"
    )
    for r in regime_result["reasoning"]:
        st.caption(f"• {r}")
    for w in regime_result["canh_bao"]:
        st.warning(w)

    # --- Khuyến nghị phân bổ vốn tổng thể ---
    st.markdown("### Khuyến nghị phân bổ vốn (giai đoạn hiện tại)")
    trang_thai = regime_result["trang_thai"]
    cfg = ALLOCATION_TABLE.get(trang_thai)
    if cfg:
        ty_trong = calculate_stock_allocation_pct(trang_thai, regime_result["do_tin_cay"])
        low, high = cfg["co_phieu_range"]
        alloc_rows = [
            {
                "Hạng mục": "Cổ phiếu",
                "Tỷ trọng tham khảo": f"{low*100:.0f}-{high*100:.0f}% (đề xuất cụ thể ~{ty_trong*100:.0f}%)",
            },
            {
                "Hạng mục": "Tiền mặt/kênh trú ẩn",
                "Tỷ trọng tham khảo": f"{(1-high)*100:.0f}-{(1-low)*100:.0f}%",
            },
        ]
        st.dataframe(pd.DataFrame(alloc_rows), width='stretch', hide_index=True)
        st.caption(
            f"Ngành ưu tiên giai đoạn {trang_thai}: {', '.join(cfg['nganh_uu_tien'])}."
        )

    st.caption(
        "⚠️ Báo cáo tổng hợp mang tính THAM KHẢO, không phải khuyến nghị đầu tư — "
        "cần đối chiếu thêm trước khi ra quyết định."
    )


def render_market_regime_quant_section(storage: Storage) -> None:
    """Hiển thị kết quả mô hình 3 LỚP ĐỊNH LƯỢNG (macro score + % Breadth
    EMA200 + đối chiếu Lớp 3) — tự động phát hiện TẤT CẢ ngành đã có dữ

    liệu trong storage (không giới hạn theo danh sách ngành gõ tay ở
    sidebar), vì `run_full_market.py` tạo ra ngành THẬT từ vnstock.
    """
    st.subheader("📐 Giai đoạn thị trường — mô hình 3 lớp định lượng")

    all_sectors = storage.query_all_keys("market_regime_quant")
    if not all_sectors:
        st.info(
            "Chưa có dữ liệu. Chạy `main.py` hoặc `run_full_market.py` để "
            "tính giai đoạn thị trường theo mô hình 3 lớp."
        )
        return

    regime_emoji = {"UPTREND": "🟢", "DOWNTREND": "🔴", "SIDEWAY": "🟡"}
    confidence_emoji = {"CAO": "✅", "TRUNG_BINH": "🔶", "THAP": "⚠️"}

    rows = []
    details_by_sector = {}
    for sector in all_sectors:
        record = storage.get_latest("market_regime_quant", sector)
        if record is None:
            continue
        data = record["data"]
        rows.append({
            "Ngành": sector,
            "Trạng thái": f"{regime_emoji.get(data['trang_thai'], '⚪')} {data['trang_thai']}",
            "Độ tin cậy": f"{confidence_emoji.get(data['do_tin_cay'], '')} {data['do_tin_cay']}",
            "Điểm vĩ mô": f"{data['macro_score']:+.2f}",
            "% Breadth EMA200": (
                f"{data['breadth_pct']:.1f}%" if data.get("breadth_pct") is not None else "—"
            ),
            "Số cảnh báo": len(data.get("canh_bao", [])),
        })
        details_by_sector[sector] = data

    df = pd.DataFrame(rows).sort_values("Ngành")
    st.dataframe(df, width='stretch', hide_index=True)

    for sector, data in sorted(details_by_sector.items()):
        if data.get("canh_bao") or data.get("reasoning"):
            with st.expander(f"Chi tiết — {sector}"):
                if data.get("canh_bao"):
                    for w in data["canh_bao"]:
                        st.warning(w)
                for r in data.get("reasoning", []):
                    st.write(f"- {r}")

    st.caption(
        "💡 Ngưỡng phân loại % Breadth (>80%/>60%/40-60%/<40%/<20%) là điểm khởi "
        "đầu tham khảo theo thông lệ quốc tế — cần backtest lại trên dữ liệu "
        "lịch sử VN-Index trước khi dùng cho quyết định thực tế."
    )


# ==============================================================================
# PHẦN 3 — KHUYẾN NGHỊ PHÂN BỔ VỐN
# ==============================================================================

def render_allocation_section(storage: Storage, symbols: list[str]) -> None:
    st.subheader("💰 Khuyến nghị phân bổ vốn")

    if not symbols:
        st.info("Chưa có mã nào để hiển thị khuyến nghị.")
        return

    rows = []
    for symbol in symbols:
        record = storage.get_latest("allocation_recommendation", symbol)
        if record is None:
            continue
        data = record["data"]
        entry_range = data.get("entry_price_range", {})
        rows.append({
            "Mã": symbol,
            "Tỷ trọng khuyến nghị (%)": f"{data.get('target_pct', 0):.1f}",
            "Vùng entry (thấp)": _fmt_price(entry_range.get("low")),
            "Vùng entry (cao)": _fmt_price(entry_range.get("high")),
            "Cắt lỗ": _fmt_price(data.get("stop_loss")),
            "KL tối đa": _fmt_number(data.get("max_position_size")),
        })

    if not rows:
        st.info("Chưa có khuyến nghị phân bổ vốn nào được lưu.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    # Hiển thị ghi chú chi tiết cho từng mã
    for symbol in symbols:
        record = storage.get_latest("allocation_recommendation", symbol)
        if record is None:
            continue
        notes = record["data"].get("notes", [])
        if notes:
            with st.expander(f"Ghi chú chi tiết — {symbol}"):
                for note in notes:
                    st.write(f"- {note}")


def render_capital_allocation_v2_section(storage: Storage, symbols: list[str]) -> None:
    """Hiển thị khuyến nghị từ module `core/capital_allocation_engine`
    (mô hình 3 đợt giải ngân + ATR14 + hỗ trợ/kháng cự tự động) — khác
    với `render_allocation_section` ở trên (module `capital_allocator`
    cũ, đơn giản hơn). Cả 2 tồn tại song song để tham khảo/đối chiếu.
    """
    st.subheader("📦 Khuyến nghị phân bổ vốn — mô hình 3 lớp + ATR14")

    if not symbols:
        st.info("Chưa có mã nào để hiển thị khuyến nghị.")
        return

    any_data = False
    for symbol in symbols:
        record = storage.get_latest("capital_allocation_v2", symbol)
        if record is None:
            continue
        any_data = True
        data = record["data"]

        with st.expander(
            f"{symbol} — {data['trang_thai_thi_truong']} "
            f"(tỷ trọng khuyến nghị {data['ty_trong_co_phieu_khuyen_nghi'] * 100:.0f}%)"
        ):
            if not data["cac_dot_giai_ngan"]:
                st.info("Không có khuyến nghị giải ngân cho mã này ở thời điểm hiện tại.")
            for dot in data["cac_dot_giai_ngan"]:
                if not dot["danh_sach_ma"]:
                    continue
                for ma_info in dot["danh_sach_ma"]:
                    st.markdown(f"**Đợt {dot['dot']}** (tỷ lệ {dot['ty_le_dot'] * 100:.0f}%)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Vùng vào lệnh",
                        f"{ma_info['khoang_gia_vao_lenh'][0]:,.2f} – "
                        f"{ma_info['khoang_gia_vao_lenh'][1]:,.2f}",
                    )
                    col2.metric("Khối lượng dự kiến", f"{ma_info['khoi_luong_du_kien']:,}")
                    col3.metric(
                        "Vốn phân bổ", f"{ma_info['von_phan_bo']:,.0f}"
                    )
                    col4, col5 = st.columns(2)
                    col4.metric(
                        "Cắt lỗ",
                        f"{ma_info['khoang_cat_lo'][0]:,.2f} – {ma_info['khoang_cat_lo'][1]:,.2f}",
                    )
                    col5.metric(
                        "Chốt lời tham khảo",
                        f"{ma_info['khoang_chot_loi_tham_khao'][0]:,.2f} – "
                        f"{ma_info['khoang_chot_loi_tham_khao'][1]:,.2f}",
                    )

            if data["canh_bao"]:
                for w in data["canh_bao"]:
                    st.warning(w)

            st.caption(
                f"Tổng rủi ro danh mục ước tính: "
                f"{data['tong_rui_ro_danh_muc_hien_tai_pct'] * 100:.2f}% NAV. "
                f"{data['ghi_chu']}"
            )

    if not any_data:
        st.info(
            "Chưa có dữ liệu. Chạy `main.py` để tính khuyến nghị phân bổ vốn "
            "theo mô hình 3 lớp cho watchlist."
        )

    st.caption(
        "⚠️ Đây là khuyến nghị THAM KHẢO có cấu trúc, KHÔNG PHẢI lệnh giao dịch "
        "tự động — mọi lệnh do bạn tự xác nhận và đặt thủ công."
    )


# ==============================================================================
# PHẦN 4 — MÃ CÓ MÔ HÌNH THU HẸP BIÊN ĐỘ (sắp xếp theo confidence giảm dần)
# ==============================================================================

def render_pattern_section(storage: Storage, symbols: Optional[list[str]] = None) -> None:
    st.subheader("📐 Mã đang có mô hình thu hẹp biên độ")

    # SỬA (06/08/2026): 2 cải tiến theo yêu cầu —
    #  1) CHỈ hiện mã đang nằm trong danh sách watchlist 212 mã MỚI NHẤT
    #     (đọc trực tiếp từ config.yaml) — trước đây hiện TẤT CẢ key còn
    #     tồn đọng trong storage "pattern_result", có thể gồm cả mã CŨ
    #     đã bị loại khỏi watchlist từ lâu.
    #  2) LOẠI BỎ mã có khối lượng giao dịch TB 20 phiên < 500.000 —
    #     tránh gợi ý mã quá thanh khoản thấp, khó vào/ra lệnh thực tế.
    from main import load_config

    try:
        config = load_config()
        danh_sach_watchlist = set(config.get("watchlist", {}).get("symbols", {}).keys())
    except Exception:  # noqa: BLE001
        danh_sach_watchlist = None  # không đọc được config -> không lọc theo watchlist

    candidate_symbols = symbols or storage.query_all_keys("pattern_result")
    if danh_sach_watchlist is not None:
        candidate_symbols = [s for s in candidate_symbols if s in danh_sach_watchlist]

    NGUONG_KHOI_LUONG_TOI_THIEU = 500_000
    ohlcv_map = storage.get_latest_many("ohlcv_history", list(candidate_symbols))

    rows = []
    so_ma_bi_loai_thanh_khoan = 0
    for symbol in candidate_symbols:
        record = storage.get_latest("pattern_result", symbol)
        if record is None:
            continue
        data = record["data"]

        khoi_luong_tb20 = None
        ohlcv_record = ohlcv_map.get(symbol)
        if ohlcv_record:
            recs = ohlcv_record["data"].get("records", [])
            if recs:
                khoi_luong_tb20 = float(pd.DataFrame(recs[-20:])["volume"].mean())

        if khoi_luong_tb20 is not None and khoi_luong_tb20 < NGUONG_KHOI_LUONG_TOI_THIEU:
            so_ma_bi_loai_thanh_khoan += 1
            continue

        rows.append({
            "Mã": symbol,
            "Độ tin cậy (%)": round(data.get("confidence", 0.0) * 100, 1),
            "Giá đỉnh tích lũy (breakout ref.)": _fmt_price(data.get("accumulation_high")),
            "Số tháng hình thành": data.get("effective_scan_months"),
            "Khối lượng TB 20 phiên": (
                f"{khoi_luong_tb20:,.0f}" if khoi_luong_tb20 is not None else "—"
            ),
        })

    if not rows:
        st.info("Chưa phát hiện mã nào có mô hình thu hẹp biên độ (sau khi áp dụng các bộ lọc).")
        return

    df = pd.DataFrame(rows).sort_values("Độ tin cậy (%)", ascending=False)
    st.dataframe(df, width='stretch', hide_index=True)

    ghi_chu = [f"Đã áp dụng: chỉ hiện mã trong watchlist {len(danh_sach_watchlist)} mã hiện tại" if danh_sach_watchlist else None]
    if so_ma_bi_loai_thanh_khoan > 0:
        ghi_chu.append(f"đã loại {so_ma_bi_loai_thanh_khoan} mã có khối lượng TB 20 phiên < {NGUONG_KHOI_LUONG_TOI_THIEU:,}")
    ghi_chu_hop_le = [g for g in ghi_chu if g]
    if ghi_chu_hop_le:
        st.caption("ℹ️ " + "; ".join(ghi_chu_hop_le) + ".")


@st.cache_data(ttl=600, show_spinner="Đang quét lịch sử tìm các tình huống tương tự...")
def _compute_recovery_probability_cached(
    df: pd.DataFrame, ma: str, dieu_kien_loc: dict,
    cac_so_phien_du_bao: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Bọc `tinh_xac_suat_phuc_hoi_lich_su()` bằng cache 10 phút — việc
    quét toàn bộ lịch sử (750-1250 phiên) có chi phí tính toán đáng kể,
    không nên chạy lại mỗi lần trang rerun nếu mã/điều kiện lọc không đổi.
    """
    from core.historical_recovery_probability import tinh_xac_suat_phuc_hoi_lich_su

    return tinh_xac_suat_phuc_hoi_lich_su(
        ma, df, dieu_kien_loc=dieu_kien_loc, cac_so_phien_du_bao=cac_so_phien_du_bao
    )


@st.cache_data(ttl=600, show_spinner="Đang quét toàn bộ watchlist tìm mã đứt gãy vùng nền...")
def _compute_base_breakdown_scan_cached(
    _storage: Storage, symbols: tuple[str, ...], params: dict
) -> pd.DataFrame:
    """Bọc `quet_co_phieu_dut_gay_qua_ban()` (Module 7) bằng cache 10
    phút — quét TOÀN BỘ watchlist mỗi mã đều tốn chi phí O(n^2) tìm vùng
    nền, không nên chạy lại mỗi lần trang rerun nếu watchlist/ngưỡng lọc
    không đổi. Tham số `_storage` có dấu gạch dưới để Streamlit KHÔNG cố
    hash đối tượng kết nối DB (không hashable) khi tính cache key.
    """
    from core.base_breakdown_screener import quet_co_phieu_dut_gay_qua_ban

    return quet_co_phieu_dut_gay_qua_ban(
        list(symbols),
        lambda ma: _load_ohlcv_history_df(_storage, ma),
        lookback_vung_nen=params["lookback_vung_nen"],
        min_ngay_vung_nen=params["min_ngay_vung_nen"],
        nguong_giam_toi_thieu_pct=params["nguong_giam_toi_thieu_pct"],
        nguong_rsi=params["nguong_rsi"],
        nguong_volume_ratio=params["nguong_volume_ratio"],
    )


def render_historical_recovery_probability_section(storage: Storage) -> None:
    """Hiển thị XÁC SUẤT TẦN SUẤT LỊCH SỬ (empirical) mà 1 mã phục hồi sau
    khi rơi vào tình huống "giảm mạnh + quá bán + volume đột biến + đóng
    cửa yếu" — dựa trên `core.historical_recovery_probability` (Module 6).

    ĐÂY LÀ THỐNG KÊ TẦN SUẤT QUÁ KHỨ, KHÔNG PHẢI DỰ BÁO — luôn hiển thị
    kèm cỡ mẫu và mức độ tin cậy thống kê để người xem tự đánh giá.
    """
    from core.historical_recovery_probability import DIEU_KIEN_MAC_DINH

    st.subheader("📊 Xác suất phục hồi lịch sử")
    st.caption(
        "⚠️ Đây là TẦN SUẤT THỰC NGHIỆM tính từ chính lịch sử giá của mã đó — "
        "KHÔNG phải xác suất dự báo tương lai được đảm bảo. Quá khứ không chắc "
        "lặp lại; cỡ mẫu càng nhỏ thì độ tin cậy càng thấp. Luôn xem kèm cỡ mẫu "
        "và mức độ tin cậy thống kê bên dưới trước khi tham khảo."
    )

    available_symbols = sorted(storage.query_all_keys("ohlcv_history"))
    if not available_symbols:
        st.info(
            "Chưa có dữ liệu lịch sử OHLCV. Chạy `main.py` hoặc "
            "`run_full_market.py` trước để có dữ liệu tính toán."
        )
        return

    # --------------------------------------------------------------------
    # BẢNG TỔNG HỢP SÀNG LỌC (Module 7 — core.base_breakdown_screener):
    # quét TOÀN BỘ watchlist tại THỜI ĐIỂM HIỆN TẠI (khác Module 6 ở trên,
    # vốn quét NGƯỢC quá khứ để tính tần suất) — tìm các mã ĐANG thỏa đồng
    # thời 3 tiêu chí: đứt gãy vùng nền + quá bán + volume đột biến.
    # --------------------------------------------------------------------
    st.markdown("### 📋 Danh sách phục hồi ngắn hạn")
    st.caption(
        "Quét TOÀN BỘ watchlist tại thời điểm hiện tại — khác với phần \"Xác suất "
        "phục hồi lịch sử\" phía dưới (vốn quét ngược quá khứ để tính tần suất). "
        "Đây là bộ lọc kỹ thuật RÚT GỌN (chỉ 3 tiêu chí cốt lõi), KHÔNG phải tín "
        "hiệu mua/bán. Mã lọt qua bộ lọc nên được xem tiếp mục \"Tính cách giao "
        "dịch từng mã\" và \"Xác suất phục hồi lịch sử\" cho riêng mã đó trước khi "
        "cân nhắc bất kỳ quyết định nào."
    )

    with st.expander("⚙️ Tùy chỉnh ngưỡng sàng lọc"):
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            s_giam_pct = st.number_input(
                "Giảm tối thiểu từ pivot (%)", value=15.0, min_value=5.0, max_value=80.0,
                step=1.0, key="screen_giam_pct",
            )
            s_lookback = st.number_input(
                "Số phiên tra ngược tìm vùng nền", value=60, min_value=20, max_value=250,
                step=5, key="screen_lookback",
            )
        with scol2:
            s_rsi = st.number_input(
                "RSI(14) tối đa (quá bán)", value=30.0, min_value=5.0, max_value=50.0,
                step=1.0, key="screen_rsi",
            )
            s_min_ngay = st.number_input(
                "Số phiên tối thiểu để công nhận vùng nền", value=10, min_value=5, max_value=60,
                step=1, key="screen_min_ngay",
            )
        with scol3:
            s_vol_ratio = st.number_input(
                "Tỷ lệ khối lượng tối thiểu (x TB20)", value=1.5, min_value=1.0, max_value=5.0,
                step=0.1, key="screen_vol_ratio",
            )

    if st.button("🔍 Quét watchlist ngay", key="screen_scan_btn"):
        screen_params = {
            "lookback_vung_nen": int(s_lookback),
            "min_ngay_vung_nen": int(s_min_ngay),
            "nguong_giam_toi_thieu_pct": s_giam_pct,
            "nguong_rsi": s_rsi,
            "nguong_volume_ratio": s_vol_ratio,
        }
        try:
            screen_result = _compute_base_breakdown_scan_cached(
                storage, tuple(available_symbols), screen_params
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Lỗi khi quét: {exc}")
            screen_result = pd.DataFrame()

        if screen_result.empty:
            st.session_state["screen_display_df"] = pd.DataFrame()
            st.session_state["screen_match_count"] = 0
        else:
            # Với MỖI mã lọt qua bộ lọc, tính thêm xác suất tăng/giảm sau
            # 3-5-7-10 phiên — dựa trên tần suất các lần TRONG QUÁ KHỨ 3
            # NĂM của CHÍNH mã đó rơi vào tình huống tương tự (tái sử dụng
            # Module 6 — core.historical_recovery_probability — với điều
            # kiện lọc MẶC ĐỊNH của module đó, độc lập với 3 tiêu chí sàng
            # lọc ở Module 7 phía trên).
            cac_moc_phien = (3, 5, 7, 10)
            xac_suat_rows = []
            for ma in screen_result["ma"]:
                df_ma = _load_ohlcv_history_df(storage, ma)
                if df_ma is None or df_ma.empty:
                    xac_suat_rows.append({"ma": ma, "so_lan_quan_sat": None, "do_tin_cay": None})
                    continue
                try:
                    prob_result = _compute_recovery_probability_cached(
                        df_ma, ma, {}, cac_so_phien_du_bao=cac_moc_phien
                    )
                except Exception:  # noqa: BLE001
                    xac_suat_rows.append({"ma": ma, "so_lan_quan_sat": None, "do_tin_cay": None})
                    continue

                row = {
                    "ma": ma,
                    "so_lan_quan_sat": prob_result["so_lan_quan_sat_lich_su"],
                    "do_tin_cay": prob_result["do_tin_cay_thong_ke"],
                }
                for n_phien in cac_moc_phien:
                    r = prob_result["ket_qua_theo_so_phien_du_bao"].get(f"sau_{n_phien}_phien", {})
                    xac_suat_tang = r.get("ty_le_phuc_hoi_pct")
                    row[f"xac_suat_tang_{n_phien}"] = xac_suat_tang
                    row[f"xac_suat_giam_{n_phien}"] = (
                        round(100 - xac_suat_tang, 1) if xac_suat_tang is not None else None
                    )
                xac_suat_rows.append(row)

            xac_suat_df = pd.DataFrame(xac_suat_rows)
            display_df = screen_result.merge(xac_suat_df, on="ma", how="left")

            rename_map = {
                "ma": "Mã", "gia_pivot_ho_tro": "Giá pivot hỗ trợ",
                "so_phien_vung_nen": "Số phiên vùng nền", "gia_hien_tai": "Giá hiện tại",
                "pct_giam_tu_pivot": "% giảm từ pivot", "rsi_hien_tai": "RSI(14)",
                "volume_ratio": "Tỷ lệ KL/TB20",
                "so_lan_quan_sat": "Số lần quan sát (3 năm)", "do_tin_cay": "Độ tin cậy",
            }
            for n_phien in cac_moc_phien:
                rename_map[f"xac_suat_tang_{n_phien}"] = f"Xác suất tăng sau {n_phien} phiên (%)"
                rename_map[f"xac_suat_giam_{n_phien}"] = f"Xác suất giảm sau {n_phien} phiên (%)"

            display_df = display_df.rename(columns=rename_map)

            # LƯU LẠI vào session_state — giữ nguyên kết quả hiển thị xuyên
            # suốt các lần rerun tiếp theo (đổi mã ở phần bên dưới, cuộn
            # trang...), CHỈ mất đi khi bấm "Quét watchlist ngay" lần kế
            # tiếp. Tránh việc bảng kết quả biến mất ngay khi có bất kỳ
            # thao tác nào khác trên trang (hành vi mặc định của Streamlit:
            # st.button() chỉ trả về True ĐÚNG 1 lần tại lượt rerun do
            # chính nó gây ra).
            st.session_state["screen_display_df"] = display_df
            st.session_state["screen_match_count"] = len(screen_result)

    # --- Hiển thị KẾT QUẢ ĐÃ LƯU (nếu có) — độc lập với việc nút bấm có
    #     được nhấn ở lượt rerun hiện tại hay không ---
    if "screen_display_df" in st.session_state:
        saved_df = st.session_state["screen_display_df"]
        if saved_df.empty:
            st.info("Không có mã nào trong watchlist thỏa đồng thời cả 3 tiêu chí tại thời điểm này.")
        else:
            st.success(f"Tìm thấy {st.session_state['screen_match_count']} mã thỏa đồng thời cả 3 tiêu chí:")
            st.caption(
                "⚠️ Cột xác suất tăng/giảm là TẦN SUẤT THỰC NGHIỆM từ lịch sử 3 năm của "
                "chính mã đó — xem kèm \"Số lần quan sát\" và \"Độ tin cậy\": số lần quan "
                "sát càng ít, độ tin cậy càng thấp, không nên dùng làm căn cứ duy nhất."
            )
            dinh_dang_screen = {
                "Giá pivot hỗ trợ": "{:.1f}", "Giá hiện tại": "{:.1f}",
                "% giảm từ pivot": "{:.1f}", "RSI(14)": "{:.1f}", "Tỷ lệ KL/TB20": "{:.1f}",
                "Số phiên vùng nền": "{:.0f}", "Số lần quan sát (3 năm)": "{:.0f}",
            }
            for n_phien in (3, 5, 7, 10):
                dinh_dang_screen[f"Xác suất tăng sau {n_phien} phiên (%)"] = "{:.1f}"
                dinh_dang_screen[f"Xác suất giảm sau {n_phien} phiên (%)"] = "{:.1f}"

            st.dataframe(
                style_tang_giam(saved_df, dinh_dang_so=dinh_dang_screen),
                width='stretch', hide_index=True,
            )
    else:
        st.caption("Bấm nút bên trên để bắt đầu quét lần đầu.")

    st.divider()

    # --------------------------------------------------------------------
    # TÍNH TOÁN CHO 1 MÃ ĐƠN LẺ (Module 6 — xem lại tần suất lịch sử)
    # --------------------------------------------------------------------
    selected_symbol = st.selectbox(
        "Chọn mã để tính xác suất phục hồi", available_symbols,
        key="recovery_prob_symbol",
    )

    with st.expander("⚙️ Tùy chỉnh điều kiện lọc \"tình huống giảm\" (để mặc định nếu không chắc)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            giam_toi_thieu_pct = st.number_input(
                "Giảm tối thiểu (%)", value=float(DIEU_KIEN_MAC_DINH["giam_toi_thieu_pct"]),
                min_value=1.0, max_value=50.0, step=0.5, key="recovery_giam_pct",
            )
            so_phien_toi_da = st.number_input(
                "Trong tối đa (phiên)", value=int(DIEU_KIEN_MAC_DINH["so_phien_toi_da"]),
                min_value=1, max_value=10, step=1, key="recovery_so_phien",
            )
        with col2:
            rsi_toi_da = st.number_input(
                "RSI(14) tối đa (quá bán)", value=float(DIEU_KIEN_MAC_DINH["rsi_toi_da"]),
                min_value=5.0, max_value=50.0, step=1.0, key="recovery_rsi",
            )
            volume_ratio_toi_thieu = st.number_input(
                "Tỷ lệ khối lượng tối thiểu (x TB20)", value=float(DIEU_KIEN_MAC_DINH["volume_ratio_toi_thieu"]),
                min_value=1.0, max_value=5.0, step=0.1, key="recovery_vol_ratio",
            )
        with col3:
            so_phien_giam_lien_tiep = st.number_input(
                "Số phiên giảm liên tiếp tối thiểu", value=int(DIEU_KIEN_MAC_DINH["so_phien_giam_lien_tiep_toi_thieu"]),
                min_value=1, max_value=10, step=1, key="recovery_streak",
            )
            closing_strength_toi_da = st.number_input(
                "Đóng cửa yếu tối đa (closing strength)", value=float(DIEU_KIEN_MAC_DINH["closing_strength_toi_da"]),
                min_value=0.0, max_value=1.0, step=0.05, key="recovery_closing_strength",
            )

    dieu_kien_loc = {
        "giam_toi_thieu_pct": giam_toi_thieu_pct,
        "so_phien_toi_da": int(so_phien_toi_da),
        "rsi_toi_da": rsi_toi_da,
        "volume_ratio_toi_thieu": volume_ratio_toi_thieu,
        "so_phien_giam_lien_tiep_toi_thieu": int(so_phien_giam_lien_tiep),
        "closing_strength_toi_da": closing_strength_toi_da,
    }

    df = _load_ohlcv_history_df(storage, selected_symbol)
    if df is None or df.empty:
        st.warning(f"Không có dữ liệu lịch sử OHLCV cho mã {selected_symbol}.")
        return

    try:
        result = _compute_recovery_probability_cached(df, selected_symbol, dieu_kien_loc)
    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠️ Lỗi khi tính toán: {exc}")
        return

    so_lan = result["so_lan_quan_sat_lich_su"]
    do_tin_cay = result["do_tin_cay_thong_ke"]
    do_tin_cay_emoji = {
        "RAT_THAP": "🔴", "THAP": "🟠", "TRUNG_BINH": "🟡", "KHA_CAO": "🟢",
    }.get(do_tin_cay, "")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Số lần quan sát trong lịch sử", so_lan)
    col_b.metric("Tổng số phiên dữ liệu", result["tong_so_phien_du_lieu_dau_vao"])
    col_c.metric("Độ tin cậy thống kê", f"{do_tin_cay_emoji} {do_tin_cay}")

    st.caption(result["ghi_chu_do_tin_cay"])

    if so_lan == 0:
        st.info("Không tìm thấy tình huống nào khớp điều kiện lọc trong lịch sử dữ liệu hiện có.")
    else:
        rows = []
        for key, label in [
            ("sau_1_phien", "Sau 1 phiên"), ("sau_3_phien", "Sau 3 phiên"), ("sau_5_phien", "Sau 5 phiên"),
        ]:
            r = result["ket_qua_theo_so_phien_du_bao"].get(key, {})
            rows.append({
                "Mốc thời gian": label,
                "Số lần quan sát": r.get("so_lan_quan_sat"),
                "Số lần phục hồi": r.get("so_lan_phuc_hoi"),
                "Tỷ lệ phục hồi (%)": r.get("ty_le_phuc_hoi_pct"),
                "% thay đổi TB": r.get("pct_thay_doi_trung_binh"),
                "% thay đổi trung vị": r.get("pct_thay_doi_trung_vi"),
                "Min (%)": r.get("pct_thay_doi_min"),
                "Max (%)": r.get("pct_thay_doi_max"),
                "Độ lệch chuẩn": r.get("do_lech_chuan"),
            })
        detail_df = pd.DataFrame(rows)
        st.dataframe(
            style_tang_giam(
                detail_df,
                cot_theo_ten=False,
                cot_theo_dau=["% thay đổi TB", "% thay đổi trung vị", "Min (%)", "Max (%)"],
                dinh_dang_so={
                    "Tỷ lệ phục hồi (%)": "{:.1f}", "% thay đổi TB": "{:.1f}",
                    "% thay đổi trung vị": "{:.1f}", "Min (%)": "{:.1f}", "Max (%)": "{:.1f}",
                    "Độ lệch chuẩn": "{:.1f}", "Số lần quan sát": "{:.0f}", "Số lần phục hồi": "{:.0f}",
                },
            ),
            width='stretch', hide_index=True,
        )


# ==============================================================================
# PHẦN 5 — HIỆU SUẤT DANH MỤC MÔ PHỎNG
# ==============================================================================

@st.cache_data(ttl=600, show_spinner="Đang tính thống kê tăng/giảm lịch sử...")
def _tinh_thong_ke_tang_giam_cached(
    df: pd.DataFrame, ma: str, tieu_chi_dat: tuple[str, ...], so_phien_du_bao: int,
    duong_tham_chieu: str = "ema200",
    chuoi_giai_doan: Optional[pd.Series] = None,
    giai_doan_loc: Optional[str] = None,
) -> dict:
    """Bọc `tinh_thong_ke_tang_giam_lich_su()` bằng cache 10 phút."""
    from core.entry_screener import tinh_thong_ke_tang_giam_lich_su

    return tinh_thong_ke_tang_giam_lich_su(
        df, list(tieu_chi_dat), so_phien_du_bao=so_phien_du_bao,
        duong_tham_chieu=duong_tham_chieu,
        chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan_loc,
    )


def render_entry_screener_section(storage: Storage) -> None:
    """Hiển thị báo cáo rà soát danh sách vào lệnh ngắn hạn
    (`core.entry_screener`) — cho phép lọc lại theo tiêu chí mong muốn
    trên kết quả ĐÃ TÍNH SẴN (không cần chạy lại pipeline).

    ĐƯỜNG THAM CHIẾU + CHU KỲ TÙY CHỌN (bổ sung 04/08/2026):
        1. Có thể chọn EMA200 (mặc định, như trước) hoặc MA20 làm đường
           tham chiếu cho xếp hạng ưu tiên + tiêu chí "Giá trên EMA200...".
           Việc này được TÍNH LẠI ngay trong dashboard (không cần chạy
           lại pipeline `main.py`/`run_full_market.py`), dựa trên dữ liệu
           "close"/"ema200"/"ma20" đã có sẵn trong indicator_snapshot.
        2. Chu kỳ đo % thay đổi cho 8 cột thống kê lịch sử giờ chọn được
           1 trong 4 mốc: 5, 10, 15, hoặc 30 phiên (trước đây cố định 30).
    """
    st.subheader("🔍 Rà soát danh sách vào lệnh ngắn hạn")
    st.caption(
        "⚠️ Danh sách CHỜ tham khảo — cần đối chiếu với mục 🚦 Tín hiệu Mua/Bán "
        "phía trên trước khi ra quyết định vào lệnh cụ thể."
    )

    record = storage.get_latest("entry_screener_report", "latest")
    if record is None:
        st.info(
            "Chưa có báo cáo. Chạy `main.py` hoặc `run_full_market.py` để "
            "quét danh sách vào lệnh."
        )
        return

    report = record["data"]
    from core.entry_screener import TIEU_CHI_KHA_DUNG, xep_hang_uu_tien_theo_duong_tham_chieu

    col_ref1, col_ref2 = st.columns(2)
    with col_ref1:
        duong_tham_chieu_nhan = st.radio(
            "Đường tham chiếu (xếp hạng ưu tiên + tiêu chí \"Giá trên...\")",
            ["EMA200", "MA20"], horizontal=True, key="entry_screener_duong_tham_chieu",
        )
    with col_ref2:
        so_phien_du_bao = st.radio(
            "Chu kỳ đo % thay đổi (8 cột thống kê lịch sử)",
            [5, 10, 15, 30], index=3, horizontal=True,
            format_func=lambda x: f"{x} phiên", key="entry_screener_so_phien",
        )
    duong_tham_chieu_key = "ema200" if duong_tham_chieu_nhan == "EMA200" else "ma20"

    # --- Tính lại "Ưu tiên" / "Độ lệch" / tiêu chí "dieu_kien_nen_ema200"
    #     cho TOÀN BỘ mã trong báo cáo, theo đường tham chiếu đã chọn —
    #     dùng dữ liệu "close"/"ema200"/"ma20" đã có sẵn trong
    #     indicator_snapshot, KHÔNG cần chạy lại pipeline. ---
    tat_ca_ma = [m["ma"] for m in report["danh_sach_ma"]]
    snapshot_map = storage.get_latest_many("indicator_snapshot", tat_ca_ma)

    danh_sach_da_tinh_lai = []
    for m in report["danh_sach_ma"]:
        snap = snapshot_map.get(m["ma"])
        m_moi = dict(m)  # không sửa trực tiếp dữ liệu gốc trong report
        if snap:
            data_snap = snap["data"]
            close = data_snap.get("close")
            duong_gia_tri = data_snap.get(duong_tham_chieu_key)
            if close is not None:
                xep_hang = xep_hang_uu_tien_theo_duong_tham_chieu(
                    close, duong_gia_tri, duong_tham_chieu_nhan
                )
                m_moi["xep_hang_uu_tien"] = xep_hang["xep_hang_uu_tien"]
                m_moi["do_lech_ema200_pct"] = xep_hang["do_lech_pct"]  # giữ tên khóa cũ để không phá vỡ chỗ khác

                # Cập nhật lại tiêu chí "dieu_kien_nen_ema200" theo đường MỚI.
                tieu_chi_dat_moi = [t for t in m_moi["tieu_chi_dat"] if t != "dieu_kien_nen_ema200"]
                if xep_hang["xep_hang_uu_tien"] != "KHONG_DAT":
                    tieu_chi_dat_moi.append("dieu_kien_nen_ema200")
                m_moi["tieu_chi_dat"] = tieu_chi_dat_moi
        danh_sach_da_tinh_lai.append(m_moi)

    nhan_tieu_chi = dict(TIEU_CHI_KHA_DUNG)
    nhan_tieu_chi["dieu_kien_nen_ema200"] = f"Giá trên {duong_tham_chieu_nhan} hoặc trong ±10%"

    tieu_chi_chon = st.multiselect(
        "Lọc theo tiêu chí (mặc định: tất cả)",
        options=list(nhan_tieu_chi.keys()),
        default=list(nhan_tieu_chi.keys()),
        format_func=lambda k: nhan_tieu_chi[k],
        key="entry_screener_filter",
    )

    danh_sach_loc = [
        m for m in danh_sach_da_tinh_lai
        if set(m["tieu_chi_dat"]) & set(tieu_chi_chon)
    ]

    # --- Lọc thêm theo khối lượng trung bình 20 phiên (bổ sung 01/08/2026):
    #     loại bỏ các mã thanh khoản quá thấp — dữ liệu lấy từ
    #     indicator_snapshot đã tính sẵn (KHÔNG cần chạy lại pipeline). ---
    nguong_volume_toi_thieu = st.number_input(
        "Khối lượng TB 20 phiên tối thiểu (cổ phiếu/phiên)",
        value=500_000, min_value=0, step=50_000,
        key="entry_screener_min_volume",
        help="Loại bỏ khỏi danh sách các mã có khối lượng giao dịch trung bình "
             "20 phiên THẤP HƠN ngưỡng này — tránh mã thanh khoản quá thấp, khó "
             "vào/ra lệnh với khối lượng lớn.",
    )

    danh_sach_sau_loc_volume = []
    for m in danh_sach_loc:
        snap = snapshot_map.get(m["ma"])
        volume_ma20 = snap["data"].get("volume_ma_20") if snap else None
        if volume_ma20 is not None and volume_ma20 >= nguong_volume_toi_thieu:
            danh_sach_sau_loc_volume.append({**m, "volume_ma_20": volume_ma20})
    so_bi_loai_vi_volume = len(danh_sach_loc) - len(danh_sach_sau_loc_volume)
    danh_sach_loc = danh_sach_sau_loc_volume

    if so_bi_loai_vi_volume > 0:
        st.caption(
            f"🔻 Đã loại {so_bi_loai_vi_volume} mã có khối lượng TB 20 phiên "
            f"dưới {nguong_volume_toi_thieu:,.0f} cổ phiếu/phiên."
        )

    danh_sach_ma_hien_co = [m["ma"] for m in danh_sach_loc]
    ma_da_loc_theo_tim_kiem = render_search_box_if_needed(
        danh_sach_ma_hien_co, key="entry_screener_search",
    )
    danh_sach_loc = [m for m in danh_sach_loc if m["ma"] in ma_da_loc_theo_tim_kiem]

    st.metric("Số mã đạt (theo tiêu chí đã lọc)", len(danh_sach_loc))

    if not danh_sach_loc:
        st.info("Không có mã nào đạt tiêu chí đã chọn.")
        return

    uu_tien_emoji = {"UU_TIEN_CAO": "🟢", "UU_TIEN_TRUNG_BINH": "🟡", "KHONG_DAT": "⚪"}
    ten_cot_bac = [
        "Giảm >15% (%)", "Giảm 10-15% (%)", "Giảm 5-10% (%)", "Giảm 0-5% (%)",
        "Tăng 0-5% (%)", "Tăng 5-10% (%)", "Tăng 10-15% (%)", "Tăng >15% (%)",
    ]
    khoa_bac = [
        "giam_tren_15", "giam_10_15", "giam_5_10", "giam_0_5",
        "tang_0_5", "tang_5_10", "tang_10_15", "tang_tren_15",
    ]

    rows = []
    for m in danh_sach_loc:
        row = {
            "Mã": m["ma"],
            "Ưu tiên": f"{uu_tien_emoji.get(m['xep_hang_uu_tien'], '')} {m['xep_hang_uu_tien']}",
            f"Độ lệch {duong_tham_chieu_nhan}": f"{m['do_lech_ema200_pct']:+.1f}%" if m["do_lech_ema200_pct"] is not None else "—",
            "Volume TB 20 phiên": f"{m['volume_ma_20']:,.0f}" if m.get("volume_ma_20") is not None else "—",
            "Tiêu chí đạt": ", ".join(nhan_tieu_chi.get(t, t) for t in m["tieu_chi_dat"]),
            "Sắp breakout": "🔶 Có" if m["sap_breakout"] else "—",
            "Mẫu hình": m["mau_hinh_kich_hoat"] or "—",
        }

        # --- Thống kê xác suất tăng/giảm theo bậc %, dựa trên tình huống
        #     tương tự trong quá khứ của CHÍNH mã đó (chu kỳ đã chọn ở
        #     trên) — chỉ phát lại được với 2 tiêu chí tính nhanh, xem
        #     docstring `tinh_thong_ke_tang_giam_lich_su` để biết giới hạn. ---
        df_ma = _load_ohlcv_history_df(storage, m["ma"])
        if df_ma is not None and not df_ma.empty:
            try:
                thong_ke = _tinh_thong_ke_tang_giam_cached(
                    df_ma, m["ma"], tuple(m["tieu_chi_dat"]), so_phien_du_bao,
                    duong_tham_chieu_key,
                )
            except Exception:  # noqa: BLE001
                thong_ke = {"so_lan_quan_sat": 0, "phan_bo": {}}
        else:
            thong_ke = {"so_lan_quan_sat": 0, "phan_bo": {}}

        row["Số lần quan sát (lịch sử)"] = thong_ke.get("so_lan_quan_sat", 0)
        phan_bo = thong_ke.get("phan_bo", {})
        for ten_cot, khoa in zip(ten_cot_bac, khoa_bac):
            row[ten_cot] = phan_bo.get(khoa, {}).get("ty_le_pct") if phan_bo else None

        rows.append(row)

    display_df = pd.DataFrame(rows)
    dinh_dang_screen = {c: "{:.1f}" for c in ten_cot_bac}
    st.caption(
        f"📊 8 cột cuối: xác suất tăng/giảm (%) sau {so_phien_du_bao} phiên, dựa trên "
        f"TẦN SUẤT các lần trong quá khứ CHÍNH mã đó từng thỏa 2 tiêu chí tính nhanh "
        f"(Giá so {duong_tham_chieu_nhan}, Tích lũy dài hạn) — nếu mã CHỈ đạt tiêu chí "
        f"Mô hình thu hẹp biên độ / Khối lượng breakout, cột này sẽ trống (không phát "
        f"lại được 2 tiêu chí đó vì quá chậm). Xem cột \"Số lần quan sát\" để đánh giá "
        f"độ tin cậy — quá ít lần quan sát thì không nên dùng làm căn cứ."
    )
    st.dataframe(
        style_tang_giam(display_df, dinh_dang_so=dinh_dang_screen),
        width='stretch', hide_index=True,
    )
    st.caption(report["ghi_chu"])

CHARACTER_LABEL_DISPLAY = {
    # Đổi tên hiển thị (KHÔNG đổi giá trị nội bộ — các module khác như
    # stock_signal_engine.py, capital_allocator.py vẫn so khớp đúng
    # chuỗi gốc "BUNG_NO_NGAN", "LINH_XINH"...) để tránh gây hiểu lầm là
    # khuyến nghị MUA/BÁN hay dự báo xu hướng — đây CHỈ mô tả CÁCH giá đã
    # vận động trong quá khứ, không phải tín hiệu giao dịch.
    "DUT_KHOAT_TANG": "Vận động dứt khoát — chiều tăng (Directional Move – Up)",
    "DUT_KHOAT_GIAM": "Vận động dứt khoát — chiều giảm (Directional Move – Down)",
    "BUNG_NO_NGAN": "Biến động mạnh, ngắn hạn (Short-term Volatility Spike)",
    "LINH_XINH": "Dao động hẹp, thiếu xu hướng (Choppy / Range-bound)",
    "TRUNG_TINH": "Chưa đủ rõ đặc tính (Undetermined)",
}

CHARACTER_LABEL_EMOJI = {
    "DUT_KHOAT_TANG": "🟢", "DUT_KHOAT_GIAM": "🔴",
    "BUNG_NO_NGAN": "🟠", "LINH_XINH": "🟡", "TRUNG_TINH": "⚪",
}


def _dich_canh_bao(canh_bao_list: list[str]) -> str:
    """Đổi tiền tố SQUAT/CHURNING sang dạng song ngữ "Việt (English)" cho
    nhất quán với toàn bộ mục — nội dung câu giải thích phía sau giữ
    nguyên (đã là tiếng Việt đầy đủ)."""
    ket_qua = []
    for c in canh_bao_list:
        c = c.replace("SQUAT —", "Bứt phá giả (Squat) —")
        c = c.replace("CHURNING —", "Nghi ngờ phân phối ẩn (Churning) —")
        ket_qua.append(c)
    return "; ".join(ket_qua) if ket_qua else "—"


@st.cache_data(ttl=600, show_spinner="Đang đếm số lần lịch sử có đặc tính tương tự...")
def _dem_lich_su_nhan_cached(df: pd.DataFrame, ma: str, nhan: str) -> dict:
    """Bọc `dem_lich_su_nhan_tuong_tu()` bằng cache 10 phút — mỗi lần gọi
    phải "phát lại" thuật toán qua tối đa 250 phiên, có chi phí tính toán
    đáng kể, không nên chạy lại mỗi lần trang rerun nếu mã không đổi."""
    from core.stock_character_classifier import dem_lich_su_nhan_tuong_tu

    return dem_lich_su_nhan_tuong_tu(df, nhan, so_phien_kiem_tra=250)


def render_stock_character_section(storage: Storage) -> None:
    """Hiển thị báo cáo tính cách giao dịch (`core.stock_character_classifier`)
    cho toàn bộ mã đã quét.

    TỐI ƯU TỐC ĐỘ (29/07/2026): trước đây gọi `get_latest()` RIÊNG cho
    từng mã (N lượt round-trip Supabase). Giờ dùng `get_latest_many()`
    để gộp lại CHỈ CÒN 1 lượt gọi tổng, bất kể quét bao nhiêu mã.

    CHỌN NHIỀU MÃ + LƯU/XÓA LỰA CHỌN (29/07/2026): trước đây chỉ có ô
    tìm kiếm lọc theo 1 từ khóa — muốn xem vài mã cụ thể phải tìm từng
    mã một. Giờ thêm `st.multiselect` cho phép chọn NHIỀU mã cùng lúc,
    kèm nút lưu lựa chọn (bền vào storage, riêng theo từng người xem
    qua `?user=...` — giống cơ chế watchlist) và nút xóa lựa chọn đã
    lưu để quay lại xem TOÀN BỘ mã như mặc định.

    NGÔN NGỮ SONG NGỮ + THỐNG KÊ LỊCH SỬ (31/07/2026):
        1. Toàn bộ nhãn/cột đổi sang "Tiếng Việt (English)" cho dễ hiểu.
        2. Đổi tên nhãn tính cách (xem CHARACTER_LABEL_DISPLAY) để tránh
           gây hiểu lầm là khuyến nghị giao dịch — đây thuần túy mô tả
           CÁCH giá đã vận động, không phải tín hiệu mua/bán.
        3. Thêm tùy chọn cột "Số lần lịch sử có đặc tính tương tự" — đếm
           trong 250 phiên gần nhất (~1 năm), mã đó từng có CÙNG nhãn bao
           nhiêu lần (tính bằng cách phát lại thuật toán tại từng thời
           điểm trong quá khứ). Mặc định TẮT vì tốn thời gian tính toán
           nếu bật cho nhiều mã cùng lúc — nên chỉ bật sau khi đã thu hẹp
           danh sách bằng ô chọn mã bên dưới.
    """
    st.subheader("🎭 Tính cách giao dịch từng mã (Trading Character)")
    st.caption(
        "⚠️ Đây là mô tả CÁCH GIÁ ĐÃ VẬN ĐỘNG trong quá khứ của mã (dựa trên "
        "percentile so với chính lịch sử của mã đó) — KHÔNG phải khuyến nghị "
        "mua/bán hay dự báo xu hướng tương lai, chỉ dùng để điều chỉnh độ tin "
        "cậy tín hiệu Mua/Bán và phân bổ vốn."
    )

    all_symbol_ids = storage.query_all_keys("stock_character")
    if not all_symbol_ids:
        st.info(
            "Chưa có dữ liệu. Chạy `main.py` hoặc `run_full_market.py` để "
            "tính tính cách giao dịch cho các mã."
        )
        return

    user_id = get_current_user_id()
    saved_record = storage.get_latest("stock_character_selection", user_id)
    saved_selection = saved_record["data"].get("symbols", []) if saved_record else []
    # Chỉ giữ lại các mã đã lưu mà VẪN còn tồn tại dữ liệu (tránh lỗi nếu
    # mã đã bị bỏ khỏi lần quét gần nhất).
    default_selection = [s for s in saved_selection if s in all_symbol_ids]

    selected = st.multiselect(
        "🔎 Chọn các mã muốn xem (để trống = xem TẤT CẢ mã)",
        options=sorted(all_symbol_ids),
        default=default_selection,
        key="stock_character_multiselect",
    )

    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("💾 Lưu lựa chọn này", key="stock_character_save_btn"):
            storage.save("stock_character_selection", user_id, {"symbols": selected})
            st.success(f"Đã lưu {len(selected)} mã đã chọn.")
    with col_clear:
        if st.button("🗑️ Xóa lựa chọn đã lưu (xem lại toàn bộ)", key="stock_character_clear_btn"):
            storage.save("stock_character_selection", user_id, {"symbols": []})
            st.rerun()

    symbol_ids = selected if selected else all_symbol_ids
    if not selected:
        symbol_ids = render_search_box_if_needed(symbol_ids, key="stock_character_search")

    hien_cot_lich_su = st.checkbox(
        "📊 Tính thêm cột \"Số lần lịch sử có đặc tính tương tự\" "
        "(chỉ nên bật khi đã chọn 1 vài mã cụ thể — có thể CHẬM nếu bật cho nhiều mã)",
        key="stock_character_show_history_count",
    )
    if hien_cot_lich_su and len(symbol_ids) > 30:
        st.warning(
            f"Đang hiện {len(symbol_ids)} mã — tính cột lịch sử cho quá nhiều mã "
            "cùng lúc sẽ RẤT CHẬM. Nên thu hẹp lại bằng ô chọn mã ở trên trước."
        )

    character_map = storage.get_latest_many("stock_character", symbol_ids)

    rows = []
    for sym in symbol_ids:
        record = character_map.get(sym)
        if record is None:
            continue
        data = record["data"]
        nhan = data.get("nhan_tinh_cach")
        nhan_hien_thi = CHARACTER_LABEL_DISPLAY.get(nhan, nhan)
        emoji = CHARACTER_LABEL_EMOJI.get(nhan, "")

        row = {
            "Mã": sym,
            "Tính cách (Character)": f"{emoji} {nhan_hien_thi}",
            "Điểm dứt khoát (Character Score)": data.get("character_score"),
            "Điểm lình xình (Choppiness Score)": data.get("choppiness_score"),
            "Cảnh báo (Warning)": _dich_canh_bao(data.get("canh_bao", [])),
            "Độ tin cậy thấp (Low Confidence)": "⚠️ Có" if data.get("do_tin_cay_thap") else "",
        }

        if hien_cot_lich_su and nhan:
            df_ma = _load_ohlcv_history_df(storage, sym)
            if df_ma is not None and not df_ma.empty:
                try:
                    dem = _dem_lich_su_nhan_cached(df_ma, sym, nhan)
                    row["Số lần lịch sử có đặc tính tương tự (trong ~1 năm)"] = (
                        f"{dem['so_lan_khop']}/{dem['so_phien_da_kiem_tra']} phiên"
                    )
                except Exception:  # noqa: BLE001
                    row["Số lần lịch sử có đặc tính tương tự (trong ~1 năm)"] = "—"
            else:
                row["Số lần lịch sử có đặc tính tương tự (trong ~1 năm)"] = "—"

        rows.append(row)

    if not rows:
        st.info("Không có mã nào khớp từ khóa tìm kiếm / lựa chọn.")
        return

    st.dataframe(
        pd.DataFrame(rows), width='stretch', hide_index=True,
        column_config={
            # Cột "Mã" và các cột TỪ "Điểm dứt khoát" trở đi đều đặt
            # width="small" — buộc tiêu đề dài (VD: "Điểm dứt khoát
            # (Character Score)") tự XUỐNG DÒNG bên trong ô tiêu đề thay
            # vì kéo giãn cả cột ra rất rộng theo chiều ngang. Cột "Tính
            # cách (Character)" CHỦ Ý không thu hẹp vì bản thân NỘI DUNG
            # (không chỉ tiêu đề) đã là câu mô tả dài, thu hẹp sẽ làm mất
            # chữ, khó đọc.
            "Mã": st.column_config.TextColumn(width="small"),
            "Điểm dứt khoát (Character Score)": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Điểm lình xình (Choppiness Score)": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Cảnh báo (Warning)": st.column_config.TextColumn(width="small"),
            "Độ tin cậy thấp (Low Confidence)": st.column_config.TextColumn(width="small"),
            "Số lần lịch sử có đặc tính tương tự (trong ~1 năm)": st.column_config.TextColumn(width="small"),
        },
    )


def render_short_term_signal_section(storage: Storage) -> None:
    """Hiển thị báo cáo tiêu chí ngắn hạn (`core.short_term_signal`):
    cảnh báo quá mua VN-Index/cổ phiếu, thống kê xác suất điều chỉnh,
    tín hiệu bắt cá hồi sau giảm mạnh.
    """
    st.subheader("⏱️ Tiêu chí ngắn hạn (Overextension & Bắt cá hồi)")
    st.caption(
        "⚠️ Chỉ tiêu tham khảo thống kê, KHÔNG PHẢI khuyến nghị đầu tư cá "
        "nhân hóa hay tín hiệu giao dịch độc lập."
    )

    record = storage.get_latest("short_term_signal_report", "latest")
    if record is None:
        st.info(
            "Chưa có báo cáo. Cần thêm 'VNINDEX' vào watchlist (mục Watchlist "
            "ở sidebar) rồi chạy `main.py` để tính báo cáo này."
        )
        return

    report = record["data"]
    vnindex = report["vnindex"]

    regime_color = {
        "BINH_THUONG": "🟢", "CANH_BAO_DIEU_CHINH": "🟡", "NGUY_CO_CAO": "🔴",
    }
    col1, col2 = st.columns(2)
    col1.metric(
        "VN-Index — độ lệch MA20",
        f"{vnindex['do_lech_ma20_pct']:+.2f}%",
        f"{regime_color.get(vnindex['muc_canh_bao'], '')} {vnindex['muc_canh_bao']}",
    )
    bounce = report["tin_hieu_bat_ca_hoi"]
    col2.metric(
        "Tín hiệu bắt cá hồi",
        "🟢 KÍCH HOẠT" if bounce["kich_hoat"] else "— Chưa kích hoạt",
        f"Giảm {bounce['muc_giam_tu_dinh_40_phien_pct']:.1f}% từ đỉnh 40 phiên",
    )

    with st.expander("📊 Thống kê xác suất điều chỉnh VN-Index (event study lịch sử)"):
        xac_suat = vnindex["xac_suat_dieu_chinh"]
        st.caption(f"Dựa trên {xac_suat['tong_so_su_kien_lich_su']} sự kiện vượt ngưỡng trong lịch sử.")
        rows = []
        for khung, data in xac_suat["theo_khung_ngay"].items():
            rows.append({
                "Khung (phiên)": khung,
                "Xác suất điều chỉnh": f"{data['xac_suat_pct']:.1f}%" if data["xac_suat_pct"] is not None else "—",
                "Mức điều chỉnh TB": f"{data['muc_dieu_chinh_tb_pct']:.2f}%" if data["muc_dieu_chinh_tb_pct"] is not None else "—",
                "Số phiên TB tới đáy": (
                    f"{data['so_phien_tb_toi_day']:.1f}" if data["so_phien_tb_toi_day"] is not None else "—"
                ),
                "Số sự kiện mẫu": data["so_su_kien_hop_le"],
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.caption(
            "💡 Số lượng sự kiện mẫu càng nhỏ, độ tin cậy của % xác suất càng thấp "
            "— nên xem là tham khảo, không phải quy luật chắc chắn."
        )

    if bounce["kich_hoat"]:
        st.success(
            f"🎯 Tín hiệu bắt cá hồi đang KÍCH HOẠT — ưu tiên: "
            f"{', '.join(bounce['nganh_uu_tien'])} (rổ {bounce['ro_ma_uu_tien']}). "
            f"{bounce['ghi_chu']}"
        )
    elif bounce["phu_quyet_ly_do"]:
        for ly_do in bounce["phu_quyet_ly_do"]:
            st.warning(f"Bắt cá hồi bị phủ quyết: {ly_do}")

    if report["co_phieu_qua_mua"]:
        st.markdown("**Cổ phiếu đang quá mua ngắn hạn (vượt xa MA20):**")
        rows = [
            {"Mã": s["ma"], "Độ lệch MA20": f"{s['do_lech_ma20_pct']:+.1f}%", "Mức cảnh báo": s["muc_canh_bao"]}
            for s in report["co_phieu_qua_mua"]
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    for w in report["canh_bao"]:
        st.warning(w)


@st.cache_data(ttl=600, show_spinner="Đang tính thống kê tăng/giảm theo tín hiệu...")
def _tinh_thong_ke_theo_tin_hieu_cached(df: pd.DataFrame, ma: str, khuyen_nghi: str) -> dict:
    """Bọc `tinh_thong_ke_tang_giam_theo_tin_hieu()` bằng cache 10 phút."""
    from core.stock_signal_engine import tinh_thong_ke_tang_giam_theo_tin_hieu

    return tinh_thong_ke_tang_giam_theo_tin_hieu(df, khuyen_nghi, cac_phien_du_bao=(10, 20, 40, 60))


def _cot_thong_ke_tin_hieu(storage: Storage, ma: str, khuyen_nghi: str) -> dict:
    """Trả về dict các cột %tăng theo 4 mốc phiên (10/20/40/60) cho 1 mã,
    dùng chung cho cả tab MUA và tab BÁN chốt lời."""
    df_ma = _load_ohlcv_history_df(storage, ma)
    cols = {
        "Số lần quan sát (lịch sử)": 0,
        "% tăng sau 10 phiên": None, "% tăng sau 20 phiên": None,
        "% tăng sau 40 phiên": None, "% tăng sau 60 phiên": None,
    }
    if df_ma is None or df_ma.empty:
        return cols
    try:
        ket_qua = _tinh_thong_ke_theo_tin_hieu_cached(df_ma, ma, khuyen_nghi)
    except Exception:  # noqa: BLE001
        return cols
    cols["Số lần quan sát (lịch sử)"] = ket_qua.get("so_lan_quan_sat", 0)
    for sp in (10, 20, 40, 60):
        entry = ket_qua.get("theo_phien", {}).get(sp)
        if entry and entry.get("so_lan", 0) > 0:
            cols[f"% tăng sau {sp} phiên"] = entry["ty_le_tang_pct"]
    return cols


def render_stock_signal_report_section(storage: Storage) -> None:
    """Báo cáo tổng hợp danh sách mã đủ điều kiện khuyến nghị MUA/BÁN
    (`core.stock_signal_engine`) — bước cuối cùng của chuỗi module Điểm
    Vĩ Mô -> Trạng thái Thị trường -> Phân bổ Vốn -> Tín hiệu Mua/Bán.
    """
    st.subheader("🚦 Báo cáo tín hiệu Mua/Bán từng mã")
    st.caption(
        "⚠️ Chỉ tiêu lượng hóa dựa trên quy tắc kỹ thuật cố định — KHÔNG PHẢI "
        "khuyến nghị đầu tư cá nhân hóa hay tín hiệu giao dịch tự động. Lớp cơ "
        "bản (EPS/ROE/D-E/CFO) hiện CHƯA có dữ liệu, chỉ đánh giá dựa trên kỹ thuật."
    )

    record = storage.get_latest("signal_summary_report", "latest")
    if record is None:
        st.info(
            "Chưa có báo cáo tín hiệu. Chạy `main.py` hoặc `run_full_market.py` "
            "để tính tín hiệu mua/bán cho các mã."
        )
        return

    report = record["data"]

    tong_so_ma = (
        len(report["mua"]) + len(report["ban_cat_lo"])
        + len(report["ban_chot_loi"]) + len(report["giu_theo_doi"])
    )
    if tong_so_ma > 5:
        search_text = st.text_input(
            "🔍 Tìm mã (áp dụng cho cả 3 tab bên dưới)",
            key="stock_signal_search", placeholder="Gõ để lọc, vd: HPG",
        )
        if search_text and search_text.strip():
            keyword = search_text.strip().upper()
            report = {
                **report,
                "mua": [e for e in report["mua"] if keyword in e["ma"].upper()],
                "ban_cat_lo": [e for e in report["ban_cat_lo"] if keyword in e["ma"].upper()],
                "ban_chot_loi": [e for e in report["ban_chot_loi"] if keyword in e["ma"].upper()],
                "giu_theo_doi": [e for e in report["giu_theo_doi"] if keyword in e["ma"].upper()],
            }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 MUA", len(report["mua"]))
    col2.metric("🔴 BÁN CẮT LỖ", len(report["ban_cat_lo"]))
    col3.metric("🟠 BÁN CHỐT LỜI", len(report["ban_chot_loi"]))
    col4.metric("🟡 GIỮ/THEO DÕI", len(report["giu_theo_doi"]))

    tab_mua, tab_ban, tab_giu = st.tabs(["🟢 Danh sách MUA", "🔴 Danh sách BÁN", "🟡 Giữ/theo dõi"])

    hien_thong_ke_tin_hieu = st.checkbox(
        "📊 Tính thêm tỷ lệ % tăng sau 10/20/40/60 phiên (dựa trên các lần trong quá khứ "
        "mã đó từng thỏa CÙNG điều kiện KỸ THUẬT MUA/BÁN như hiện tại — có thể CHẬM nếu "
        "danh sách có nhiều mã)",
        key="stock_signal_hien_thong_ke",
    )

    with tab_mua:
        if not report["mua"]:
            st.info("Hiện không có mã nào đủ điều kiện khuyến nghị MUA.")
        else:
            rows = []
            for e in report["mua"]:
                entry_range = e.get("khoang_gia_vao_lenh_de_xuat")
                row = {
                    "Mã": e["ma"],
                    "Stock Score": f"{e['stock_score']:.2f}" if e.get("stock_score") is not None else "—",
                    "Mẫu hình kỹ thuật": e["chi_tiet"].get("mau_hinh_ky_thuat", "—"),
                    "Vùng giá vào lệnh": f"{entry_range[0]:,.2f} - {entry_range[1]:,.2f}" if entry_range else "—",
                }
                if hien_thong_ke_tin_hieu:
                    row.update(_cot_thong_ke_tin_hieu(storage, e["ma"], "MUA"))
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            if hien_thong_ke_tin_hieu:
                st.caption(
                    "📊 4 cột cuối: tỷ lệ % số lần giá TĂNG sau đúng N phiên, dựa trên các "
                    "lần trong quá khứ mã đó từng kích hoạt ĐÚNG điều kiện kỹ thuật MUA như "
                    "hiện tại (KHÔNG xét điều kiện vĩ mô/thị trường — xem giải thích trong "
                    "docstring `tinh_thong_ke_tang_giam_theo_tin_hieu`). Xem cột \"Số lần "
                    "quan sát\" để đánh giá độ tin cậy."
                )
            for e in report["mua"]:
                with st.expander(f"Chi tiết — {e['ma']}"):
                    for r in e["chi_tiet"].get("ky_thuat_dat", []):
                        st.write(f"✅ {r}")
                    for r in e["chi_tiet"].get("co_ban_dat", []):
                        st.write(f"✅ {r}")

    with tab_ban:
        all_sell = report["ban_cat_lo"] + report["ban_chot_loi"]
        if not all_sell:
            st.info("Hiện không có mã nào đủ điều kiện khuyến nghị BÁN.")
        else:
            rows = []
            for e in all_sell:
                loai = "CẮT LỖ" if e.get("loai_ban") == "CAT_LO" else "CHỐT LỜI"
                row = {"Mã": e["ma"], "Loại": loai, "Ưu tiên": e.get("uu_tien") or "—"}
                if hien_thong_ke_tin_hieu:
                    if loai == "CHỐT LỜI":
                        row.update(_cot_thong_ke_tin_hieu(storage, e["ma"], "BAN"))
                    else:
                        # CẮT LỖ phụ thuộc vị thế THẬT (giá vào lệnh/cắt lỗ cụ
                        # thể) — không có mẫu hình kỹ thuật thuần túy để phát
                        # lại lịch sử, nên KHÔNG tính, để trống rõ ràng.
                        row.update({
                            "Số lần quan sát (lịch sử)": "—", "% tăng sau 10 phiên": "—",
                            "% tăng sau 20 phiên": "—", "% tăng sau 40 phiên": "—", "% tăng sau 60 phiên": "—",
                        })
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            if hien_thong_ke_tin_hieu:
                st.caption(
                    "📊 4 cột cuối: tỷ lệ % số lần giá TĂNG sau đúng N phiên, dựa trên các lần "
                    "trong quá khứ mã đó từng kích hoạt ĐÚNG điều kiện kỹ thuật BÁN CHỐT LỜI như "
                    "hiện tại. Mã loại CẮT LỖ để trống (—) vì phụ thuộc vị thế thật, không có "
                    "mẫu hình kỹ thuật thuần túy để phát lại lịch sử."
                )
            for e in all_sell:
                with st.expander(f"Chi tiết — {e['ma']}"):
                    chi_tiet = e.get("chi_tiet", {})
                    for r in chi_tiet.get("ly_do", []):
                        st.write(f"⚠️ {r}")
                    for r in chi_tiet.get("co_ban", []):
                        st.write(f"⚠️ {r}")
                    for r in chi_tiet.get("ky_thuat", []):
                        st.write(f"⚠️ {r}")

    with tab_giu:
        if not report["giu_theo_doi"]:
            st.info("Không có mã nào ở trạng thái giữ/theo dõi.")
        else:
            rows = [{"Mã": e["ma"], "Cảnh báo": "; ".join(e.get("canh_bao", [])) or "—"} for e in report["giu_theo_doi"]]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def _doc_chuoi_giai_doan_da_luu(storage: Storage, key_luu: str) -> Optional[pd.Series]:
    """Đọc chuỗi giai đoạn ĐÃ LƯU SẴN (tính bởi `run_market_regime_history_step()`
    trong main.py, chạy 1 lần/ngày qua run_full_market.py) — RẤT NHANH
    (1 lượt đọc duy nhất), ưu tiên dùng trước khi phải tính lại (live).
    Trả về None nếu chưa có dữ liệu lưu sẵn cho `key_luu` này.
    """
    record = storage.get_latest("chuoi_giai_doan_lich_su", key_luu)
    if record is None:
        return None
    records = record["data"].get("records", [])
    if not records:
        return None
    df_tam = pd.DataFrame(records)
    df_tam["date"] = pd.to_datetime(df_tam["date"])
    return pd.Series(df_tam["giai_doan"].values, index=df_tam["date"])


@st.cache_data(ttl=1800, show_spinner="Đang tính giai đoạn ngành/thị trường theo lịch sử (có thể mất chút thời gian)...")
def _tinh_chuoi_giai_doan_song(_storage: Storage, danh_sach_ma: tuple[str, ...]) -> pd.Series:
    """TÍNH LẠI TRỰC TIẾP (live) chuỗi giai đoạn — chỉ dùng khi CHƯA có
    bản lưu sẵn (VD mới thêm ngành mới, hoặc chưa từng chạy pipeline có
    bước `run_market_regime_history_step`). Cache 30 phút để đỡ tính lại
    nhiều lần trong cùng phiên làm việc.
    """
    from core.market_regime_detector import tinh_chuoi_giai_doan_theo_ngay

    du_lieu_theo_ma: dict[str, pd.DataFrame] = {}
    # 1 LƯỢT TRUY VẤN duy nhất cho TOÀN BỘ danh sách (thay vì lặp từng mã
    # — xem giải thích chi tiết ở bản ghi chú lịch sử bổ sung 05/08/2026).
    ohlcv_map = _storage.get_latest_many("ohlcv_history", list(danh_sach_ma))
    for ma, record in ohlcv_map.items():
        records = record["data"].get("records", [])
        if not records:
            continue
        df_ma = pd.DataFrame(records)
        df_ma["date"] = pd.to_datetime(df_ma["date"])
        df_ma = df_ma.sort_values("date").reset_index(drop=True)
        du_lieu_theo_ma[ma] = df_ma

    return tinh_chuoi_giai_doan_theo_ngay(du_lieu_theo_ma)


def _tinh_chuoi_giai_doan_cached(_storage: Storage, nguon: str, ma_hoac_nganh: str) -> pd.Series:
    """Lấy chuỗi giai đoạn Uptrend/Sideway/Downtrend THEO NGÀY, tổng hợp
    nhiều mã (bổ sung 05/08/2026, tối ưu tốc độ 05/08/2026) — dùng để lọc
    thống kê Kelly/kiểm định theo đúng giai đoạn thị trường chung (toàn
    thị trường) hoặc riêng ngành của 1 mã.

    ƯU TIÊN đọc bản ĐÃ LƯU SẴN trong Supabase (tính 1 lần/ngày qua
    `run_market_regime_history_step()` trong pipeline chính — RẤT NHANH,
    chỉ 1 lượt đọc). CHỈ khi chưa có dữ liệu lưu sẵn (VD ngành mới, hoặc
    chưa chạy pipeline bản mới) mới TÍNH LẠI TRỰC TIẾP (live, chậm hơn).

    `nguon`: "thi_truong" (toàn bộ mã đang có dữ liệu) hoặc "nganh"
    (chỉ các mã CÙNG NGÀNH với `ma_hoac_nganh`, tra qua symbol_sector).
    """
    if nguon == "thi_truong":
        key_luu = "thi_truong"
    else:
        all_keys = _storage.query_all_keys("symbol_sector")
        sector_map = _storage.get_latest_many("symbol_sector", all_keys)
        snap = sector_map.get(ma_hoac_nganh)
        key_luu = snap["data"].get("sector") if snap else None

    if key_luu:
        da_luu = _doc_chuoi_giai_doan_da_luu(_storage, key_luu)
        if da_luu is not None and len(da_luu) > 0:
            return da_luu

    # --- Fallback: chưa có bản lưu sẵn -> tính lại trực tiếp (live) ---
    if nguon == "thi_truong":
        danh_sach_ma = sorted(_storage.query_all_keys("ohlcv_history"))
    else:
        all_keys = _storage.query_all_keys("symbol_sector")
        sector_map = _storage.get_latest_many("symbol_sector", all_keys)
        danh_sach_ma = [
            ma for ma, rec in sector_map.items()
            if key_luu is not None and rec["data"].get("sector") == key_luu
        ]

    return _tinh_chuoi_giai_doan_song(_storage, tuple(danh_sach_ma))


def render_tong_hop_section(storage: Storage) -> None:
    """Module "Tổng hợp" (bổ sung 04/08/2026) — rà soát TOÀN DIỆN 1 mã cổ
    phiếu do người dùng chọn, tích hợp:
        1. Giá hiện tại do NGƯỜI DÙNG TỰ NHẬP (theo giao dịch thực tế —
           không phụ thuộc độ trễ của pipeline tự động).
        2. ATR14 tính MỚI trực tiếp từ dữ liệu OHLCV đã lưu.
        3. Báo cáo Tín hiệu Mua/Bán (core.stock_signal_engine) của mã đó.
        4. Vùng entry/stop-loss/chốt lời tính MỚI (core.capital_allocation_engine),
           dựa trên giá người dùng nhập + ATR14 vừa tính + hỗ trợ/kháng cự.
        5. Công thức phân bổ vốn theo Kelly Criterion (core.entry_screener.
           tinh_kelly_fraction), dựa trên tỷ lệ ăn/thua thống kê được ở
           module "Rà soát danh sách vào lệnh ngắn hạn".

    ĐÂY LÀ CÔNG CỤ TÍNH TOÁN THAM KHẢO — không phải khuyến nghị đầu tư cá
    nhân hóa. Người dùng tự chịu trách nhiệm về quyết định giao dịch.
    """
    st.subheader("📌 Tổng hợp — Rà soát toàn diện 1 mã")
    st.caption(
        "⚠️ Công cụ TÍNH TOÁN THAM KHẢO, tổng hợp các module đã có cho ĐÚNG 1 mã "
        "bạn chọn — KHÔNG phải khuyến nghị đầu tư cá nhân hóa. Mọi công thức (ATR14, "
        "entry/stop-loss, Kelly Criterion) đều minh bạch, bạn tự đối chiếu và ra "
        "quyết định."
    )

    all_symbols = sorted(storage.query_all_keys("ohlcv_history"))
    if not all_symbols:
        st.info("Chưa có dữ liệu OHLCV. Chạy `main.py` hoặc `run_full_market.py` trước.")
        return

    col_chon1, col_chon2 = st.columns(2)
    with col_chon1:
        ma_chon = st.selectbox("Chọn mã cổ phiếu", all_symbols, key="tong_hop_ma_chon")
    with col_chon2:
        chien_luoc = st.selectbox(
            "Chiến lược tính vùng entry", ["breakout", "pullback", "support"],
            format_func=lambda s: {"breakout": "Breakout (phá kháng cự)", "pullback": "Pullback (chờ hồi)", "support": "Hỗ trợ (Sideway)"}[s],
            key="tong_hop_chien_luoc",
        )

    df_ma = _load_ohlcv_history_df(storage, ma_chon)
    if df_ma is None or df_ma.empty:
        st.warning(f"Không có dữ liệu lịch sử OHLCV cho mã {ma_chon}.")
        return

    # --- Thêm nhanh vào "Danh sách theo dõi + Lý do" (bổ sung 04/08/2026) ---
    # Gợi ý sẵn 1 lý do dựa trên tín hiệu Mua/Bán hiện có (nếu có), người
    # dùng có thể sửa lại tự do trước khi lưu.
    _signal_goi_y = storage.get_latest("stock_signal", ma_chon)
    goi_y_ly_do = ""
    if _signal_goi_y:
        _sig_data = _signal_goi_y["data"]
        _ly_do_ky_thuat = _sig_data.get("chi_tiet", {}).get("ky_thuat", []) or _sig_data.get("chi_tiet", {}).get("ky_thuat_dat", [])
        if _ly_do_ky_thuat:
            goi_y_ly_do = f"{_sig_data.get('khuyen_nghi', '')}: " + "; ".join(_ly_do_ky_thuat)

    with st.expander("🔖 Thêm mã này vào Danh sách theo dõi + Lý do", expanded=False):
        ly_do_theo_doi = st.text_area(
            "Lý do theo dõi", value=goi_y_ly_do, key="tong_hop_ly_do_theo_doi",
            placeholder="VD: Phát hiện phân kỳ giảm, cần theo dõi thêm trước khi chốt lời...",
        )
        if st.button("🔖 Thêm vào danh sách theo dõi", key="tong_hop_them_theo_doi_btn"):
            them_vao_danh_sach_theo_doi(storage, ma_chon, ly_do_theo_doi, get_current_user_id())
            st.success(f"Đã thêm {ma_chon} vào Danh sách theo dõi.")

    gia_dong_cua_gan_nhat = float(df_ma["close"].iloc[-1])
    col_gia1, col_gia2 = st.columns(2)
    with col_gia1:
        gia_hien_tai = st.number_input(
            "💰 Giá hiện tại (nhập theo giao dịch thực tế — không phụ thuộc "
            "độ trễ của pipeline tự động)",
            value=gia_dong_cua_gan_nhat, min_value=0.01, step=0.05,
            key="tong_hop_gia_hien_tai",
            help=f"Giá đóng cửa gần nhất theo dữ liệu đã lưu: {gia_dong_cua_gan_nhat:,.2f} "
                 "— sửa lại nếu giá thực tế bạn đang thấy trên bảng giá khác số này.",
        )
    with col_gia2:
        von_giao_dich = st.number_input(
            "💵 Số vốn định giao dịch cho lệnh này (VNĐ)",
            value=100_000_000, min_value=0, step=10_000_000,
            key="tong_hop_von_giao_dich",
        )

    st.divider()

    # === PHẦN 1: ATR14 + Entry/Stop-loss/Take-profit (tính MỚI) ===
    st.markdown("#### 📐 ATR14 và vùng entry/stop-loss/chốt lời (tính mới theo giá vừa nhập)")
    from core.capital_allocation_engine import (
        InvalidCapitalAllocationError, calculate_entry_price_range,
        calculate_stop_loss_range, calculate_take_profit_range,
        calculate_position_size, find_support_resistance,
    )
    from core.market_breadth import calculate_atr

    try:
        atr_series = calculate_atr(df_ma, period=14)
        atr14 = float(atr_series.iloc[-1])
        if pd.isna(atr14) or atr14 <= 0:
            raise ValueError("ATR14 chưa đủ dữ liệu hoặc bằng 0.")

        support, resistance = find_support_resistance(df_ma, lookback=60)
        entry_range = calculate_entry_price_range(gia_hien_tai, atr14, strategy=chien_luoc, support_level=support)
        stop_loss_range = calculate_stop_loss_range(support, atr14)
        take_profit_range = calculate_take_profit_range(resistance, atr14)

        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("ATR14", f"{atr14:,.2f}")
        col_a2.metric("Hỗ trợ (60 phiên)", f"{support:,.2f}")
        col_a3.metric("Kháng cự (60 phiên)", f"{resistance:,.2f}")

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Vùng vào lệnh (Entry)", f"{entry_range[0]:,.2f} — {entry_range[1]:,.2f}")
        col_b2.metric("Vùng cắt lỗ (Stop-loss)", f"{stop_loss_range[0]:,.2f} — {stop_loss_range[1]:,.2f}")
        col_b3.metric("Vùng chốt lời tham khảo", f"{take_profit_range[0]:,.2f} — {take_profit_range[1]:,.2f}")

        # Khối lượng theo rủi ro cố định 2% NAV (mặc định hệ thống) — để đối chiếu.
        qty_theo_rui_ro = calculate_position_size(
            nav=von_giao_dich, risk_per_trade_pct=0.02,
            entry_price_range=entry_range, stop_loss_range=stop_loss_range,
            capital_budget=von_giao_dich,
        )
        st.caption(
            f"📎 Đối chiếu: nếu dùng nguyên tắc rủi ro cố định ≤2% vốn/lệnh (mặc định "
            f"hệ thống), khối lượng tối đa mua được là **{qty_theo_rui_ro:,} cổ phiếu** "
            f"(giá trị ~{qty_theo_rui_ro * entry_range[1]:,.0f} đ)."
        )
    except (InvalidCapitalAllocationError, ValueError, ZeroDivisionError) as exc:
        st.error(f"⚠️ Không tính được vùng entry/stop-loss: {exc}")
        entry_range = stop_loss_range = take_profit_range = None

    st.divider()

    # === PHẦN 2: Báo cáo Tín hiệu Mua/Bán (TRUNG/DÀI HẠN) ===
    st.markdown("#### 🚦 Tín hiệu Mua/Bán (Trung/Dài hạn)")
    st.caption(
        "📅 Khung thời gian: TRUNG/DÀI HẠN (vài tuần đến vài tháng) — dựa trên xu "
        "hướng EMA200, ADX, mẫu hình breakout/pullback (core.stock_signal_engine). "
        "KHÔNG cùng khung thời gian với phần Kelly Criterion bên dưới (vốn tính theo "
        "chu kỳ ngắn hạn bạn chọn) — 2 mục này CÓ THỂ cho kết luận khác nhau, thậm chí "
        "ngược nhau, vì đang trả lời 2 câu hỏi khác nhau (xu hướng dài hạn vs. xác suất "
        "ngắn hạn), không phải mâu thuẫn hay lỗi."
    )
    signal_record = storage.get_latest("stock_signal", ma_chon)
    if signal_record:
        sig = signal_record["data"]
        khuyen_nghi_mau = {"MUA": "🟢 MUA", "BAN": "🔴 BÁN", "GIU_THEO_DOI": "🟡 GIỮ/THEO DÕI"}.get(sig.get("khuyen_nghi"), "—")
        st.metric("Khuyến nghị (Trung/Dài hạn)", khuyen_nghi_mau)
        chi_tiet = sig.get("chi_tiet", {})
        if chi_tiet.get("mau_hinh_ky_thuat"):
            st.write(f"**Mẫu hình kỹ thuật:** {chi_tiet['mau_hinh_ky_thuat']}")
        if chi_tiet.get("ky_thuat_dat"):
            st.write("**Lý do:** " + "; ".join(chi_tiet["ky_thuat_dat"]))
        if sig.get("canh_bao"):
            st.warning("; ".join(sig["canh_bao"]) if isinstance(sig["canh_bao"], list) else sig["canh_bao"])
    else:
        st.info("Chưa có báo cáo tín hiệu Mua/Bán cho mã này. Chạy `main.py`/`run_full_market.py` trước.")

    st.divider()

    # === PHẦN 3: Kelly Criterion — phân bổ vốn tối ưu theo tỷ lệ ăn/thua (NGẮN HẠN) ===
    st.markdown("#### 🎯 Phân bổ vốn tối ưu theo Kelly Criterion (Ngắn hạn)")
    st.caption(
        "📅 Khung thời gian: NGẮN HẠN — xác suất tăng/giảm đo theo đúng SỐ PHIÊN bạn "
        "chọn bên dưới (5/10/15/30 phiên, tối đa ~1,5 tháng), KHÔNG phải cùng khung "
        "trung/dài hạn với mục Tín hiệu Mua/Bán ở trên. Dựa trên tần suất lịch sử "
        "CHÍNH mã này từng ở tình huống tương tự hiện tại (tái sử dụng logic module "
        "\"Rà soát danh sách vào lệnh ngắn hạn\") — công thức Kelly: f* = p/L − q/G "
        "(p=xác suất thắng, q=xác suất thua, G=biên độ tăng TB khi thắng, L=biên độ "
        "giảm TB khi thua)."
    )

    from core.entry_screener import tinh_kelly_fraction

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        duong_tham_chieu_nhan_k = st.radio(
            "Đường tham chiếu", ["EMA200", "MA20"], horizontal=True, key="tong_hop_duong_tham_chieu",
        )
    with col_k2:
        so_phien_du_bao_k = st.radio(
            "Chu kỳ đo (phiên)", [5, 10, 15, 30], index=3, horizontal=True,
            format_func=lambda x: f"{x} phiên", key="tong_hop_so_phien",
        )
    duong_tham_chieu_key_k = "ema200" if duong_tham_chieu_nhan_k == "EMA200" else "ma20"
    st.caption(f"📌 Đang phân tích khung NGẮN HẠN: {so_phien_du_bao_k} phiên tới (~{so_phien_du_bao_k / 20:.1f} tháng giao dịch).")

    # === LỌC THEO GIAI ĐOẠN THỊ TRƯỜNG/NGÀNH (bổ sung 05/08/2026) — áp
    #     dụng chung cho CẢ Kelly VÀ 2 mục kiểm định bên dưới, vì kết quả
    #     Kelly/kiểm định có thể thay đổi rất nhiều tùy giai đoạn thị
    #     trường chung hoặc ngành đang Uptrend/Sideway/Downtrend. ===
    st.markdown("##### 🌐 Lọc theo giai đoạn thị trường/ngành (áp dụng cho Kelly + kiểm định bên dưới)")
    col_gd1, col_gd2 = st.columns(2)
    with col_gd1:
        nguon_giai_doan_nhan = st.radio(
            "Nguồn giai đoạn", ["Không lọc (toàn bộ lịch sử)", "Toàn thị trường (nhiều mã)", "Riêng ngành của mã này"],
            key="tong_hop_nguon_giai_doan",
        )
    giai_doan_loc_nhan = None
    with col_gd2:
        if nguon_giai_doan_nhan != "Không lọc (toàn bộ lịch sử)":
            giai_doan_loc_nhan = st.radio(
                "Giai đoạn muốn lọc", ["Uptrend", "Sideway", "Downtrend"],
                key="tong_hop_giai_doan_loc", horizontal=True,
            )

    chuoi_giai_doan = None
    giai_doan_loc_key = None
    if nguon_giai_doan_nhan != "Không lọc (toàn bộ lịch sử)":
        nguon_key = "thi_truong" if nguon_giai_doan_nhan == "Toàn thị trường (nhiều mã)" else "nganh"
        try:
            chuoi_giai_doan = _tinh_chuoi_giai_doan_cached(storage, nguon_key, ma_chon)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Không tính được chuỗi giai đoạn: {exc}")
            chuoi_giai_doan = None
        giai_doan_loc_key = giai_doan_loc_nhan.lower() if giai_doan_loc_nhan else None

        if chuoi_giai_doan is not None and len(chuoi_giai_doan) > 0:
            phan_bo_giai_doan = chuoi_giai_doan.value_counts()
            st.caption(
                f"📊 Phân bố giai đoạn trong lịch sử đã tính ({'toàn thị trường' if nguon_key == 'thi_truong' else 'ngành của ' + ma_chon}): "
                + ", ".join(f"{nhan}={so_luong}" for nhan, so_luong in phan_bo_giai_doan.items())
            )
        elif chuoi_giai_doan is not None:
            st.info("Chưa tính được chuỗi giai đoạn nào (thiếu dữ liệu các mã liên quan).")

    # --- Xác định CHÍNH XÁC mã này có ĐANG thực sự thỏa từng tiêu chí hay
    #     không, TÍNH LẠI TƯƠI theo đúng đường tham chiếu đang chọn ở trên
    #     — KHÔNG mặc định coi như luôn đạt (lỗi cũ: nếu mã không có trong
    #     báo cáo "Rà soát ngắn hạn" — tức thất bại CẢ 4 tiêu chí — code cũ
    #     vẫn ngầm coi là đạt "Giá trên EMA200/MA20...", gây sai lệch số
    #     liệu Kelly so với chính module "Rà soát danh sách vào lệnh ngắn
    #     hạn"). Sửa 04/08/2026: tính lại đúng như mục đó đang làm. ---
    from core.entry_screener import kiem_tra_tich_luy_dai_han, xep_hang_uu_tien_theo_duong_tham_chieu

    snap = storage.get_latest("indicator_snapshot", ma_chon)
    tieu_chi_dat_k: list[str] = []

    if snap:
        close_k = snap["data"].get("close")
        gia_tri_duong_k = snap["data"].get(duong_tham_chieu_key_k)
        if close_k is not None:
            xep_hang_k = xep_hang_uu_tien_theo_duong_tham_chieu(close_k, gia_tri_duong_k, duong_tham_chieu_nhan_k)
            if xep_hang_k["xep_hang_uu_tien"] != "KHONG_DAT":
                tieu_chi_dat_k.append("dieu_kien_nen_ema200")

    try:
        tich_luy_k = kiem_tra_tich_luy_dai_han(df_ma)
        if tich_luy_k.get("dat"):
            tieu_chi_dat_k.append("tich_luy_dai_han")
    except Exception:  # noqa: BLE001
        pass

    # 2 tiêu chí còn lại ("dao_dong_tat_dan", "volume_breakout") tốn kém để
    # tính lại tươi (cần pattern_detector quét 10-30 tháng) — lấy từ báo
    # cáo "Rà soát ngắn hạn" NẾU mã có mặt trong đó (bất kể mã đó có đạt
    # "dieu_kien_nen_ema200" ở lần quét gốc hay không), còn nếu mã không
    # xuất hiện ở báo cáo đó thì KHÔNG suy diễn — chỉ đơn giản là chưa có
    # thông tin cho 2 tiêu chí này, không mặc định đạt hay không đạt.
    entry_report = storage.get_latest("entry_screener_report", "latest")
    if entry_report:
        for m in entry_report["data"]["danh_sach_ma"]:
            if m["ma"] == ma_chon:
                for t in ("dao_dong_tat_dan", "volume_breakout"):
                    if t in m["tieu_chi_dat"] and t not in tieu_chi_dat_k:
                        tieu_chi_dat_k.append(t)
                break

    if not tieu_chi_dat_k:
        st.info(
            f"Mã {ma_chon} hiện KHÔNG thỏa tiêu chí \"Giá trên {duong_tham_chieu_nhan_k.upper()}...\" "
            "hoặc \"Tích lũy dài hạn\" — nhất quán với việc mã này không xuất hiện trong "
            "mục \"Rà soát danh sách vào lệnh ngắn hạn\" khi bỏ chọn 2 tiêu chí đó."
        )
        thong_ke_k = {"so_lan_quan_sat": 0, "phan_bo": {}}
    else:
        try:
            thong_ke_k = _tinh_thong_ke_tang_giam_cached(
                df_ma, ma_chon, tuple(tieu_chi_dat_k), so_phien_du_bao_k, duong_tham_chieu_key_k,
                chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan_loc_key,
            )
        except Exception as exc:  # noqa: BLE001
            thong_ke_k = {"so_lan_quan_sat": 0, "phan_bo": {}}
            st.error(f"Lỗi khi tính thống kê: {exc}")

    so_lan_k = thong_ke_k.get("so_lan_quan_sat", 0)
    if so_lan_k == 0:
        st.info(thong_ke_k.get("ghi_chu", "Không có đủ dữ liệu lịch sử cho mã này."))
    else:
        kelly = tinh_kelly_fraction(thong_ke_k["phan_bo"])
        st.caption(f"Số lần quan sát trong lịch sử: **{so_lan_k}**")

        if kelly.get("kelly_f") is None:
            st.warning(kelly.get("ghi_chu", "Không tính được Kelly."))
        else:
            col_kq1, col_kq2, col_kq3, col_kq4 = st.columns(4)
            col_kq1.metric("Xác suất thắng", f"{kelly['xac_suat_thang']:.1f}%")
            col_kq2.metric("Xác suất thua", f"{kelly['xac_suat_thua']:.1f}%")
            col_kq3.metric("TB tăng khi thắng", f"+{kelly['trung_binh_tang_pct']:.1f}%")
            col_kq4.metric("TB giảm khi thua", f"-{kelly['trung_binh_giam_pct']:.1f}%")

            st.markdown(
                f"**Kelly đầy đủ (f): {kelly['kelly_f']*100:.1f}% vốn** — "
                f"**Nửa Kelly (khuyến nghị thực tế): {kelly['kelly_f_nua']*100:.1f}% vốn**"
            )
            st.info(kelly["ghi_chu"])

            if kelly["kelly_f"] > 0 and entry_range:
                so_tien_full_kelly = von_giao_dich * kelly["kelly_f"]
                so_tien_nua_kelly = von_giao_dich * kelly["kelly_f_nua"]
                gia_vao_tb = sum(entry_range) / 2
                sl_full = int(so_tien_full_kelly // gia_vao_tb // 100) * 100
                sl_nua = int(so_tien_nua_kelly // gia_vao_tb // 100) * 100

                rows_kelly = [
                    {"Phương án": "Kelly đầy đủ", "% vốn": f"{kelly['kelly_f']*100:.1f}%",
                     "Số tiền": f"{so_tien_full_kelly:,.0f} đ", "Số lượng (làm tròn lô 100)": f"{sl_full:,}"},
                    {"Phương án": "Nửa Kelly (khuyến nghị)", "% vốn": f"{kelly['kelly_f_nua']*100:.1f}%",
                     "Số tiền": f"{so_tien_nua_kelly:,.0f} đ", "Số lượng (làm tròn lô 100)": f"{sl_nua:,}"},
                ]
                st.dataframe(pd.DataFrame(rows_kelly), width='stretch', hide_index=True)

    st.divider()

    # === PHẦN 3B (bổ sung 06/08/2026): Mô phỏng giao dịch KHÔNG CHỒNG LẤN
    #     thời gian — trả lời chính xác "thực tế có bao nhiêu giao dịch
    #     ĐỘC LẬP, lãi/lỗ THẬT là bao nhiêu" — KHÁC với "Số lần quan sát"
    #     ở phần Kelly trên (đếm CẢ các lần chồng lấn thời gian, phù hợp
    #     tính xác suất nhưng KHÔNG phải số giao dịch thực hiện được). ===
    st.markdown("#### 💰 Mô phỏng giao dịch không chồng lấn (số tiền THẬT)")
    st.caption(
        "Khác với \"Số lần quan sát\" ở Kelly phía trên (đếm CẢ các lần chồng lấn thời "
        "gian — VD ngày 1-5, ngày 2-6... đều tính riêng, phù hợp tính xác suất) — mục "
        "này chỉ đếm các giao dịch THỰC SỰ ĐỘC LẬP (mỗi giao dịch phải \"xong\" đủ số "
        "phiên giữ mới tính giao dịch tiếp theo, đúng ràng buộc T+ thật), rồi CỘNG DỒN "
        "lợi nhuận THẬT theo đúng thứ tự lịch sử — cho biết SỐ TIỀN THẬT, không phải "
        "xác suất."
    )
    col_mp1, col_mp2, col_mp3 = st.columns(3)
    with col_mp1:
        von_mo_phong = st.number_input(
            "Vốn ban đầu (VNĐ)", value=1_000_000_000, min_value=0, step=10_000_000,
            key="tong_hop_von_mo_phong",
        )
    with col_mp2:
        so_phien_giu_mo_phong = st.number_input(
            "Số phiên giữ mỗi giao dịch (T+)", value=4, min_value=1, max_value=60, step=1,
            key="tong_hop_so_phien_giu",
            help="Thị trường VN hiện là T+2 — số phiên giữ tối thiểu 1 vòng mua/bán "
                 "thường quy ước khoảng 3-4 phiên tùy cách tính.",
        )
    with col_mp3:
        ty_trong_mo_phong_pct = st.number_input(
            "% vốn dùng mỗi lần", value=50.0, min_value=1.0, max_value=100.0, step=5.0,
            key="tong_hop_ty_trong_mo_phong",
        )

    dung_khoang_tuy_chinh = st.checkbox(
        "🎯 Chỉ vào lệnh khi độ lệch % so với đường tham chiếu nằm ĐÚNG trong 1 khoảng "
        "cụ thể (chặt hơn tiêu chí mặc định \"-10% tới bất kỳ mức dương nào\")",
        key="tong_hop_dung_khoang_tuy_chinh",
    )
    khoang_do_lech_mo_phong = None
    if dung_khoang_tuy_chinh:
        col_kh_mp1, col_kh_mp2 = st.columns(2)
        with col_kh_mp1:
            kh_mp_tu = st.number_input(
                "Từ (%, có thể âm)", value=0.0, min_value=-99.0, max_value=99.0, step=0.5,
                key="tong_hop_kh_mp_tu",
            )
        with col_kh_mp2:
            kh_mp_den = st.number_input(
                "Đến, không gồm (%, có thể âm)", value=5.0, min_value=-99.0, max_value=100.0, step=0.5,
                key="tong_hop_kh_mp_den",
            )
        khoang_do_lech_mo_phong = (kh_mp_tu, kh_mp_den)
        st.caption(
            f"📌 Chỉ mô phỏng vào lệnh khi độ lệch so với "
            f"{'EMA200' if duong_tham_chieu_key_k == 'ema200' else 'MA20'} nằm trong "
            f"[{kh_mp_tu:+.1f}%, {kh_mp_den:+.1f}%) — âm = dưới đường, dương = trên đường."
        )

    if st.button("💰 Chạy mô phỏng", key="tong_hop_chay_mo_phong_btn"):
        from core.entry_screener import mo_phong_giao_dich_khong_chong_lap

        try:
            kq_mp = mo_phong_giao_dich_khong_chong_lap(
                df_ma, tieu_chi_dat_k, so_phien_giu=int(so_phien_giu_mo_phong),
                duong_tham_chieu=duong_tham_chieu_key_k,
                chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan_loc_key,
                von_ban_dau=von_mo_phong, ty_trong_von_moi_lenh=ty_trong_mo_phong_pct / 100,
                khoang_do_lech=khoang_do_lech_mo_phong,
            )
        except Exception as exc:  # noqa: BLE001
            kq_mp = {"so_giao_dich": 0, "ghi_chu": f"Lỗi: {exc}"}

        if kq_mp.get("so_giao_dich", 0) == 0:
            st.warning(kq_mp.get("ghi_chu", "Không mô phỏng được."))
        else:
            col_mpq1, col_mpq2, col_mpq3 = st.columns(3)
            col_mpq1.metric("Số giao dịch ĐỘC LẬP", f"{kq_mp['so_giao_dich']:,}")
            col_mpq2.metric(
                "Thắng/Thua", f"{kq_mp['so_lan_thang']}/{kq_mp['so_lan_thua']}",
                f"{kq_mp['so_lan_thang']/kq_mp['so_giao_dich']*100:.1f}% thắng",
            )
            col_mpq3.metric(
                "Vốn cuối cùng", f"{kq_mp['von_cuoi_cung']:,.0f} đ",
                f"{kq_mp['lai_lo_pct']:+.1f}%",
            )
            st.caption(kq_mp["ghi_chu"])

            with st.expander(f"Xem chi tiết {kq_mp['so_giao_dich']} giao dịch"):
                st.dataframe(pd.DataFrame(kq_mp["chi_tiet_giao_dich"]), width='stretch', hide_index=True)

            st.warning(
                "⚠️ Đây là MÔ PHỎNG trên dữ liệu lịch sử ĐÃ XẢY RA (không phải dự báo "
                "tương lai) — giả định tương lai lặp lại đúng như quá khứ là một giả "
                "định KHÔNG được đảm bảo. Không phải cam kết lợi nhuận."
            )

    st.divider()

    # === PHẦN 3C (bổ sung 06/08/2026): Hiện tại cổ phiếu mô phỏng — đối
    #     chiếu NHANH bối cảnh HIỆN TẠI (giai đoạn ngành + giá/MA20 tự
    #     nhập) với kết quả mô phỏng lịch sử ở trên, để biết mã đang ở
    #     đâu SO VỚI đúng điều kiện vừa mô phỏng. ===
    st.markdown("#### 🔎 Hiện tại cổ phiếu mô phỏng")

    sector_snap_hien_tai = storage.get_latest("symbol_sector", ma_chon)
    nganh_hien_tai = sector_snap_hien_tai["data"].get("sector") if sector_snap_hien_tai else None

    col_ht1, col_ht2 = st.columns(2)
    with col_ht1:
        if nganh_hien_tai:
            regime_record_ht = storage.get_latest("market_regime", nganh_hien_tai)
            regime_ht = regime_record_ht["data"].get("regime") if regime_record_ht else None
            confidence_ht = regime_record_ht["data"].get("confidence", 0.0) if regime_record_ht else 0.0
            regime_emoji_ht = {"uptrend": "🟢", "downtrend": "🔴", "sideway": "🟡"}.get(regime_ht, "⚪")
            st.metric(f"Ngành: {_nhan_nganh(nganh_hien_tai)}", f"{regime_emoji_ht} {regime_ht or 'Chưa xác định'}")
            st.caption(
                f"Độ tin cậy {confidence_ht*100:.0f}% — xem chi tiết đầy đủ tại mục "
                f"\"🌐 Giai đoạn thị trường (định tính)\" (Nhóm 2 — Thị trường chung)."
            )
        else:
            st.info(f"Chưa xác định được ngành của mã {ma_chon}.")

    with col_ht2:
        gia_hien_tai_ht = st.number_input(
            "Giá cổ phiếu hiện tại (tự nhập)", value=0.0, min_value=0.0, step=0.05,
            key="tong_hop_gia_hien_tai_mo_phong",
        )
        ma20_hien_tai_ht = st.number_input(
            "MA20 hiện tại (tự nhập)", value=0.0, min_value=0.0, step=0.05,
            key="tong_hop_ma20_hien_tai_mo_phong",
        )

    if gia_hien_tai_ht > 0 and ma20_hien_tai_ht > 0:
        do_lech_hien_tai = (gia_hien_tai_ht - ma20_hien_tai_ht) / ma20_hien_tai_ht * 100
        st.metric("Độ lệch % hiện tại so với MA20 (tự nhập)", f"{do_lech_hien_tai:+.2f}%")
        if dung_khoang_tuy_chinh and khoang_do_lech_mo_phong:
            khop = khoang_do_lech_mo_phong[0] <= do_lech_hien_tai < khoang_do_lech_mo_phong[1]
            if khop:
                st.success(
                    f"✅ Độ lệch hiện tại ({do_lech_hien_tai:+.2f}%) ĐANG NẰM TRONG khoảng vừa mô phỏng "
                    f"[{khoang_do_lech_mo_phong[0]:+.1f}%, {khoang_do_lech_mo_phong[1]:+.1f}%)."
                )
            else:
                st.warning(
                    f"⚠️ Độ lệch hiện tại ({do_lech_hien_tai:+.2f}%) ĐANG NẰM NGOÀI khoảng vừa mô phỏng "
                    f"[{khoang_do_lech_mo_phong[0]:+.1f}%, {khoang_do_lech_mo_phong[1]:+.1f}%) — kết quả mô "
                    f"phỏng ở trên CHƯA áp dụng cho tình huống hiện tại."
                )

    st.divider()

    # === PHẦN 4 (bổ sung 04/08/2026): Kiểm định thống kê so sánh 2 khoảng
    #     độ lệch — trả lời đúng câu hỏi "khoảng ±5% có khác gì khoảng
    #     5-10% không" bằng kiểm định 2 tỷ lệ (two-proportion z-test),
    #     KHÔNG chỉ so sánh cảm tính 2 con số %. ===
    st.markdown("#### 📐 Kiểm định thống kê: so sánh 2 khoảng độ lệch")
    st.caption(
        "So sánh 2 khoảng ĐỘ LỆCH % CÓ DẤU so với đường tham chiếu đã chọn ở trên "
        "(số ÂM = giá đang DƯỚI đường tham chiếu, số DƯƠNG = giá đang TRÊN) — xem có "
        "khác biệt CÓ Ý NGHĨA THỐNG KÊ về tỷ lệ thắng hay không, dùng kiểm định 2 tỷ lệ "
        "(two-proportion z-test), không phải so sánh cảm tính. VD: nhập -5 đến 0 để lấy "
        "các phiên giá THẤP HƠN đường tham chiếu tối đa 5%."
    )

    col_kh1, col_kh2 = st.columns(2)
    with col_kh1:
        st.markdown("**Khoảng 1**")
        kh1_tu = st.number_input("Từ (%, có thể âm)", value=-5.0, min_value=-99.0, max_value=99.0, step=0.5, key="tong_hop_kh1_tu")
        kh1_den = st.number_input("Đến, không gồm (%, có thể âm)", value=0.0, min_value=-99.0, max_value=100.0, step=0.5, key="tong_hop_kh1_den")
    with col_kh2:
        st.markdown("**Khoảng 2**")
        kh2_tu = st.number_input("Từ (%, có thể âm)", value=0.0, min_value=-99.0, max_value=99.0, step=0.5, key="tong_hop_kh2_tu")
        kh2_den = st.number_input("Đến, không gồm (%, có thể âm)", value=5.0, min_value=-99.0, max_value=100.0, step=0.5, key="tong_hop_kh2_den")

    if st.button("🔬 Chạy kiểm định", key="tong_hop_chay_kiem_dinh"):
        from core.entry_screener import so_sanh_2_khoang_do_lech

        try:
            kq_kd = so_sanh_2_khoang_do_lech(
                df_ma, khoang_1=(kh1_tu, kh1_den), khoang_2=(kh2_tu, kh2_den),
                duong_tham_chieu=duong_tham_chieu_key_k, so_phien_du_bao=so_phien_du_bao_k,
                chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan_loc_key,
            )
        except Exception as exc:  # noqa: BLE001
            kq_kd = {"hop_le": False, "ghi_chu": f"Lỗi: {exc}"}

        if not kq_kd.get("hop_le"):
            st.warning(kq_kd.get("ghi_chu", "Không kiểm định được."))
        else:
            col_kq_a, col_kq_b = st.columns(2)
            with col_kq_a:
                st.metric(
                    f"Khoảng 1 ({kh1_tu:+.0f}% đến {kh1_den:+.0f}%) — Xác suất thắng",
                    f"{kq_kd['xac_suat_thang_khoang_1_pct']:.1f}%",
                    help=f"Số lần quan sát: {kq_kd['so_lan_khoang_1']}",
                )
                st.caption(f"Số lần quan sát: {kq_kd['so_lan_khoang_1']}/{kq_kd.get('tong_so_phien_quet', '?')} phiên giao dịch · TB thay đổi: {kq_kd['pct_thay_doi_trung_binh_khoang_1']:+.2f}%")
            with col_kq_b:
                st.metric(
                    f"Khoảng 2 ({kh2_tu:+.0f}% đến {kh2_den:+.0f}%) — Xác suất thắng",
                    f"{kq_kd['xac_suat_thang_khoang_2_pct']:.1f}%",
                    help=f"Số lần quan sát: {kq_kd['so_lan_khoang_2']}",
                )
                st.caption(f"Số lần quan sát: {kq_kd['so_lan_khoang_2']}/{kq_kd.get('tong_so_phien_quet', '?')} phiên giao dịch · TB thay đổi: {kq_kd['pct_thay_doi_trung_binh_khoang_2']:+.2f}%")

            st.metric("P-value (kiểm định 2 tỷ lệ)", f"{kq_kd['p_value']:.4f}")
            if kq_kd["co_y_nghia_thong_ke"]:
                st.success(kq_kd["ghi_chu"])
            else:
                st.info(kq_kd["ghi_chu"])
            st.caption(
                "⚠️ P-value nhỏ chỉ cho biết khác biệt QUAN SÁT ĐƯỢC khó xảy ra do ngẫu "
                "nhiên thuần túy trong dữ liệu lịch sử — KHÔNG chứng minh quan hệ nhân quả "
                "hay đảm bảo lặp lại trong tương lai."
            )

    st.divider()

    # === PHẦN 5 (bổ sung 04/08/2026): "Kiểm định có RSI (14)" — tương tự
    #     Phần 4 ở trên nhưng phân nhóm theo GIÁ TRỊ RSI(14) thay vì % độ
    #     lệch so với đường tham chiếu. ===
    st.markdown("#### 📊 Kiểm định có RSI (14)")
    st.caption(
        "Tương tự kiểm định ở trên, nhưng phân nhóm theo GIÁ TRỊ RSI(14) tại từng "
        "thời điểm thay vì % độ lệch so với EMA200/MA20 — VD so sánh vùng quá bán "
        "(RSI < 30) với vùng quá mua (RSI > 69) xem xác suất thắng có khác biệt CÓ "
        "Ý NGHĨA THỐNG KÊ hay không. 3 vùng RSI kinh điển: Quá bán (0-30), Trung "
        "tính (31-69), Quá mua (70-100) — có thể chọn bất kỳ 2 trong số này hoặc "
        "khoảng tùy ý."
    )

    col_rsi1, col_rsi2 = st.columns(2)
    with col_rsi1:
        st.markdown("**Khoảng 1 (RSI)**")
        rsi1_tu = st.number_input("Từ", value=0.0, min_value=0.0, max_value=100.0, step=1.0, key="tong_hop_rsi1_tu")
        rsi1_den = st.number_input("Đến, không gồm", value=30.0, min_value=0.1, max_value=101.0, step=1.0, key="tong_hop_rsi1_den")
    with col_rsi2:
        st.markdown("**Khoảng 2 (RSI)**")
        rsi2_tu = st.number_input("Từ", value=70.0, min_value=0.0, max_value=100.0, step=1.0, key="tong_hop_rsi2_tu")
        rsi2_den = st.number_input("Đến, không gồm", value=101.0, min_value=0.1, max_value=101.0, step=1.0, key="tong_hop_rsi2_den")

    if st.button("🔬 Chạy kiểm định RSI", key="tong_hop_chay_kiem_dinh_rsi"):
        from core.entry_screener import so_sanh_2_khoang_rsi

        try:
            kq_rsi = so_sanh_2_khoang_rsi(
                df_ma, khoang_1=(rsi1_tu, rsi1_den), khoang_2=(rsi2_tu, rsi2_den),
                so_phien_du_bao=so_phien_du_bao_k,
                chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan_loc_key,
            )
        except Exception as exc:  # noqa: BLE001
            kq_rsi = {"hop_le": False, "ghi_chu": f"Lỗi: {exc}"}

        if not kq_rsi.get("hop_le"):
            st.warning(kq_rsi.get("ghi_chu", "Không kiểm định được."))
        else:
            col_rq_a, col_rq_b = st.columns(2)
            with col_rq_a:
                st.metric(
                    f"RSI {rsi1_tu:.0f} — {rsi1_den:.0f} — Xác suất thắng",
                    f"{kq_rsi['xac_suat_thang_khoang_1_pct']:.1f}%",
                    help=f"Số lần quan sát: {kq_rsi['so_lan_khoang_1']}",
                )
                st.caption(f"Số lần quan sát: {kq_rsi['so_lan_khoang_1']}/{kq_rsi.get('tong_so_phien_quet', '?')} phiên giao dịch · TB thay đổi: {kq_rsi['pct_thay_doi_trung_binh_khoang_1']:+.2f}%")
            with col_rq_b:
                st.metric(
                    f"RSI {rsi2_tu:.0f} — {rsi2_den:.0f} — Xác suất thắng",
                    f"{kq_rsi['xac_suat_thang_khoang_2_pct']:.1f}%",
                    help=f"Số lần quan sát: {kq_rsi['so_lan_khoang_2']}",
                )
                st.caption(f"Số lần quan sát: {kq_rsi['so_lan_khoang_2']}/{kq_rsi.get('tong_so_phien_quet', '?')} phiên giao dịch · TB thay đổi: {kq_rsi['pct_thay_doi_trung_binh_khoang_2']:+.2f}%")

            st.metric("P-value (kiểm định 2 tỷ lệ)", f"{kq_rsi['p_value']:.4f}")
            if kq_rsi["co_y_nghia_thong_ke"]:
                st.success(kq_rsi["ghi_chu"])
            else:
                st.info(kq_rsi["ghi_chu"])
            st.caption(
                "⚠️ P-value nhỏ chỉ cho biết khác biệt QUAN SÁT ĐƯỢC khó xảy ra do ngẫu "
                "nhiên thuần túy trong dữ liệu lịch sử — KHÔNG chứng minh quan hệ nhân quả "
                "hay đảm bảo lặp lại trong tương lai."
            )

    st.warning(
        "⚠️ Kelly Criterion là CÔNG THỨC TOÁN HỌC áp dụng lên TẦN SUẤT LỊCH SỬ, "
        "giả định tương lai lặp lại quá khứ — một giả định KHÔNG được đảm bảo. Kelly "
        "đầy đủ trên thực tế biến động RẤT MẠNH; nên cân nhắc dùng nửa Kelly hoặc thấp "
        "hơn, và KHÔNG bao giờ vượt quá nguyên tắc quản trị rủi ro chung của danh mục "
        "(rủi ro mỗi lệnh ≤2% NAV, toàn danh mục ≤20% NAV)."
    )

def render_portfolio_section(storage: Storage, portfolio_key: str = "default") -> None:
    st.subheader("💼 Hiệu suất danh mục mô phỏng")

    history = storage.get_history("portfolio_snapshot", portfolio_key, limit=500)
    if not history:
        st.info("Chưa có dữ liệu danh mục mô phỏng nào được lưu.")
        return

    # history đang sắp xếp mới nhất trước -> đảo lại để vẽ biểu đồ theo thời gian tăng dần
    history_sorted = list(reversed(history))

    equity_df = pd.DataFrame({
        "Thời gian": [h["timestamp"] for h in history_sorted],
        "NAV": [h["data"].get("nav") for h in history_sorted],
    }).set_index("Thời gian")
    st.line_chart(equity_df)

    latest = history[0]["data"]
    col1, col2, col3 = st.columns(3)
    col1.metric("NAV hiện tại", f"{latest.get('nav', 0):,.0f} VND")
    col2.metric("Tổng lợi nhuận", f"{latest.get('total_return_pct', 0):.2f}%")
    col3.metric("Tỷ trọng cổ phiếu thực tế", f"{latest.get('total_stock_weight_pct', 0):.1f}%")

    positions = latest.get("positions", [])
    if positions:
        st.write("**Vị thế hiện tại:**")
        st.dataframe(pd.DataFrame(positions), width='stretch', hide_index=True)


# ==============================================================================
# QUẢN LÝ WATCHLIST (lưu bền qua storage — không cần gõ tay lại mỗi lần)
# ==============================================================================

DEFAULT_WATCHLIST = ["HPG", "VNM", "FPT"]


def get_current_user_id() -> str:
    """Lấy định danh người dùng hiện tại từ tham số URL `?user=...` — mỗi
    người dùng 1 LINK RIÊNG để có watchlist RIÊNG (danh sách mã quan tâm
    khác nhau), trong khi TOÀN BỘ dữ liệu thị trường chung (giá, chỉ báo
    kỹ thuật, tín hiệu mua/bán, giai đoạn thị trường...) vẫn dùng CHUNG
    cho mọi người (đọc từ cùng 1 Supabase).

    Không có `?user=` trong link -> dùng watchlist mặc định "default".
    """
    return st.query_params.get("user", "default")


def load_watchlist(storage: Storage, user_id: str = "default") -> list[str]:
    record = storage.get_latest("watchlist", user_id)
    if record is None:
        return list(DEFAULT_WATCHLIST)
    return record["data"].get("symbols", list(DEFAULT_WATCHLIST))


def save_watchlist(storage: Storage, symbols: list[str], user_id: str = "default") -> None:
    storage.save("watchlist", user_id, {"symbols": symbols})


# ==============================================================================
# DANH SÁCH THEO DÕI + LÝ DO (bổ sung 04/08/2026) — KHÁC với "Watchlist" ở
# trên (chỉ là danh sách mã đơn thuần): mục này lưu THÊM lý do cụ thể vì
# sao cần theo dõi mã đó, phát sinh trong lúc rà soát/phân tích — có thể
# thêm nhanh trực tiếp từ module "📌 Tổng hợp".
# ==============================================================================

def load_danh_sach_theo_doi(storage: Storage, user_id: str = "default") -> list[dict]:
    record = storage.get_latest("danh_sach_theo_doi", user_id)
    if record is None:
        return []
    return record["data"].get("danh_sach", [])


def save_danh_sach_theo_doi(storage: Storage, danh_sach: list[dict], user_id: str = "default") -> None:
    storage.save("danh_sach_theo_doi", user_id, {"danh_sach": danh_sach})


def them_vao_danh_sach_theo_doi(storage: Storage, ma: str, ly_do: str, user_id: str = "default") -> None:
    """Thêm 1 mã vào danh sách theo dõi kèm lý do — nếu mã ĐÃ CÓ SẴN thì
    CẬP NHẬT lại lý do mới (không tạo bản ghi trùng lặp cho cùng 1 mã)."""
    from datetime import date

    danh_sach = load_danh_sach_theo_doi(storage, user_id)
    danh_sach = [d for d in danh_sach if d["ma"] != ma]
    danh_sach.append({"ma": ma, "ly_do": ly_do, "ngay_them": date.today().isoformat()})
    danh_sach.sort(key=lambda d: d["ma"])
    save_danh_sach_theo_doi(storage, danh_sach, user_id)


@st.cache_data(ttl=30, show_spinner="Đang tải dữ liệu watchlist...")
def build_watchlist_detail_table(_storage: Storage, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Tổng hợp thông tin CƠ BẢN + KỸ THUẬT đã tính sẵn ở CÁC MODULE KHÁC
    (không tính toán lại) cho từng mã trong watchlist, thành 1 bảng duy
    nhất — dùng lại: `indicator_snapshot` (giá/MA/EMA/volume),
    `symbol_sector` + `market_regime_quant` (ngành + giai đoạn thị
    trường), `stock_signal` (khuyến nghị mua/bán), `pattern_result` (mô
    hình thu hẹp biên độ), `entry_screener_report` (xếp hạng ưu tiên).

    Cột số (giá/MA/EMA/khối lượng) giữ kiểu số THẬT (float, không format
    thành chuỗi) để `st.dataframe`/`st.data_editor` tự căn phải + định
    dạng đẹp qua `column_config` — tránh lỗi chữ/số bị ngắt dòng giữa từ
    khi hiển thị (đã gặp thực tế khi dùng chuỗi trong cột hẹp).

    TỐI ƯU TỐC ĐỘ (29/07/2026): trước đây hàm này gọi `get_latest()`
    RIÊNG cho từng mã × từng loại dữ liệu (5 loại/mã) — với watchlist
    N mã là 5×N lượt round-trip mạng tới Supabase, rất chậm (N=5 mã ->
    25 lượt gọi nối tiếp nhau). Giờ dùng `get_latest_many()` để gộp lại:
    CHỈ còn ~5-6 lượt gọi TỔNG CỘNG, bất kể watchlist có bao nhiêu mã.
    """
    symbols = list(symbols)

    entry_screener_record = _storage.get_latest("entry_screener_report", "latest")
    entry_screener_lookup = {}
    if entry_screener_record:
        for m in entry_screener_record["data"].get("danh_sach_ma", []):
            entry_screener_lookup[m["ma"]] = m

    # --- Bước 1: lấy gộp snapshot/sector/signal/pattern cho TOÀN BỘ mã,
    #     mỗi loại dữ liệu chỉ 1 lượt gọi (thay vì N lượt) ---
    snapshot_map = _storage.get_latest_many("indicator_snapshot", symbols)
    sector_map = _storage.get_latest_many("symbol_sector", symbols)
    signal_map = _storage.get_latest_many("stock_signal", symbols)
    pattern_map = _storage.get_latest_many("pattern_result", symbols)

    # --- Bước 2: giai đoạn thị trường theo NGÀNH — cần biết ngành của
    #     từng mã trước (từ sector_map ở trên), rồi gộp 1 lượt gọi cho
    #     TẤT CẢ các ngành khác nhau (thường ít hơn nhiều so với số mã) ---
    distinct_sectors = {
        sector_map[sym]["data"].get("sector")
        for sym in symbols
        if sym in sector_map and sector_map[sym]["data"].get("sector")
    }
    regime_map = _storage.get_latest_many("market_regime_quant", list(distinct_sectors))

    rows = []
    for sym in symbols:
        try:
            snapshot = snapshot_map[sym]["data"] if sym in snapshot_map else {}
            sector = sector_map[sym]["data"].get("sector") if sym in sector_map else None
            regime_data = regime_map[sector]["data"] if sector in regime_map else {}
            signal_data = signal_map[sym]["data"] if sym in signal_map else {}
            pattern_data = pattern_map[sym]["data"] if sym in pattern_map else {}
            screener_entry = entry_screener_lookup.get(sym, {})

            khuyen_nghi = signal_data.get("khuyen_nghi")
            khuyen_nghi_emoji = {"MUA": "🟢 MUA", "BAN": "🔴 BÁN", "GIU_THEO_DOI": "🟡 GIỮ"}.get(
                khuyen_nghi, "—"
            )
            tren_ema200 = snapshot.get("price_above_ema200")

            rows.append({
                "Mã": sym,
                "Ngành": sector or "—",
                "Giá": snapshot.get("close"),
                "MA20": snapshot.get("ma20"),
                "EMA50": snapshot.get("ema50"),
                "EMA200": snapshot.get("ema200"),
                "Trên EMA200": "✅" if tren_ema200 is True else "❌" if tren_ema200 is False else "—",
                "Khối lượng": snapshot.get("volume"),
                "Đột biến KL": "🔶" if snapshot.get("is_volume_breakout") else "",
                "Giai đoạn ngành": regime_data.get("trang_thai") or "—",
                "Tin cậy": regime_data.get("do_tin_cay") or "—",
                "Tín hiệu": khuyen_nghi_emoji,
                "Mô hình (%)": (
                    round(pattern_data["confidence"] * 100)
                    if pattern_data.get("confidence") is not None else None
                ),
                "Ưu tiên": (screener_entry.get("xep_hang_uu_tien") or "—").replace("UU_TIEN_", "").replace("_", " "),
            })
        except Exception as exc:  # noqa: BLE001
            # KHÔNG để 1 mã lỗi làm sập toàn bộ bảng — hiện dòng báo lỗi
            # riêng cho mã đó, các mã khác vẫn hiển thị bình thường.
            rows.append({
                "Mã": sym, "Ngành": f"⚠️ Lỗi đọc dữ liệu: {exc}",
                "Giá": None, "MA20": None, "EMA50": None, "EMA200": None,
                "Trên EMA200": "—", "Khối lượng": None, "Đột biến KL": "",
                "Giai đoạn ngành": "—", "Tin cậy": "—",
                "Tín hiệu": "—", "Mô hình (%)": None,
                "Ưu tiên": "—",
            })

    return pd.DataFrame(rows)


def remove_symbols_from_watchlist(watchlist: list[str], symbols_to_remove: list[str]) -> list[str]:
    """Loại bỏ các mã trong `symbols_to_remove` khỏi `watchlist` (bỏ qua
    những mã không có trong danh sách). Trả về danh sách MỚI, không sửa
    `watchlist` gốc.
    """
    to_remove = set(symbols_to_remove)
    return [s for s in watchlist if s not in to_remove]


def render_watchlist_manager_section(storage: Storage) -> None:
    """Quản lý watchlist (thêm/xóa mã), lưu bền vào storage — hiển thị
    như MỘT MỤC trong danh sách điều hướng (không còn cố định ở sidebar).

    Watchlist RIÊNG theo từng người xem, dựa trên tham số URL `?user=...`
    — cho phép nhiều người CÙNG xem 1 dashboard (cùng dữ liệu thị trường
    dùng chung) nhưng MỖI NGƯỜI theo dõi danh sách mã KHÁC NHAU.
    """
    user_id = get_current_user_id()
    watchlist = load_watchlist(storage, user_id)

    if user_id == "default":
        st.info(
            "💡 Đang xem watchlist **mặc định** (chung). Để có watchlist RIÊNG "
            "không ảnh hưởng người khác, thêm `?user=ten_cua_ban` vào cuối link "
            "trình duyệt — ví dụ: `...?user=tuyen` — rồi lưu link đó lại để dùng "
            "riêng về sau."
        )
    else:
        st.success(f"👤 Đang xem watchlist riêng của: **{user_id}**")

    # Bọc trong st.form để bấm Enter trong ô nhập cũng submit được, không
    # bắt buộc phải bấm chuột vào nút "Thêm vào watchlist" (hành vi mặc
    # định của Streamlit: text_input đứng riêng lẻ KHÔNG submit khi Enter,
    # chỉ submit khi nằm trong form).
    with st.form("add_symbol_form", clear_on_submit=True):
        new_symbol = st.text_input(
            "Thêm mã mới", key="new_symbol_input", placeholder="Ví dụ: SSI"
        )
        submitted = st.form_submit_button("➕ Thêm vào watchlist")

    if submitted:
        symbol_clean = new_symbol.strip().upper()
        if symbol_clean and symbol_clean not in watchlist:
            watchlist.append(symbol_clean)
            save_watchlist(storage, watchlist, user_id)
            st.rerun()
        elif symbol_clean in watchlist:
            st.warning(f"'{symbol_clean}' đã có trong watchlist.")

    if not watchlist:
        st.info("Watchlist đang trống.")
        return

    st.caption(f"Danh sách hiện tại ({len(watchlist)} mã):")
    displayed_watchlist = render_search_box_if_needed(watchlist, key="watchlist_search")
    if not displayed_watchlist:
        st.info("Không có mã nào khớp từ khóa tìm kiếm.")

    # --- Bảng chi tiết tổng hợp (cơ bản + kỹ thuật đã tính từ module khác)
    #     — dùng st.data_editor (bảng dữ liệu THẬT, Streamlit tự căn cột,
    #     KHÔNG ngắt chữ/số giữa dòng như cách vẽ thủ công st.columns cũ)
    #     + cột tick "🗑️ Xóa" tích hợp ngay trong bảng, xử lý xóa hàng
    #     loạt qua 1 nút bấm bên dưới. ---
    st.markdown("### 📊 Thông tin chi tiết")
    detail_df = build_watchlist_detail_table(storage, tuple(displayed_watchlist))
    detail_df.insert(len(detail_df.columns), "🗑️ Xóa", False)

    edited_df = st.data_editor(
        detail_df,
        hide_index=True,
        width='stretch',
        disabled=[c for c in detail_df.columns if c != "🗑️ Xóa"],
        column_config={
            "Giá": st.column_config.NumberColumn(format="%.2f"),
            "MA20": st.column_config.NumberColumn(format="%.0f"),
            "EMA50": st.column_config.NumberColumn(format="%.0f"),
            "EMA200": st.column_config.NumberColumn(format="%.0f"),
            "Khối lượng": st.column_config.NumberColumn(format="%.0f"),
            "Mô hình (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "🗑️ Xóa": st.column_config.CheckboxColumn(help="Tick để xóa mã khỏi watchlist"),
        },
        key="watchlist_detail_editor",
    )

    ma_can_xoa = edited_df.loc[edited_df["🗑️ Xóa"], "Mã"].tolist()
    if ma_can_xoa:
        if st.button(f"🗑️ Xóa {len(ma_can_xoa)} mã đã tick", key="confirm_delete_ticked_btn"):
            watchlist = remove_symbols_from_watchlist(watchlist, ma_can_xoa)
            save_watchlist(storage, watchlist, user_id)
            st.rerun()

    st.caption(
        "💡 Dữ liệu lấy từ các module đã tính sẵn (chỉ báo kỹ thuật, giai đoạn "
        "thị trường theo ngành, tín hiệu mua/bán, mô hình tích lũy, xếp hạng "
        "ưu tiên vào lệnh) — chạy `main.py`/`run_full_market.py` để cập nhật."
    )


# ==============================================================================
# NHẬT KÝ GIAO DỊCH MUA/BÁN (mô phỏng)
# ==============================================================================

def render_danh_sach_theo_doi_section(storage: Storage) -> None:
    """Hiển thị + quản lý "Danh sách theo dõi + Lý do" (bổ sung 04/08/2026)
    — KHÁC với "Watchlist" (chỉ là danh sách mã đơn thuần): mục này lưu
    thêm LÝ DO cụ thể vì sao cần theo dõi từng mã, phát sinh trong quá
    trình rà soát/phân tích. Có thể thêm nhanh trực tiếp từ module
    "📌 Tổng hợp", hoặc thêm/xóa thủ công ngay tại đây.
    """
    st.subheader("🔖 Danh sách theo dõi + Lý do")
    st.caption(
        "Khác với \"📋 Watchlist\" (chỉ là danh sách mã đơn thuần) — mục này lưu "
        "kèm LÝ DO cụ thể vì sao cần theo dõi từng mã, phát sinh trong lúc rà soát/"
        "phân tích. Có thể thêm nhanh ngay từ module \"📌 Tổng hợp\", hoặc thêm/sửa/"
        "xóa thủ công tại đây."
    )

    user_id = get_current_user_id()
    danh_sach = load_danh_sach_theo_doi(storage, user_id)

    with st.form("them_theo_doi_thu_cong_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            ma_moi = st.text_input("Mã cổ phiếu", key="theo_doi_ma_moi").strip().upper()
        with col2:
            ly_do_moi = st.text_input("Lý do theo dõi", key="theo_doi_ly_do_moi")
        submitted = st.form_submit_button("➕ Thêm vào danh sách")
        if submitted:
            if not ma_moi:
                st.warning("Cần nhập mã cổ phiếu.")
            else:
                them_vao_danh_sach_theo_doi(storage, ma_moi, ly_do_moi, user_id)
                st.success(f"Đã thêm {ma_moi} vào Danh sách theo dõi.")
                st.rerun()

    if not danh_sach:
        st.info("Danh sách theo dõi hiện đang trống.")
        return

    df_hien_thi = pd.DataFrame(danh_sach).rename(
        columns={"ma": "Mã", "ly_do": "Lý do theo dõi", "ngay_them": "Ngày thêm"}
    )
    df_hien_thi.insert(len(df_hien_thi.columns), "🗑️ Xóa", False)

    edited = st.data_editor(
        df_hien_thi, hide_index=True, width='stretch', key="theo_doi_editor",
        column_config={"🗑️ Xóa": st.column_config.CheckboxColumn(help="Tick để xóa mã khỏi danh sách theo dõi")},
    )
    if st.button("Xác nhận xóa các mã đã tick", key="theo_doi_xoa_btn"):
        so_luong_truoc = len(danh_sach)
        ma_can_xoa = set(edited.loc[edited["🗑️ Xóa"], "Mã"])
        danh_sach_moi = [d for d in danh_sach if d["ma"] not in ma_can_xoa]
        save_danh_sach_theo_doi(storage, danh_sach_moi, user_id)
        st.success(f"Đã xóa {so_luong_truoc - len(danh_sach_moi)} mã khỏi danh sách theo dõi.")
        st.rerun()


def render_trade_journal_section(storage: Storage, symbols: list[str]) -> None:
    st.subheader("📒 Nhật ký giao dịch mua/bán (mô phỏng)")
    st.caption(
        "⚠️ Chỉ ghi nhận giao dịch MÔ PHỎNG do bạn tự nhập — KHÔNG đặt lệnh "
        "giao dịch thật dưới bất kỳ hình thức nào."
    )

    from datetime import date as date_cls

    from core.trade_journal import (
        VALID_REFERENCE_INDICATORS,
        close_trade_entry,
        create_trade_entry,
        summarize_trades,
    )

    ref_options = sorted(VALID_REFERENCE_INDICATORS)

    # --- Form ghi nhận MUA mới ---
    with st.expander("➕ Ghi nhận lệnh MUA mới"):
        with st.form("buy_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                buy_symbol = st.selectbox("Mã", symbols or ["HPG"], key="buy_symbol")
                buy_qty = st.number_input(
                    "Khối lượng", min_value=1, value=100, step=100, key="buy_qty"
                )
                buy_price = st.number_input(
                    "Giá mua", min_value=0.0, step=0.01, format="%.2f", key="buy_price"
                )
            with col2:
                buy_date_input = st.date_input(
                    "Ngày mua", value=date_cls.today(), key="buy_date"
                )
                buy_ref = st.selectbox(
                    "Điểm vào lệnh tại đường", ref_options, key="buy_ref"
                )
                buy_reason = st.text_area("Lý do mua", key="buy_reason")

            submitted = st.form_submit_button("Ghi nhận mua")
            if submitted:
                if buy_price <= 0:
                    st.error("Giá mua phải > 0.")
                else:
                    entry = create_trade_entry(
                        symbol=buy_symbol, qty=int(buy_qty), buy_price=float(buy_price),
                        buy_date=buy_date_input, buy_reason=buy_reason,
                        buy_reference_indicator=buy_ref,
                    )
                    storage.save("trade_journal", entry["trade_id"], entry)
                    st.success(f"Đã ghi nhận lệnh mua {buy_symbol}.")
                    st.rerun()

    # --- Đọc toàn bộ giao dịch đã lưu ---
    trade_ids = storage.query_all_keys("trade_journal")
    all_trades: list[dict] = []
    for trade_id in trade_ids:
        record = storage.get_latest("trade_journal", trade_id)
        if record is not None:
            all_trades.append(record["data"])

    open_trades = [t for t in all_trades if not t.get("is_closed")]

    # --- Form đóng vị thế (ghi nhận bán) ---
    with st.expander("📤 Đóng vị thế (ghi nhận bán)"):
        if not open_trades:
            st.info("Không có vị thế nào đang mở để đóng.")
        else:
            options = {
                f"{t['symbol']} · mua {t['buy_date']} @ {t['buy_price']:.2f} (KL {t['qty']})": t["trade_id"]
                for t in open_trades
            }
            selected_label = st.selectbox(
                "Chọn vị thế cần đóng", list(options.keys()), key="close_select"
            )
            selected_trade_id = options[selected_label]
            selected_entry = next(t for t in open_trades if t["trade_id"] == selected_trade_id)

            with st.form("sell_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    sell_price = st.number_input(
                        "Giá bán", min_value=0.0, step=0.01, format="%.2f", key="sell_price"
                    )
                    sell_date_input = st.date_input(
                        "Ngày bán", value=date_cls.today(), key="sell_date"
                    )
                with col2:
                    sell_ref = st.selectbox(
                        "Điểm ra lệnh tại đường", ref_options, key="sell_ref"
                    )
                    sell_reason = st.text_area("Lý do bán", key="sell_reason")

                submitted_sell = st.form_submit_button("Ghi nhận bán")
                if submitted_sell:
                    if sell_price <= 0:
                        st.error("Giá bán phải > 0.")
                    else:
                        updated = close_trade_entry(
                            selected_entry, sell_price=float(sell_price),
                            sell_date=sell_date_input, sell_reason=sell_reason,
                            sell_reference_indicator=sell_ref,
                        )
                        storage.save("trade_journal", selected_trade_id, updated)
                        st.success(f"Đã đóng vị thế {selected_entry['symbol']}.")
                        st.rerun()

    # --- Bảng tổng hợp toàn bộ giao dịch ---
    if not all_trades:
        st.info("Chưa có giao dịch nào được ghi nhận.")
        return

    rows = []
    for t in sorted(all_trades, key=lambda x: x["buy_date"], reverse=True):
        rows.append({
            "Mã": t["symbol"],
            "KL": t["qty"],
            "Giá mua": _fmt_price(t["buy_price"]),
            "Ngày mua": t["buy_date"],
            "Điểm vào (MA/EMA)": t["buy_reference_indicator"],
            "Lý do mua": t.get("buy_reason", "") or "—",
            "Giá bán": _fmt_price(t["sell_price"]) if t.get("sell_price") else "—",
            "Ngày bán": t.get("sell_date") or "—",
            "Điểm ra (MA/EMA)": t.get("sell_reference_indicator") or "—",
            "Lý do bán": t.get("sell_reason", "") or "—",
            "PnL": _fmt_price(t["pnl"]) if t.get("pnl") is not None else "—",
            "PnL %": f"{t['pnl_pct']:.2f}%" if t.get("pnl_pct") is not None else "—",
            "Trạng thái": "Đã đóng" if t.get("is_closed") else "Đang mở",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    summary = summarize_trades(all_trades)
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng lãi/lỗ đã thực hiện", f"{summary['total_pnl']:,.2f}")
    col2.metric("Tỷ lệ thắng", f"{summary['win_rate_pct']:.1f}%")
    col3.metric("Số lệnh đang mở", summary["n_open"])


# ==============================================================================
# MAIN — lắp ráp toàn bộ dashboard
# ==============================================================================

# Danh sách nhãn các mục (đúng thứ tự hiển thị) — dùng cho menu điều
# hướng nhanh ở sidebar. Định nghĩa hàm/tham số thực tế nằm trong main()
# vì cần `storage`/`symbols` đã tính tại thời điểm chạy.
# Tổ chức các mục theo ĐÚNG 3 nhóm trong báo cáo kỹ thuật (Vĩ mô -> Thị
# trường chung -> Giao dịch cổ phiếu) — tiêu đề nhóm viết HOA, mỗi nhóm có
# thể MỞ RỘNG/THU HẸP độc lập ở chế độ "Xem tất cả" (dùng st.expander),
# và ở chế độ "Chỉ xem 1 mục" thì chọn NHÓM trước rồi mới chọn mục trong
# nhóm đó — giúp danh sách lựa chọn gọn hơn thay vì liệt kê phẳng cả 17 mục.
DASHBOARD_GROUPS = [
    ("NHÓM 1 — VĨ MÔ (THẾ GIỚI + VIỆT NAM)", [
        "🌍 Nhập dữ liệu vĩ mô thủ công",
        "📉 Rà soát mô hình co hẹp (XAUUSD/BTC)",
    ]),
    ("NHÓM 2 — THỊ TRƯỜNG CHUNG", [
        "🌐 Giai đoạn thị trường (định tính)",
        "📋 Báo cáo tổng hợp thị trường chung",
        "📐 Giai đoạn thị trường (3 lớp định lượng)",
        "📐 HĐTL VN30 — Entry/Vốn/R:R",
    ]),
    ("NHÓM 3 — GIAO DỊCH CỔ PHIẾU", [
        "📌 Tổng hợp",
        "🔖 Danh sách theo dõi + Lý do",
        "📋 Watchlist (thêm/xóa mã)",
        "📈 Bảng giá theo dõi (Watchlist)",
        "🕯️ Biểu đồ nến",
        "📒 Nhật ký giao dịch mua/bán",
        "📦 Khuyến nghị phân bổ vốn (đơn giản)",
        "📦 Khuyến nghị phân bổ vốn (ATR14 chi tiết)",
        "🔎 Mã có mô hình thu hẹp biên độ",
        "🚦 Báo cáo tín hiệu Mua/Bán",
        "🎭 Tính cách giao dịch từng mã",
        "📊 Xác suất phục hồi lịch sử",
        "🔍 Rà soát danh sách vào lệnh ngắn hạn",
        "⏱️ Tiêu chí ngắn hạn",
        "💼 Danh mục mô phỏng",
    ]),
]


def require_login() -> None:
    """Chặn xem TOÀN BỘ nội dung dashboard cho tới khi đăng nhập bằng
    Google — dùng `st.login()`/`st.user` có sẵn của Streamlit (yêu cầu
    streamlit>=1.42 + gói `Authlib`, đã khai báo trong requirements.txt).

    CẦN CẤU HÌNH TRƯỚC KHI DÙNG (xem hướng dẫn đầy đủ trong CLAUDE.md):
      1. Tạo OAuth Client ID trên Google Cloud Console.
      2. Thêm mục [auth] vào Streamlit Cloud -> Settings -> Secrets
         (redirect_uri, cookie_secret, client_id, client_secret,
         server_metadata_url).
      3. Thêm `ALLOWED_EMAILS` vào Secrets — danh sách email được phép
         xem dashboard, cách nhau bởi dấu phẩy (VD: "a@gmail.com,
         b@gmail.com"). Để trống/không khai báo -> CHO PHÉP MỌI email
         Google đăng nhập được (không khuyến khích khi public link).

    LƯU Ý: đây là lớp bảo vệ Ở MỨC DASHBOARD — không mã hóa/ẩn dữ liệu
    trong Supabase, ai có connection string thật vẫn truy cập được DB
    trực tiếp. Đủ dùng cho mục đích ngăn người lạ tình cờ vào xem/sửa
    qua link công khai, KHÔNG thay thế cho bảo mật hạ tầng đầy đủ.
    """
    if not st.user.is_logged_in:
        st.title("📊 pm_ck — Đăng nhập")
        st.write(
            "Đây là dashboard riêng tư. Vui lòng đăng nhập bằng tài khoản "
            "Google đã được cấp quyền để tiếp tục."
        )
        st.button("🔐 Đăng nhập bằng Google", on_click=st.login, type="primary")
        st.stop()

    allowed_emails_raw = st.secrets.get("ALLOWED_EMAILS", "")
    allowed_emails = {
        e.strip().lower() for e in allowed_emails_raw.split(",") if e.strip()
    }

    if allowed_emails and (st.user.email or "").lower() not in allowed_emails:
        st.title("📊 pm_ck")
        st.error(
            f"🚫 Tài khoản **{st.user.email}** chưa được cấp quyền xem "
            f"dashboard này. Liên hệ quản trị viên nếu bạn cần truy cập."
        )
        st.button("Đăng xuất", on_click=st.logout)
        st.stop()

    with st.sidebar:
        st.caption(f"👤 Đã đăng nhập: {st.user.email}")
        st.button("🔓 Đăng xuất", on_click=st.logout, key="logout_btn")


def main() -> None:
    st.set_page_config(
        page_title="pm_ck — Dashboard theo dõi CK Việt Nam",
        layout="wide",
    )

    require_login()

    st.title("📊 pm_ck — Theo dõi & mô phỏng giao dịch chứng khoán Việt Nam")
    st.caption(
        "⚠️ Đây là công cụ THEO DÕI VÀ MÔ PHỎNG — không đặt lệnh giao dịch "
        "thật dưới bất kỳ hình thức nào. Mọi khuyến nghị chỉ mang tính "
        "tham khảo trên danh mục mô phỏng."
    )

    storage = load_storage()
    symbols = load_watchlist(storage, get_current_user_id())  # tải âm thầm theo đúng người xem — quản lý UI giờ nằm trong mục "📋 Watchlist" bên dưới

    with st.sidebar:
        st.header("Cấu hình hiển thị")

        st.divider()
        st.header("📑 Chuyển nhanh tới mục")
        mode = st.radio(
            "Chế độ xem", ["Xem tất cả (cuộn trang)", "Chỉ xem 1 mục"],
            key="dashboard_view_mode", label_visibility="collapsed",
        )
        selected_section = None
        if mode == "Chỉ xem 1 mục":
            selected_group_title = st.radio(
                "Chọn nhóm", [group_title for group_title, _ in DASHBOARD_GROUPS],
                key="dashboard_selected_group",
            )
            labels_trong_nhom = next(
                labels for group_title, labels in DASHBOARD_GROUPS
                if group_title == selected_group_title
            )
            selected_section = st.radio(
                "Chọn mục trong nhóm", labels_trong_nhom,
                key=f"dashboard_selected_section__{selected_group_title}",
            )

    # --- Danh sách mục theo đúng thứ tự hiển thị, kèm hàm + tham số riêng ---
    sections_to_call = [
        ("📌 Tổng hợp", render_tong_hop_section, (storage,)),
        ("🔖 Danh sách theo dõi + Lý do", render_danh_sach_theo_doi_section, (storage,)),
        ("📋 Watchlist (thêm/xóa mã)", render_watchlist_manager_section, (storage,)),
        ("📈 Bảng giá theo dõi (Watchlist)", render_watchlist_section, (storage, symbols)),
        ("🕯️ Biểu đồ nến", render_chart_section, (storage, symbols)),
        ("📒 Nhật ký giao dịch mua/bán", render_trade_journal_section, (storage, symbols)),
        ("🌐 Giai đoạn thị trường (định tính)", render_market_regime_section, (storage,)),
        ("🌍 Nhập dữ liệu vĩ mô thủ công", render_manual_macro_data_section, (storage,)),
        ("📉 Rà soát mô hình co hẹp (XAUUSD/BTC)", render_vcp_scan_section, (storage,)),
        ("📋 Báo cáo tổng hợp thị trường chung", render_market_summary_report_section, (storage,)),
        ("📐 Giai đoạn thị trường (3 lớp định lượng)", render_market_regime_quant_section, (storage,)),
        ("📐 HĐTL VN30 — Entry/Vốn/R:R", render_hdtl_vn30_section, (storage,)),
        ("📦 Khuyến nghị phân bổ vốn (đơn giản)", render_allocation_section, (storage, symbols)),
        ("📦 Khuyến nghị phân bổ vốn (ATR14 chi tiết)", render_capital_allocation_v2_section, (storage, symbols)),
        ("🔎 Mã có mô hình thu hẹp biên độ", render_pattern_section, (storage,)),
        ("🚦 Báo cáo tín hiệu Mua/Bán", render_stock_signal_report_section, (storage,)),
        ("🎭 Tính cách giao dịch từng mã", render_stock_character_section, (storage,)),
        ("📊 Xác suất phục hồi lịch sử", render_historical_recovery_probability_section, (storage,)),
        ("🔍 Rà soát danh sách vào lệnh ngắn hạn", render_entry_screener_section, (storage,)),
        ("⏱️ Tiêu chí ngắn hạn", render_short_term_signal_section, (storage,)),
        ("💼 Danh mục mô phỏng", render_portfolio_section, (storage,)),
    ]

    # --- Tra cứu nhanh render_fn + args theo label, dùng chung cho cả 2 chế độ ---
    sections_map = {label: (render_fn, args) for label, render_fn, args in sections_to_call}

    if selected_section is not None:
        # --- Chế độ CHỈ XEM 1 MỤC — không cuộn, chỉ hiện đúng mục đã chọn ---
        # LƯU Ý: KHÔNG thêm tiêu đề markdown ở đây — mỗi hàm render_* bên
        # dưới đã tự gọi st.subheader() với đúng tên mục rồi, thêm tiêu đề
        # ở ngoài nữa sẽ làm tiêu đề hiện lặp lại 2 lần trên trang.
        render_fn, args = sections_map[selected_section]
        render_fn(*args)
    else:
        # --- Chế độ xem TẤT CẢ — chia theo 3 NHÓM (khớp báo cáo kỹ thuật),
        #     mỗi nhóm bọc trong st.expander để người xem tự MỞ RỘNG hoặc
        #     THU HẸP cho gọn, không bắt buộc phải cuộn qua toàn bộ 17 mục
        #     cùng lúc. Tiêu đề nhóm viết HOA. ---
        for group_title, labels in DASHBOARD_GROUPS:
            with st.expander(f"🔽 {group_title}", expanded=True):
                for i, label in enumerate(labels):
                    if i > 0:
                        st.divider()
                    render_fn, args = sections_map[label]
                    render_fn(*args)


main()
