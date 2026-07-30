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

def render_market_regime_section(storage: Storage, sectors: list[str]) -> None:
    st.subheader("📊 Giai đoạn thị trường hiện tại")

    if not sectors:
        st.info("Chưa có dữ liệu giai đoạn thị trường cho ngành nào.")
        return

    regime_emoji = {"uptrend": "🟢", "downtrend": "🔴", "sideway": "🟡"}
    all_affected_sectors: set[str] = set()

    for sector in sectors:
        record = storage.get_latest("market_regime", sector)
        if record is None:
            continue

        data = record["data"]
        regime = data.get("regime")
        confidence = data.get("confidence", 0.0)
        emoji = regime_emoji.get(regime, "⚪")

        with st.expander(f"{emoji} {sector}: {regime or 'chưa xác định'} "
                          f"(độ tin cậy {confidence * 100:.0f}%)"):
            for reason in data.get("reasoning", []):
                st.write(f"- {reason}")

        all_affected_sectors.update(data.get("affected_sectors", []))

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

    tab_series, tab_cpi_us, tab_target, tab_event = st.tabs([
        "📈 Chuỗi số liệu (Fed/FX/CPI VN/Liên NH)",
        "🇺🇸 CPI Mỹ",
        "🎯 Mục tiêu CPI VN",
        "⚠️ Sự kiện địa chính trị",
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

    # --- TAB 4: Sự kiện địa chính trị (chọn 1 trong 5 mức, không phải chuỗi) ---
    with tab_event:
        st.caption(
            "Đây là điểm DUY NHẤT có thể ghi đè (override) toàn bộ Macro Score "
            "về mức rất âm bất kể các chỉ số khác — cập nhật ngay khi có tin tức "
            "quan trọng, không chờ dữ liệu kinh tế phản ánh (luôn trễ hơn thị trường)."
        )
        current_event_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
        current_event_key = (
            current_event_record["data"]["event_key"] if current_event_record else "none"
        )
        event_keys = list(EVENT_OPTIONS.keys())
        current_index = event_keys.index(current_event_key) if current_event_key in event_keys else 0

        selected_event = st.selectbox(
            "Mức độ sự kiện hiện tại", event_keys,
            index=current_index, format_func=lambda k: EVENT_OPTIONS[k], key="event_select",
        )
        event_note = st.text_area("Ghi chú (tùy chọn)", key="event_note")
        if st.button("Cập nhật trạng thái sự kiện", key="update_event_btn"):
            storage.save("manual_macro_setting", "geopolitical_event", {
                "event_key": selected_event, "note": event_note,
                "updated_date": date_cls.today().isoformat(),
            })
            st.success(f"Đã cập nhật: {EVENT_OPTIONS[selected_event]}.")
            st.rerun()

        if current_event_record:
            st.info(
                f"Trạng thái hiện tại: **{EVENT_OPTIONS[current_event_key]}** "
                f"(cập nhật lần cuối: {current_event_record['data'].get('updated_date', '—')})"
            )

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

    candidate_symbols = symbols or storage.query_all_keys("pattern_result")
    rows = []
    for symbol in candidate_symbols:
        record = storage.get_latest("pattern_result", symbol)
        if record is None:
            continue
        data = record["data"]
        rows.append({
            "Mã": symbol,
            "Độ tin cậy (%)": round(data.get("confidence", 0.0) * 100, 1),
            "Giá đỉnh tích lũy (breakout ref.)": _fmt_price(data.get("accumulation_high")),
            "Số tháng hình thành": data.get("effective_scan_months"),
        })

    if not rows:
        st.info("Chưa phát hiện mã nào có mô hình thu hẹp biên độ.")
        return

    df = pd.DataFrame(rows).sort_values("Độ tin cậy (%)", ascending=False)
    st.dataframe(df, width='stretch', hide_index=True)


@st.cache_data(ttl=600, show_spinner="Đang quét lịch sử tìm các tình huống tương tự...")
def _compute_recovery_probability_cached(df: pd.DataFrame, ma: str, dieu_kien_loc: dict) -> dict:
    """Bọc `tinh_xac_suat_phuc_hoi_lich_su()` bằng cache 10 phút — việc
    quét toàn bộ lịch sử (750-1250 phiên) có chi phí tính toán đáng kể,
    không nên chạy lại mỗi lần trang rerun nếu mã/điều kiện lọc không đổi.
    """
    from core.historical_recovery_probability import tinh_xac_suat_phuc_hoi_lich_su

    return tinh_xac_suat_phuc_hoi_lich_su(ma, df, dieu_kien_loc=dieu_kien_loc)


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
    st.markdown("### 🔎 Bảng tổng hợp: Đứt gãy vùng nền + Quá bán + Volume đột biến")
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
            st.info("Không có mã nào trong watchlist thỏa đồng thời cả 3 tiêu chí tại thời điểm này.")
        else:
            st.success(f"Tìm thấy {len(screen_result)} mã thỏa đồng thời cả 3 tiêu chí:")
            display_df = screen_result.rename(columns={
                "ma": "Mã", "gia_pivot_ho_tro": "Giá pivot hỗ trợ",
                "so_phien_vung_nen": "Số phiên vùng nền", "gia_hien_tai": "Giá hiện tại",
                "pct_giam_tu_pivot": "% giảm từ pivot", "rsi_hien_tai": "RSI(14)",
                "volume_ratio": "Tỷ lệ KL/TB20",
            })
            st.dataframe(display_df, width='stretch', hide_index=True)

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
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.warning(result["canh_bao_phap_ly"])


# ==============================================================================
# PHẦN 5 — HIỆU SUẤT DANH MỤC MÔ PHỎNG
# ==============================================================================

def render_entry_screener_section(storage: Storage) -> None:
    """Hiển thị báo cáo rà soát danh sách vào lệnh ngắn hạn
    (`core.entry_screener`) — cho phép lọc lại theo tiêu chí mong muốn
    trên kết quả ĐÃ TÍNH SẴN (không cần chạy lại pipeline).
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
    from core.entry_screener import TIEU_CHI_KHA_DUNG

    tieu_chi_chon = st.multiselect(
        "Lọc theo tiêu chí (mặc định: tất cả)",
        options=list(TIEU_CHI_KHA_DUNG.keys()),
        default=list(TIEU_CHI_KHA_DUNG.keys()),
        format_func=lambda k: TIEU_CHI_KHA_DUNG[k],
        key="entry_screener_filter",
    )

    danh_sach_loc = [
        m for m in report["danh_sach_ma"]
        if set(m["tieu_chi_dat"]) & set(tieu_chi_chon)
    ]

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
    rows = []
    for m in danh_sach_loc:
        rows.append({
            "Mã": m["ma"],
            "Ưu tiên": f"{uu_tien_emoji.get(m['xep_hang_uu_tien'], '')} {m['xep_hang_uu_tien']}",
            "Độ lệch EMA200": f"{m['do_lech_ema200_pct']:+.1f}%" if m["do_lech_ema200_pct"] is not None else "—",
            "Tiêu chí đạt": ", ".join(TIEU_CHI_KHA_DUNG.get(t, t) for t in m["tieu_chi_dat"]),
            "Sắp breakout": "🔶 Có" if m["sap_breakout"] else "—",
            "Mẫu hình": m["mau_hinh_kich_hoat"] or "—",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption(report["ghi_chu"])


def render_stock_character_section(storage: Storage) -> None:
    """Hiển thị báo cáo tính cách giao dịch (`core.stock_character_classifier`)
    cho toàn bộ mã đã quét — dứt khoát tăng/giảm, bùng nổ ngắn, lình xình,
    trung tính, kèm cảnh báo SQUAT/CHURNING nếu có.

    TỐI ƯU TỐC ĐỘ (29/07/2026): trước đây gọi `get_latest()` RIÊNG cho
    từng mã (N lượt round-trip Supabase). Giờ dùng `get_latest_many()`
    để gộp lại CHỈ CÒN 1 lượt gọi tổng, bất kể quét bao nhiêu mã.

    CHỌN NHIỀU MÃ + LƯU/XÓA LỰA CHỌN (29/07/2026): trước đây chỉ có ô
    tìm kiếm lọc theo 1 từ khóa — muốn xem vài mã cụ thể phải tìm từng
    mã một. Giờ thêm `st.multiselect` cho phép chọn NHIỀU mã cùng lúc,
    kèm nút lưu lựa chọn (bền vào storage, riêng theo từng người xem
    qua `?user=...` — giống cơ chế watchlist) và nút xóa lựa chọn đã
    lưu để quay lại xem TOÀN BỘ mã như mặc định.
    """
    st.subheader("🎭 Tính cách giao dịch từng mã")
    st.caption(
        "⚠️ Đây là đặc tính VẬN ĐỘNG nội tại của mã (dựa trên percentile so "
        "với chính lịch sử của mã đó) — KHÔNG phải khuyến nghị đầu tư, chỉ "
        "dùng để điều chỉnh độ tin cậy tín hiệu Mua/Bán và phân bổ vốn."
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

    nhan_emoji = {
        "DUT_KHOAT_TANG": "🟢", "DUT_KHOAT_GIAM": "🔴",
        "BUNG_NO_NGAN": "🟠", "LINH_XINH": "🟡", "TRUNG_TINH": "⚪",
    }

    character_map = storage.get_latest_many("stock_character", symbol_ids)

    rows = []
    for sym in symbol_ids:
        record = character_map.get(sym)
        if record is None:
            continue
        data = record["data"]
        nhan = data.get("nhan_tinh_cach")
        rows.append({
            "Mã": sym,
            "Tính cách": f"{nhan_emoji.get(nhan, '')} {nhan}",
            "Character Score": data.get("character_score"),
            "Choppiness Score": data.get("choppiness_score"),
            "Cảnh báo": "; ".join(data.get("canh_bao", [])) or "—",
            "Độ tin cậy thấp": "⚠️ Có" if data.get("do_tin_cay_thap") else "",
        })

    if not rows:
        st.info("Không có mã nào khớp từ khóa tìm kiếm / lựa chọn.")
        return

    st.dataframe(
        pd.DataFrame(rows), width='stretch', hide_index=True,
        column_config={
            "Character Score": st.column_config.NumberColumn(format="%.2f"),
            "Choppiness Score": st.column_config.NumberColumn(format="%.2f"),
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

    with tab_mua:
        if not report["mua"]:
            st.info("Hiện không có mã nào đủ điều kiện khuyến nghị MUA.")
        else:
            rows = []
            for e in report["mua"]:
                entry_range = e.get("khoang_gia_vao_lenh_de_xuat")
                rows.append({
                    "Mã": e["ma"],
                    "Stock Score": f"{e['stock_score']:.2f}" if e.get("stock_score") is not None else "—",
                    "Mẫu hình kỹ thuật": e["chi_tiet"].get("mau_hinh_ky_thuat", "—"),
                    "Vùng giá vào lệnh": f"{entry_range[0]:,.2f} - {entry_range[1]:,.2f}" if entry_range else "—",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
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
                rows.append({
                    "Mã": e["ma"],
                    "Loại": "CẮT LỖ" if e.get("loai_ban") == "CAT_LO" else "CHỐT LỜI",
                    "Ưu tiên": e.get("uu_tien") or "—",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
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
            "MA20": st.column_config.NumberColumn(format="%.2f"),
            "EMA50": st.column_config.NumberColumn(format="%.2f"),
            "EMA200": st.column_config.NumberColumn(format="%.2f"),
            "Khối lượng": st.column_config.NumberColumn(format="%.0f"),
            "Mô hình (%)": st.column_config.NumberColumn(format="%d%%"),
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
# vì cần `storage`/`symbols`/`sectors` đã tính tại thời điểm chạy.
DASHBOARD_SECTIONS = {
    "📋 Watchlist (thêm/xóa mã)": None,
    "📈 Bảng giá theo dõi (Watchlist)": None,
    "🕯️ Biểu đồ nến": None,
    "📒 Nhật ký giao dịch mua/bán": None,
    "🌐 Giai đoạn thị trường (định tính)": None,
    "🌍 Nhập dữ liệu vĩ mô thủ công": None,
    "📋 Báo cáo tổng hợp thị trường chung": None,
    "📐 Giai đoạn thị trường (3 lớp định lượng)": None,
    "📦 Khuyến nghị phân bổ vốn (đơn giản)": None,
    "📦 Khuyến nghị phân bổ vốn (ATR14 chi tiết)": None,
    "🔎 Mã có mô hình thu hẹp biên độ": None,
    "🚦 Báo cáo tín hiệu Mua/Bán": None,
    "🎭 Tính cách giao dịch từng mã": None,
    "📊 Xác suất phục hồi lịch sử": None,
    "🔍 Rà soát danh sách vào lệnh ngắn hạn": None,
    "⏱️ Tiêu chí ngắn hạn": None,
    "💼 Danh mục mô phỏng": None,
}


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


        sectors_input = st.text_input(
            "Danh sách ngành cần xem giai đoạn thị trường (cách nhau bởi dấu phẩy)",
            value="banking,real_estate,securities",
        )
        sectors = [s.strip() for s in sectors_input.split(",") if s.strip()]

        st.divider()
        st.header("📑 Chuyển nhanh tới mục")
        mode = st.radio(
            "Chế độ xem", ["Xem tất cả (cuộn trang)", "Chỉ xem 1 mục"],
            key="dashboard_view_mode", label_visibility="collapsed",
        )
        selected_section = None
        if mode == "Chỉ xem 1 mục":
            selected_section = st.radio(
                "Chọn mục muốn xem", list(DASHBOARD_SECTIONS.keys()),
                key="dashboard_selected_section",
            )

    # --- Danh sách mục theo đúng thứ tự hiển thị, kèm hàm + tham số riêng ---
    sections_to_call = [
        ("📋 Watchlist (thêm/xóa mã)", render_watchlist_manager_section, (storage,)),
        ("📈 Bảng giá theo dõi (Watchlist)", render_watchlist_section, (storage, symbols)),
        ("🕯️ Biểu đồ nến", render_chart_section, (storage, symbols)),
        ("📒 Nhật ký giao dịch mua/bán", render_trade_journal_section, (storage, symbols)),
        ("🌐 Giai đoạn thị trường (định tính)", render_market_regime_section, (storage, sectors)),
        ("🌍 Nhập dữ liệu vĩ mô thủ công", render_manual_macro_data_section, (storage,)),
        ("📋 Báo cáo tổng hợp thị trường chung", render_market_summary_report_section, (storage,)),
        ("📐 Giai đoạn thị trường (3 lớp định lượng)", render_market_regime_quant_section, (storage,)),
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

    if selected_section is not None:
        # --- Chế độ CHỈ XEM 1 MỤC — không cuộn, chỉ hiện đúng mục đã chọn ---
        # LƯU Ý: KHÔNG thêm tiêu đề markdown ở đây — mỗi hàm render_* bên
        # dưới đã tự gọi st.subheader() với đúng tên mục rồi, thêm tiêu đề
        # ở ngoài nữa sẽ làm tiêu đề hiện lặp lại 2 lần trên trang.
        for label, render_fn, args in sections_to_call:
            if label == selected_section:
                render_fn(*args)
                break
    else:
        # --- Chế độ xem TẤT CẢ (hành vi cũ, cuộn từ trên xuống) ---
        for i, (label, render_fn, args) in enumerate(sections_to_call):
            if i > 0:
                st.divider()
            render_fn(*args)


main()
