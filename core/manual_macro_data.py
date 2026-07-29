"""
manual_macro_data.py
======================
[Bổ sung — Quản lý dữ liệu vĩ mô NHẬP THỦ CÔNG theo thời gian]

Vì lãi suất Fed Funds Rate và tỷ giá USD/VND CHƯA có adapter thu thập dữ
liệu tự động (không giống cổ phiếu đã có `vnstock`), module này cho phép
người dùng TỰ NHẬP các điểm dữ liệu theo thời gian, rồi tự động tính các
đại lượng cần thiết làm input cho `core/macro_score_engine.py`:
    - Delta so với lần cập nhật trước (dùng cho Fed Funds Rate).
    - % thay đổi từ đầu năm (YTD) — dùng cho tỷ giá.
    - Số kỳ tăng liên tiếp gần nhất — dùng cho tỷ giá.
    - Khoảng cách (%) tới đỉnh lịch sử — dùng cho tỷ giá.

Chỉ thao tác trên dữ liệu THUẦN (list of {"date","value"}) — việc lưu trữ
bền vào storage do `dashboard/app.py` đảm nhiệm (gọi `core.storage`).
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Optional


class InvalidMacroSeriesError(ValueError):
    """Dữ liệu chuỗi thời gian vĩ mô không hợp lệ."""


def add_entry(series: list[dict], entry_date: date_cls, value: float) -> list[dict]:
    """Thêm 1 điểm dữ liệu mới vào chuỗi, giữ chuỗi LUÔN sắp xếp theo thời
    gian tăng dần. Nếu đã có sẵn 1 điểm ĐÚNG ngày đó, GHI ĐÈ giá trị cũ
    (coi như cập nhật lại số liệu ngày đó) thay vì tạo bản ghi trùng.

    Trả về DANH SÁCH MỚI (không sửa `series` gốc).
    """
    date_str = entry_date.isoformat()
    updated = [e for e in series if e["date"] != date_str]
    updated.append({"date": date_str, "value": value})
    updated.sort(key=lambda e: e["date"])
    return updated


def remove_entry(series: list[dict], entry_date: date_cls) -> list[dict]:
    """Xóa điểm dữ liệu ĐÚNG ngày `entry_date` khỏi chuỗi (dùng để sửa
    lỗi nhập nhầm — vd. nhập sai giá trị hoặc sai ngày). Không báo lỗi
    nếu ngày đó không tồn tại trong chuỗi (coi như không có gì để xóa).

    Trả về DANH SÁCH MỚI (không sửa `series` gốc).
    """
    date_str = entry_date.isoformat()
    return [e for e in series if e["date"] != date_str]


def compute_delta_last(series: list[dict]) -> Optional[float]:
    """Chênh lệch giữa điểm dữ liệu MỚI NHẤT và điểm NGAY TRƯỚC ĐÓ.

    Dùng cho `fed_rate_delta_last_meeting` trong macro_score_engine.
    Trả về None nếu chưa đủ 2 điểm dữ liệu.
    """
    if len(series) < 2:
        return None
    return series[-1]["value"] - series[-2]["value"]


def compute_ytd_change_pct(
    series: list[dict], as_of: Optional[date_cls] = None
) -> Optional[float]:
    """% thay đổi từ điểm dữ liệu ĐẦU NĂM (của năm chứa `as_of`, mặc định
    là năm của điểm dữ liệu mới nhất) tới điểm MỚI NHẤT.

    Dùng cho `fx_ytd_change_pct` trong macro_score_engine.
    """
    if not series:
        return None

    latest = series[-1]
    as_of = as_of or date_cls.fromisoformat(latest["date"])
    year = as_of.year

    year_entries = [e for e in series if date_cls.fromisoformat(e["date"]).year == year]
    if not year_entries:
        return None

    first_of_year = year_entries[0]
    if first_of_year["value"] == 0:
        raise InvalidMacroSeriesError("Giá trị đầu năm bằng 0, không thể tính % thay đổi.")

    return (latest["value"] - first_of_year["value"]) / first_of_year["value"] * 100.0


def compute_consecutive_increases(series: list[dict]) -> int:
    """Đếm số điểm dữ liệu GẦN NHẤT liên tiếp có giá trị TĂNG so với điểm
    ngay trước nó (theo đúng thứ tự các điểm đã nhập — không giả định
    khoảng cách chính xác 1 tuần giữa các điểm).

    Dùng cho `fx_so_tuan_tang_lien_tiep` trong macro_score_engine.
    """
    if len(series) < 2:
        return 0

    count = 0
    for i in range(len(series) - 1, 0, -1):
        if series[i]["value"] > series[i - 1]["value"]:
            count += 1
        else:
            break
    return count


def compute_distance_from_peak_pct(series: list[dict]) -> Optional[float]:
    """% khoảng cách từ điểm MỚI NHẤT tới ĐỈNH LỊCH SỬ (giá trị cao nhất
    từng ghi nhận trong toàn bộ chuỗi).

    Dùng cho `fx_khoang_cach_dinh_pct` trong macro_score_engine.
    Trả về 0 nếu điểm mới nhất CHÍNH LÀ đỉnh lịch sử.
    """
    if not series:
        return None

    peak = max(e["value"] for e in series)
    latest = series[-1]["value"]

    if peak == 0:
        raise InvalidMacroSeriesError("Đỉnh lịch sử bằng 0, không thể tính khoảng cách.")

    return (peak - latest) / peak * 100.0


# ==============================================================================
# TỔNG HỢP INPUT CHO macro_score_engine.calculate_macro_score()
# ==============================================================================

# Giá trị TRUNG TÍNH (đóng góp 0 điểm) cho các nhóm chỉ số CHƯA có dữ liệu
# nhập tay (CPI Mỹ/VN, lãi suất liên ngân hàng, sự kiện địa chính trị) —
# để macro_score_engine vẫn tính được điểm PHẦN NÀO từ những gì ĐÃ có
# (Fed Rate + tỷ giá), thay vì bắt buộc phải có ĐỦ cả 6 nhóm mới chạy được.
NEUTRAL_DEFAULTS = {
    "cpi_us_yoy": 2.0,               # đúng mục tiêu Fed -> sub-score = 0
    "cpi_us_mom_3thang": [0.0, 0.0, 0.0],
    "cpi_vn_yoy": 4.0,               # đúng mục tiêu NHNN -> sub-score = 0
    "muc_tieu_cpi_vn": 4.0,
    "interbank_do_doc_duong_cong": 0.0,
    "interbank_thay_doi_tuan_3m": 0.0,
    "su_kien_hien_tai": "none",
}


def build_macro_score_engine_input(
    fed_rate_series: list[dict], usdvnd_series: list[dict]
) -> dict:
    """Tổng hợp dữ liệu vĩ mô đã NHẬP TAY (Fed Rate, tỷ giá USD/VND)
    thành đúng cấu trúc input cho `core.macro_score_engine.calculate_macro_score()`.

    Các nhóm CHƯA có dữ liệu nhập tay (CPI Mỹ/VN, lãi suất liên ngân
    hàng, sự kiện địa chính trị) dùng giá trị TRUNG TÍNH mặc định (đóng
    góp đúng 0 điểm vào tổng) — nghĩa là Macro Score hiện tại CHỈ PHẢN
    ÁNH được phần Fed Rate + tỷ giá, chưa đầy đủ 6 nhóm theo tài liệu gốc.
    """
    fed_delta = compute_delta_last(fed_rate_series)
    fx_ytd = compute_ytd_change_pct(usdvnd_series)
    fx_weeks_up = compute_consecutive_increases(usdvnd_series)
    fx_distance_peak = compute_distance_from_peak_pct(usdvnd_series)

    return {
        "fed_rate_delta_last_meeting": fed_delta if fed_delta is not None else 0.0,
        "fed_dotplot_delta": 0.0,  # chưa có nguồn dữ liệu dot-plot -> trung tính
        "fx_ytd_change_pct": fx_ytd if fx_ytd is not None else 0.0,
        "fx_so_tuan_tang_lien_tiep": fx_weeks_up,
        "fx_khoang_cach_dinh_pct": fx_distance_peak if fx_distance_peak is not None else 2.0,
        **NEUTRAL_DEFAULTS,
    }


# ==============================================================================
# CPI MỸ — chuỗi 2 giá trị/điểm (YoY + MoM), khác cấu trúc chuỗi đơn giản ở trên
# ==============================================================================

def add_cpi_us_entry(
    series: list[dict], entry_date: date_cls, cpi_yoy: float, cpi_mom: float
) -> list[dict]:
    """Thêm 1 điểm dữ liệu CPI Mỹ (gồm CẢ YoY% và MoM% cùng lúc, vì công
    thức `score_cpi_us` cần cả 2). Ghi đè nếu đã có điểm đúng ngày đó.
    """
    date_str = entry_date.isoformat()
    updated = [e for e in series if e["date"] != date_str]
    updated.append({"date": date_str, "yoy": cpi_yoy, "mom": cpi_mom})
    updated.sort(key=lambda e: e["date"])
    return updated


def get_latest_cpi_us_yoy(series: list[dict]) -> Optional[float]:
    """CPI YoY Mỹ tại điểm dữ liệu MỚI NHẤT."""
    if not series:
        return None
    return series[-1]["yoy"]


def get_recent_cpi_us_mom(series: list[dict], n: int = 3) -> list[float]:
    """MoM% của `n` tháng GẦN NHẤT (mặc định 3 tháng, đúng công thức
    `score_cpi_us`). Trả về danh sách RỖNG nếu chưa có dữ liệu.
    """
    if not series:
        return []
    return [e["mom"] for e in series[-n:]]


# ==============================================================================
# TỔNG HỢP INPUT ĐẦY ĐỦ 6 NHÓM (mở rộng từ hàm trên, tương thích ngược)
# ==============================================================================

def build_full_macro_score_engine_input(
    fed_rate_series: list[dict],
    usdvnd_series: list[dict],
    cpi_us_series: Optional[list[dict]] = None,
    cpi_vn_series: Optional[list[dict]] = None,
    muc_tieu_cpi_vn: Optional[float] = None,
    interbank_overnight_series: Optional[list[dict]] = None,
    interbank_3m_series: Optional[list[dict]] = None,
    event_key: Optional[str] = None,
) -> dict:
    """Bản MỞ RỘNG của `build_macro_score_engine_input()` — nhận đủ dữ
    liệu cho CẢ 6 NHÓM chỉ số (không chỉ Fed Rate + tỷ giá). Nhóm nào
    không truyền vào (None/rỗng) sẽ dùng giá trị TRUNG TÍNH mặc định,
    nên có thể gọi hàm này ngay cả khi CHỈ có 1 vài nhóm dữ liệu — không
    bắt buộc phải có đủ cả 6 mới chạy được.
    """
    base = build_macro_score_engine_input(fed_rate_series, usdvnd_series)

    if cpi_us_series:
        yoy = get_latest_cpi_us_yoy(cpi_us_series)
        mom = get_recent_cpi_us_mom(cpi_us_series, n=3)
        if yoy is not None and mom:
            base["cpi_us_yoy"] = yoy
            base["cpi_us_mom_3thang"] = mom

    if cpi_vn_series:
        yoy_vn = cpi_vn_series[-1]["value"]
        base["cpi_vn_yoy"] = yoy_vn

    if muc_tieu_cpi_vn is not None:
        base["muc_tieu_cpi_vn"] = muc_tieu_cpi_vn

    if interbank_overnight_series and interbank_3m_series:
        overnight_latest = interbank_overnight_series[-1]["value"]
        rate_3m_latest = interbank_3m_series[-1]["value"]
        base["interbank_do_doc_duong_cong"] = rate_3m_latest - overnight_latest

        weekly_change = compute_delta_last(interbank_3m_series)
        base["interbank_thay_doi_tuan_3m"] = weekly_change if weekly_change is not None else 0.0

    if event_key is not None:
        base["su_kien_hien_tai"] = event_key

    return base
