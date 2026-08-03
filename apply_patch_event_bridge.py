"""
apply_patch_event_bridge.py

Script TỰ ĐỘNG vá file `dashboard/app.py` để tích hợp module sự kiện đa
nhóm (event_risk_classifier.py + macro_score_event_bridge.py) — thay thế
hoàn toàn việc phải tự tay copy-paste từng đoạn code vào đúng vị trí.

CÁCH DÙNG:
  1. Đặt file này vào thư mục gốc dự án (C:\\projects\\pm_ck\\).
  2. Đảm bảo đã có sẵn core/event_risk_classifier.py và
     core/macro_score_event_bridge.py (đã copy ở bước trước).
  3. Chạy: python apply_patch_event_bridge.py
  4. Đọc kỹ thông báo cuối cùng — script sẽ báo THÀNH CÔNG cho từng đoạn
     vá, hoặc DỪNG NGAY và báo lỗi rõ ràng nếu không tìm thấy đúng đoạn
     cần thay (an toàn hơn nhiều so với sửa tay — không bao giờ "sửa
     nhầm chỗ" trong im lặng).

An toàn: script tự động sao lưu file gốc thành `dashboard/app.py.bak`
TRƯỚC khi sửa — nếu có vấn đề gì, chỉ cần xóa app.py rồi đổi tên
app.py.bak trở lại thành app.py để khôi phục nguyên trạng.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

APP_PY_PATH = Path("dashboard/app.py")
BACKUP_PATH = Path("dashboard/app.py.bak")


# ==============================================================================
# Định nghĩa các đoạn cần thay — mỗi patch là (mô_tả, chuỗi_cũ, chuỗi_mới)
# ==============================================================================

PATCH_1_MO_TA = "Thêm UI chọn sự kiện Nhóm B (Tài chính-Tiền tệ) vào Tab 4"
PATCH_1_CU = '''        if current_event_record:
            st.info(
                f"Trạng thái hiện tại: **{EVENT_OPTIONS[current_event_key]}** "
                f"(cập nhật lần cuối: {current_event_record['data'].get('updated_date', '—')})"
            )'''
PATCH_1_MOI = '''        if current_event_record:
            st.info(
                f"Trạng thái hiện tại: **{EVENT_OPTIONS[current_event_key]}** "
                f"(cập nhật lần cuối: {current_event_record['data'].get('updated_date', '—')})"
            )

        st.divider()
        st.markdown("##### Sự kiện Tài chính - Tiền tệ Toàn cầu (Nhóm B — mới bổ sung)")
        st.caption(
            "Dùng cho rủi ro KHÁC xung đột quân sự — vd: can thiệp tỷ giá lớn, "
            "nguy cơ carry-trade unwind, khủng hoảng ngân hàng khu vực. Có thể "
            "TỒN TẠI ĐỒNG THỜI với sự kiện địa chính trị ở trên — hệ thống sẽ tự "
            "lấy mức XẤU NHẤT giữa 2 nhóm, không cộng dồn."
        )
        FINANCIAL_EVENT_OPTIONS = {
            "binh_thuong": "Bình thường — không có rủi ro nổi bật",
            "canh_bao_som": "Cảnh báo sớm (đang có biện pháp ngăn chặn)",
            "dau_hieu_unwind": "Bắt đầu có dấu hiệu rút vốn/unwind thực tế",
            "khung_hoang_toan_dien": "Khủng hoảng toàn diện đã xảy ra",
            "can_thiep_thanh_cong": "Can thiệp thành công, ổn định trở lại",
            "giai_toa_xac_nhan": "Rủi ro giải tỏa hoàn toàn, đã xác nhận",
        }
        current_fin_record = storage.get_latest("manual_macro_setting", "financial_event")
        current_fin_key = (
            current_fin_record["data"]["muc_do"] if current_fin_record else "binh_thuong"
        )
        fin_keys = list(FINANCIAL_EVENT_OPTIONS.keys())
        fin_index = fin_keys.index(current_fin_key) if current_fin_key in fin_keys else 0

        selected_fin_event = st.selectbox(
            "Mức độ sự kiện tài chính-tiền tệ hiện tại", fin_keys,
            index=fin_index, format_func=lambda k: FINANCIAL_EVENT_OPTIONS[k], key="fin_event_select",
        )
        fin_event_note = st.text_area("Ghi chú (tùy chọn)", key="fin_event_note")
        if st.button("Cập nhật sự kiện tài chính-tiền tệ", key="update_fin_event_btn"):
            storage.save("manual_macro_setting", "financial_event", {
                "muc_do": selected_fin_event, "note": fin_event_note,
                "updated_date": date_cls.today().isoformat(),
            })
            st.success(f"Đã cập nhật: {FINANCIAL_EVENT_OPTIONS[selected_fin_event]}.")
            st.rerun()

        if current_fin_record:
            st.info(
                f"Trạng thái hiện tại: **{FINANCIAL_EVENT_OPTIONS[current_fin_key]}** "
                f"(cập nhật lần cuối: {current_fin_record['data'].get('updated_date', '—')})"
            )'''

PATCH_2_MO_TA = "Sửa khối 'Rà soát chi tiết công thức' sang dùng bridge đa nhóm"
PATCH_2_CU = '''        from core.macro_score_engine import DEFAULT_WEIGHTS
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
        result = calc_macro_v2(macro_input)'''
PATCH_2_MOI = '''        from core.macro_score_engine import DEFAULT_WEIGHTS
        from core.macro_score_event_bridge import calculate_macro_score_v2, su_kien_dia_chinh_tri_tu_nhan_cu
        from core.event_risk_classifier import SuKien, NHOM_TAI_CHINH_TIEN_TE
        from core.manual_macro_data import build_full_macro_score_engine_input

        def _load(key):
            record = storage.get_latest("manual_macro_series", key)
            return record["data"]["entries"] if record else []

        target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
        muc_tieu = target_record["data"]["value"] if target_record else None

        geo_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
        geo_key_current = geo_record["data"]["event_key"] if geo_record else "none"
        fin_record = storage.get_latest("manual_macro_setting", "financial_event")
        fin_muc_do_current = fin_record["data"]["muc_do"] if fin_record else "binh_thuong"

        cac_su_kien = [su_kien_dia_chinh_tri_tu_nhan_cu(geo_key_current)]
        if fin_muc_do_current != "binh_thuong":
            cac_su_kien.append(SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do=fin_muc_do_current))

        macro_input = build_full_macro_score_engine_input(
            _load("fed_rate"), _load("usdvnd_rate"),
            cpi_us_series=_load("cpi_us"), cpi_vn_series=_load("cpi_vn"),
            muc_tieu_cpi_vn=muc_tieu,
            interbank_overnight_series=_load("interbank_overnight"),
            interbank_3m_series=_load("interbank_3m"),
            event_key="none",
        )
        result = calculate_macro_score_v2(macro_input, cac_su_kien=cac_su_kien)'''

PATCH_3_MO_TA = "Sửa render_market_summary_report_section() sang dùng bridge đa nhóm"
PATCH_3_CU = '''    from core.capital_allocation_engine import ALLOCATION_TABLE, calculate_stock_allocation_pct
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
    macro_result = calc_macro_v2(macro_input)'''
PATCH_3_MOI = '''    from core.capital_allocation_engine import ALLOCATION_TABLE, calculate_stock_allocation_pct
    from core.macro_score_engine import DEFAULT_WEIGHTS
    from core.macro_score_event_bridge import calculate_macro_score_v2, su_kien_dia_chinh_tri_tu_nhan_cu
    from core.event_risk_classifier import SuKien, NHOM_TAI_CHINH_TIEN_TE
    from core.manual_macro_data import build_full_macro_score_engine_input
    from core.market_breadth import aggregate_layer3_indicators_for_group, calculate_ema200_breadth
    from core.market_regime_detector import detect_market_regime_quant

    # --- LỚP 1: Điểm vĩ mô ---
    def _load(key):
        record = storage.get_latest("manual_macro_series", key)
        return record["data"]["entries"] if record else []

    target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
    muc_tieu = target_record["data"]["value"] if target_record else None

    geo_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
    geo_key_current = geo_record["data"]["event_key"] if geo_record else "none"
    geo_note = geo_record["data"].get("note", "") if geo_record else ""

    fin_record = storage.get_latest("manual_macro_setting", "financial_event")
    fin_muc_do_current = fin_record["data"]["muc_do"] if fin_record else "binh_thuong"
    fin_note = fin_record["data"].get("note", "") if fin_record else ""

    cac_su_kien = [su_kien_dia_chinh_tri_tu_nhan_cu(geo_key_current)]
    if fin_muc_do_current != "binh_thuong":
        cac_su_kien.append(SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do=fin_muc_do_current))

    macro_input = build_full_macro_score_engine_input(
        _load("fed_rate"), _load("usdvnd_rate"),
        cpi_us_series=_load("cpi_us"), cpi_vn_series=_load("cpi_vn"),
        muc_tieu_cpi_vn=muc_tieu,
        interbank_overnight_series=_load("interbank_overnight"),
        interbank_3m_series=_load("interbank_3m"),
        event_key="none",
    )
    macro_result = calculate_macro_score_v2(macro_input, cac_su_kien=cac_su_kien)
    event_note = f"Địa chính trị: {geo_note or '—'} | Tài chính-tiền tệ: {fin_note or '—'}"'''

PATCHES = [
    (PATCH_1_MO_TA, PATCH_1_CU, PATCH_1_MOI),
    (PATCH_2_MO_TA, PATCH_2_CU, PATCH_2_MOI),
    (PATCH_3_MO_TA, PATCH_3_CU, PATCH_3_MOI),
]


def main() -> int:
    if not APP_PY_PATH.exists():
        print(f"❌ KHÔNG TÌM THẤY {APP_PY_PATH}. Hãy chạy script này từ thư mục gốc dự án "
              f"(C:\\projects\\pm_ck), nơi có sẵn thư mục dashboard\\.")
        return 1

    noi_dung = APP_PY_PATH.read_text(encoding="utf-8")

    # --- Kiểm tra TRƯỚC khi sửa gì cả: mỗi đoạn cũ phải xuất hiện ĐÚNG 1 LẦN ---
    loi_kiem_tra = []
    for mo_ta, cu, _ in PATCHES:
        so_lan = noi_dung.count(cu)
        if so_lan == 0:
            loi_kiem_tra.append(
                f"  - [{mo_ta}] KHÔNG TÌM THẤY đoạn cần thay. File có thể đã được sửa "
                f"khác đi so với bản dự kiến, hoặc đã áp dụng patch này rồi trước đó."
            )
        elif so_lan > 1:
            loi_kiem_tra.append(
                f"  - [{mo_ta}] Tìm thấy {so_lan} lần (dự kiến đúng 1 lần) — DỪNG LẠI để "
                f"tránh sửa nhầm chỗ không mong muốn."
            )

    if loi_kiem_tra:
        print("❌ KHÔNG THỂ ÁP DỤNG PATCH AN TOÀN. Chi tiết:")
        print("\n".join(loi_kiem_tra))
        print(
            "\nKhông có gì bị thay đổi trong dashboard/app.py. Hãy gửi lại nội dung "
            "hiện tại của các đoạn liên quan (Tab 4 sự kiện, và "
            "render_market_summary_report_section) để được hỗ trợ viết lại patch "
            "khớp đúng phiên bản hiện tại của bạn."
        )
        return 1

    # --- Sao lưu file gốc trước khi sửa ---
    shutil.copy2(APP_PY_PATH, BACKUP_PATH)
    print(f"✅ Đã sao lưu bản gốc vào: {BACKUP_PATH}")

    # --- Áp dụng lần lượt từng patch ---
    for mo_ta, cu, moi in PATCHES:
        noi_dung = noi_dung.replace(cu, moi, 1)
        print(f"✅ Đã áp dụng: {mo_ta}")

    APP_PY_PATH.write_text(noi_dung, encoding="utf-8")
    print(f"\n🎉 HOÀN TẤT — đã ghi lại {APP_PY_PATH}.")
    print(
        "\nBước tiếp theo:\n"
        "  1. python -m py_compile dashboard\\app.py   (kiểm tra cú pháp)\n"
        "  2. streamlit run dashboard\\app.py           (chạy thử)\n"
        "\nNếu có vấn đề, khôi phục bản gốc bằng:\n"
        "  copy dashboard\\app.py.bak dashboard\\app.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
