"""
check_macro_score.py
======================
Script KIỂM TRA — hiện chi tiết TỪNG THÀNH PHẦN của công thức tính điểm
vĩ mô, dựa trên dữ liệu bạn ĐÃ NHẬP qua dashboard. Dùng để tự rà soát,
đối chiếu xem công thức có tính đúng như bạn kỳ vọng không.

Chạy: python check_macro_score.py
"""

import yaml

from core.macro_score_engine import calculate_macro_score
from core.manual_macro_data import build_full_macro_score_engine_input
from core.storage import Storage


def _load_series(storage: Storage, key: str) -> list[dict]:
    record = storage.get_latest("manual_macro_series", key)
    return record["data"]["entries"] if record else []


def main() -> None:
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    storage = Storage(db_path=config["storage"]["path"])

    fed_series = _load_series(storage, "fed_rate")
    fx_series = _load_series(storage, "usdvnd_rate")
    cpi_us_series = _load_series(storage, "cpi_us")
    cpi_vn_series = _load_series(storage, "cpi_vn")
    interbank_overnight_series = _load_series(storage, "interbank_overnight")
    interbank_3m_series = _load_series(storage, "interbank_3m")

    target_record = storage.get_latest("manual_macro_setting", "cpi_vn_target")
    muc_tieu_cpi_vn = target_record["data"]["value"] if target_record else None

    event_record = storage.get_latest("manual_macro_setting", "geopolitical_event")
    event_key = event_record["data"]["event_key"] if event_record else None

    storage.close()

    print("=" * 70)
    print("DỮ LIỆU THÔ ĐÃ NHẬP")
    print("=" * 70)
    print(f"Fed Rate ({len(fed_series)} điểm):", fed_series[-3:] if fed_series else "chưa có")
    print(f"Tỷ giá USD/VND ({len(fx_series)} điểm):", fx_series[-3:] if fx_series else "chưa có")
    print(f"CPI Mỹ ({len(cpi_us_series)} điểm):", cpi_us_series[-3:] if cpi_us_series else "chưa có")
    print(f"CPI VN ({len(cpi_vn_series)} điểm):", cpi_vn_series[-3:] if cpi_vn_series else "chưa có")
    print(f"Lãi suất qua đêm ({len(interbank_overnight_series)} điểm):",
          interbank_overnight_series[-3:] if interbank_overnight_series else "chưa có")
    print(f"Lãi suất 3 tháng ({len(interbank_3m_series)} điểm):",
          interbank_3m_series[-3:] if interbank_3m_series else "chưa có")
    print(f"Mục tiêu CPI VN:", muc_tieu_cpi_vn if muc_tieu_cpi_vn else "mặc định 4.0%")
    print(f"Sự kiện địa chính trị:", event_key if event_key else "chưa cập nhật (mặc định 'none')")

    macro_input = build_full_macro_score_engine_input(
        fed_series, fx_series,
        cpi_us_series=cpi_us_series, cpi_vn_series=cpi_vn_series,
        muc_tieu_cpi_vn=muc_tieu_cpi_vn,
        interbank_overnight_series=interbank_overnight_series,
        interbank_3m_series=interbank_3m_series,
        event_key=event_key,
    )

    print("\n" + "=" * 70)
    print("INPUT ĐÃ TỔNG HỢP CHO macro_score_engine")
    print("=" * 70)
    for k, v in macro_input.items():
        print(f"  {k}: {v}")

    result = calculate_macro_score(macro_input)

    print("\n" + "=" * 70)
    print("KẾT QUẢ CHI TIẾT TỪNG THÀNH PHẦN (đã nhân trọng số)")
    print("=" * 70)
    from core.macro_score_engine import DEFAULT_WEIGHTS
    for group, sub_score in result["chi_tiet_sub_scores"].items():
        weight = DEFAULT_WEIGHTS[group]
        print(f"  {group:12s}: điểm thô={sub_score:+.3f}  x  trọng số={weight:.2f}  =  {sub_score*weight:+.3f}")

    print("\n" + "=" * 70)
    print(f"MACRO SCORE TỔNG: {result['macro_score']:+.3f}")
    print(f"NHÃN: {result['nhan']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
