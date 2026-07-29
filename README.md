# pm_ck — Phần mềm theo dõi & mô phỏng giao dịch chứng khoán Việt Nam

## ⚠️ Tuyên bố quan trọng

Đây là phần mềm **THEO DÕI VÀ MÔ PHỎNG** giao dịch chứng khoán.
**KHÔNG** kết nối tài khoản thật, **KHÔNG** đặt lệnh giao dịch thật dưới bất
kỳ hình thức nào. Mọi module chỉ đọc dữ liệu, tính toán, đưa ra khuyến nghị
tham khảo, và mô phỏng trên danh mục ảo (paper trading).

Mọi khuyến nghị do hệ thống tạo ra chỉ mang tính chất tham khảo cá nhân,
không phải lời khuyên đầu tư. Người dùng tự chịu trách nhiệm với mọi quyết
định giao dịch thực tế của mình.

## Mục tiêu dự án

Xây dựng một hệ thống module hóa giúp:
1. Thu thập dữ liệu giá, dữ liệu cơ bản doanh nghiệp, và **đặc biệt là dữ
   liệu vĩ mô** (tỷ giá, OMO, lãi suất, chính sách ngành) — nhóm dữ liệu vĩ
   mô được ưu tiên cao nhất trong toàn bộ hệ thống.
2. Tính toán các chỉ báo kỹ thuật cốt lõi (MA20, EMA50/100/200, volume
   trung bình) và nhận diện mô hình tích lũy thu hẹp biên độ (10-30 tháng).
3. Xác định giai đoạn thị trường theo đúng trình tự: **vĩ mô trước, kỹ
   thuật (EMA200) sau**, có độ trễ xác nhận để chống nhiễu.
4. Đưa ra khuyến nghị phân bổ vốn kèm **khoảng giá vào lệnh** cụ thể
   (entry_price_range), mức cắt lỗ, và khối lượng tối đa theo nguyên tắc
   quản trị rủi ro (tối đa 2% NAV/lệnh, 20% NAV toàn danh mục).
5. Quản lý danh mục mô phỏng (paper portfolio), cảnh báo qua Telegram, và
   hiển thị toàn bộ thông tin trên dashboard Streamlit.
6. Cho phép backtest (kiểm chứng lịch sử) các quy tắc tín hiệu trước khi
   tin tưởng sử dụng.

## Cấu trúc thư mục

```
pm_ck/
├── config/
│   └── config.yaml              # Cấu hình trung tâm toàn hệ thống
├── core/
│   ├── data_collector.py        # [Giai đoạn 1] Thu thập dữ liệu (giá, cơ bản, vĩ mô)
│   ├── indicators.py            # [Giai đoạn 2] Tính chỉ báo kỹ thuật
│   ├── pattern_detector.py      # [Giai đoạn 2] Nhận diện mô hình thu hẹp biên độ
│   ├── market_regime_detector.py# [Giai đoạn 3] Xác định giai đoạn thị trường
│   ├── capital_allocator.py     # [Giai đoạn 3] Khuyến nghị phân bổ vốn
│   ├── paper_portfolio.py       # [Giai đoạn 4] Danh mục mô phỏng
│   ├── notifier.py              # [Giai đoạn 4] Cảnh báo qua Telegram
│   └── storage.py               # Lưu trữ dữ liệu (SQLite mặc định)
├── backtest/
│   └── backtest_engine.py       # [Giai đoạn 2] Kiểm chứng trên dữ liệu lịch sử
├── dashboard/
│   └── app.py                   # [Giai đoạn 5] Giao diện Streamlit
├── tests/                       # Unit test cho từng module (dùng dữ liệu giả lập)
├── main.py                      # Điểm khởi chạy tổng
├── requirements.txt
└── README.md
```

## Trạng thái triển khai các module

| Module | Giai đoạn | Trạng thái |
|---|---|---|
| Khung dự án | 0 | ✅ Hoàn thành |
| `data_collector.py` | 1 | ✅ Hoàn thành (30 test) |
| `indicators.py` | 2 | ✅ Hoàn thành (15 test) |
| `pattern_detector.py` | 2 | ✅ Hoàn thành (12 test) |
| `backtest_engine.py` | 2 | ✅ Hoàn thành (20 test) |
| `market_regime_detector.py` | 3 | ✅ Hoàn thành (26 test) |
| `capital_allocator.py` | 3 | ✅ Hoàn thành (29 test) |
| `paper_portfolio.py` | 4 | ✅ Hoàn thành (24 test) |
| `notifier.py` | 4 | ✅ Hoàn thành (17 test) |
| `storage.py` | (bổ sung) | ✅ Hoàn thành (13 test) |
| `dashboard/app.py` | 5 | ✅ Hoàn thành (5 smoke test) |

**Tổng cộng: 191/191 unit test PASSED.**

## Chạy dashboard

```bash
streamlit run dashboard/app.py
```

Mở trình duyệt tại địa chỉ được hiển thị (mặc định `http://localhost:8501`).
Dashboard đọc dữ liệu qua `core/storage.py` — cần các module khác (đặc
biệt `data_collector`, `indicators`, `pattern_detector`,
`market_regime_detector`, `capital_allocator`, `paper_portfolio`) đã lưu
dữ liệu vào storage trước đó (qua `main.py` hoặc script riêng) thì dashboard
mới có nội dung để hiển thị. Nếu chưa có dữ liệu, dashboard vẫn chạy bình
thường và hiển thị thông báo "Chưa có dữ liệu" ở từng phần.

## Cách cài đặt

### 1. Tạo môi trường ảo Python

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

Tạo file `.env` ở thư mục gốc (không commit lên Git), ví dụ:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### 4. Chạy chương trình

```bash
python main.py
```

### 5. Chạy dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Chạy unit test

```bash
pytest tests/ -v
```

## Nguyên tắc phát triển

- Code từng module theo đúng thứ tự Giai đoạn 1 → 5, không nhảy cóc — các
  module sau phụ thuộc vào output của module trước.
- Mỗi module viết kèm unit test dùng dữ liệu giả lập (mock), không gọi API
  thật khi chạy test.
- `market_regime_detector.py` và `capital_allocator.py` là 2 module phức
  tạp và ảnh hưởng trực tiếp đến khuyến nghị đầu tư — cần tự kiểm tra thủ
  công đối chiếu biểu đồ thực tế trước khi tin tưởng sử dụng.
- Chế độ dữ liệu trễ (`delayed_mode: true`) được bật mặc định trong
  `config.yaml` để tránh phát sinh chi phí dữ liệu real-time trong giai
  đoạn thử nghiệm.
