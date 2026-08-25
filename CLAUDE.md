# CLAUDE.md — Ngữ cảnh dự án pm_ck cho Claude Code

> File này được Claude (trên claude.ai) tạo ra khi bàn giao dự án sang Claude
> Code, để phiên làm việc mới nắm được toàn bộ bối cảnh mà không cần đọc lại
> lịch sử chat. Cập nhật file này mỗi khi có thay đổi kiến trúc lớn.

## 1. Dự án là gì

`pm_ck` là hệ thống **theo dõi & mô phỏng giao dịch chứng khoán Việt Nam**
cho cá nhân — **KHÔNG PHẢI** công cụ đặt lệnh tự động. Người dùng (Tuyên,
Trưởng phòng KHTH tại BV TWQĐ 108, không phải lập trình viên) tự nhập lệnh
mô phỏng dựa trên khuyến nghị hệ thống đưa ra.

**Gồm 2 phần chính:**
- **Backend Python** (`core/`, `main.py`, `run_full_market.py`) — thu thập dữ
  liệu, tính chỉ báo, phát hiện mô hình, xác định giai đoạn thị trường, khuyến
  nghị phân bổ vốn, tín hiệu mua/bán.
- **Dashboard Streamlit** (`dashboard/app.py`) — giao diện xem toàn bộ kết quả,
  quản lý watchlist, nhật ký giao dịch, nhập dữ liệu vĩ mô thủ công.

**Nguồn dữ liệu giá:** `vnstock` (Python, không cần trả phí, gói Cộng đồng
60 request/phút sau khi đăng ký qua `register_user()`).

## 2. Nguyên tắc bắt buộc xuyên suốt dự án

1. **KHÔNG BAO GIỜ đặt lệnh giao dịch thật.** Mọi module chỉ tạo khuyến nghị
   tham khảo — người dùng tự xác nhận và đặt lệnh thủ công qua công ty
   chứng khoán riêng.
2. **Luôn viết test trước khi giao nộp code.** Dự án có kỷ luật test rất cao
   — hiện tại **615 test, tất cả phải pass** trước khi coi là xong việc nào
   đó. Chạy `pytest tests/ -v` sau MỌI thay đổi.
3. **Không đoán cấu trúc API bên ngoài (đặc biệt `vnstock`).** Luôn viết
   script nhỏ kiểm tra cấu trúc dữ liệu thật trước khi code adapter — cấu
   trúc cột `vnstock` trả về đã từng khác giả định nhiều lần.
4. **Test dashboard PHẢI cách ly hoàn toàn khỏi database thật.** Từng có sự
   cố nghiêm trọng: test ghi đè dữ liệu giả vào đúng file `./data/pm_ck.db`
   thật của người dùng. Test hiện dùng biến môi trường `PM_CK_DB_PATH` +
   `tmp_path` của pytest — KHÔNG được thay đổi lại cách này.
4b. **CẢNH GIÁC với `storage.query_all_keys(category)` để suy ra "danh sách
   mã/ngành của lần chạy hiện tại"** — storage là APPEND-ONLY, key cũ từ
   CÁC LẦN CHẠY TRƯỚC (vd. quét toàn thị trường ~1500 mã theo ngành vnstock,
   sau đó đổi sang watchlist tùy chỉnh nhỏ hơn) KHÔNG BAO GIỜ tự mất — nếu
   một hàm tổng hợp tự suy ra phạm vi bằng cách gọi `query_all_keys()` thay
   vì nhận danh sách hiện tại làm tham số, kết quả sẽ TRỘN LẪN dữ liệu
   cũ/mới (sự cố thực tế 27/07/2026: 718 mã thay vì ~210, 42 ngành thay vì
   17 — xem `run_full_market.aggregate_market_regime_by_sector()` và
   `compute_stock_signals_for_all_symbols()`, cả 2 đã sửa để BẮT BUỘC nhận
   `symbol_sector_map` làm tham số, kèm bước dọn dẹp ngành cũ qua
   `storage.delete_key()`). Khi thêm hàm tổng hợp mới tương tự, LUÔN nhận
   phạm vi (danh sách mã/ngành) làm tham số tường minh, không tự suy ra từ
   toàn bộ lịch sử storage.
4c. **`run_full_market.py` giờ tính CẢ 2 loại khuyến nghị phân bổ vốn**
   (đơn giản `capital_allocator.py` + ATR14 chi tiết
   `capital_allocation_engine.py`) qua `compute_capital_allocations_for_all_symbols()`
   — trước đây 2 bước này CHỈ chạy qua `main.py` (watchlist nhỏ), khiến
   dashboard trống rỗng với watchlist tùy chỉnh lớn quét qua
   `run_full_market.py` (sự cố thực tế 27/07/2026, đã sửa). Khi thêm bước
   tính toán mới cần chạy cho watchlist lớn, LUÔN kiểm tra đã gắn vào CẢ 2
   entry point (`main.py` VÀ `run_full_market.py`), không chỉ 1 trong 2.
4c2. **CÙNG LỖI TÁI DIỄN 28/07/2026**: mục "Giai đoạn thị trường (định
   tính)" trên dashboard (đọc category `market_regime`) cũng CHỈ được ghi
   bởi `main.py` — `run_full_market.py` không hề gọi
   `detect_market_regime()` (chỉ gọi bản định lượng
   `detect_market_regime_quant()`). Đã sửa: `aggregate_market_regime_by_sector()`
   giờ tính VÀ LƯU CẢ 2 (`market_regime` + `market_regime_quant`) cho mỗi
   ngành, kèm dọn dẹp ngành cũ cho CẢ 2 category. **BÀI HỌC LẶP LẠI**: đây
   là lần THỨ HAI cùng một loại lỗi (thiếu 1 trong 2 entry point) xảy ra
   — khi thêm BẤT KỲ category storage mới nào được ghi bởi `main.py`,
   PHẢI kiểm tra ngay xem `run_full_market.py` có cần ghi category đó
   tương tự hay không, thay vì đợi người dùng phát hiện dashboard trống.
4d. **Watchlist (thêm/xóa mã hiển thị)** không còn cố định ở sidebar — đã
   chuyển thành 1 mục trong danh sách điều hướng "📑 Chuyển nhanh tới mục"
   (`render_watchlist_manager_section()`), giống các module khác. Sidebar
   giờ chỉ còn: ô nhập ngành cần xem + menu điều hướng nhanh (xem/ẩn từng
   mục, tránh phải cuộn trang dài — dashboard có 15 mục).
4e. **Ô tìm kiếm mã tự động hiện khi danh sách >5 mã**
   (`filter_symbols_by_search()` + `render_search_box_if_needed()` trong
   `dashboard/app.py`) — áp dụng ở Watchlist, chọn mã biểu đồ, rà soát
   danh sách vào lệnh, báo cáo tín hiệu mua/bán.
4g. **Bảng "Thông tin chi tiết" (Watchlist) dùng `st.data_editor`**, KHÔNG
   dùng `st.columns` vẽ thủ công — vẽ thủ công từng cột hẹp làm chữ/số bị
   NGẮT DÒNG GIỮA TỪ rất xấu (sự cố thực tế 28/07/2026, vd. "DOWNTREND"
   tách thành "DOWNTR"/"END"). Cột số (giá/MA/EMA/khối lượng) giữ kiểu
   **số thật** (float, KHÔNG format thành chuỗi) trong
   `build_watchlist_detail_table()` — định dạng hiển thị qua
   `column_config` của `st.data_editor`, không format tay bằng f-string.
   Xóa mã tích hợp qua cột tick "🗑️ Xóa" + nút xác nhận xóa hàng loạt
   (`remove_symbols_from_watchlist()` — hàm thuần túy, test độc lập).
   **LƯU Ý**: `streamlit.testing.v1.AppTest` KHÔNG hỗ trợ mô phỏng thao
   tác tick/sửa ô trong `st.data_editor` (không có `at.data_editor`) —
   logic nghiệp vụ liên quan PHẢI tách thành hàm thuần túy riêng để test
   được, không thể test qua tương tác UI giả lập.
4h. **WATCHLIST RIÊNG THEO NGƯỜI XEM** (28/07/2026) — hỗ trợ nhiều người
   CÙNG xem 1 dashboard (server chạy trên 1 máy, người khác truy cập qua
   link mạng LAN hoặc deploy public) nhưng MỖI NGƯỜI theo dõi 1 danh sách
   mã KHÁC NHAU, trong khi TOÀN BỘ dữ liệu thị trường (giá/chỉ báo/tín
   hiệu/giai đoạn thị trường...) vẫn dùng CHUNG. Cơ chế: định danh người
   xem lấy từ tham số URL `?user=...` (`get_current_user_id()`), dùng làm
   KEY riêng khi lưu/đọc watchlist (`load_watchlist(storage, user_id)`/
   `save_watchlist(storage, symbols, user_id)`) — KHÔNG dùng cookie/đăng
   nhập thật, chỉ đơn giản là quy ước link riêng cho từng người
   (`...?user=ten_nguoi`). Không có `?user=` -> dùng watchlist "default"
   chung (tương thích ngược hoàn toàn với dữ liệu cũ).
4i. **LỖI NGHIÊM TRỌNG ĐÃ SỬA 28/07/2026: `dashboard/app.py` TRƯỚC ĐÂY
   KHÔNG HỀ ĐỌC `config.yaml`** — biến `DB_PATH` cố định hard-code
   `"./data/pm_ck.db"`, chỉ override được qua biến môi trường
   `PM_CK_DB_PATH` (vốn chỉ dành cho test). Nghĩa là đổi
   `config.yaml -> storage.path` sang Supabase KHÔNG có tác dụng gì với
   dashboard (dù `main.py`/`run_full_market.py` vẫn đọc đúng
   `config.yaml` bình thường qua `load_config()`) — 3 nơi ĐỌC KHÔNG ĐỒNG
   NHẤT cùng 1 cấu hình. Đã sửa bằng hàm `_resolve_db_path()` trong
   `dashboard/app.py`, thứ tự ưu tiên: biến môi trường `PM_CK_DB_PATH` >
   `st.secrets["SUPABASE_CONNECTION_STRING"]` (Streamlit Cloud) >
   `config.yaml` > mặc định cục bộ. Đồng thời thêm `resolve_storage_path()`
   dùng CHUNG trong `main.py` (và `run_full_market.py` import lại) để
   CẢ 3 entry point (`main.py`, `run_full_market.py`, `dashboard/app.py`)
   đọc storage path theo ĐÚNG CÙNG 1 thứ tự ưu tiên — tránh lặp lại kiểu
   lỗi "đọc cấu hình không đồng nhất giữa các entry point" (xem thêm mục
   4c/4c2 — đã từng gặp 2 lần với các bước xử lý khác trong pipeline).
4j. **Bảo mật khi đưa code lên GitHub**: `config.yaml` trong repo LUÔN
   giữ giá trị AN TOÀN (SQLite cục bộ, không có mật khẩu thật). Mật khẩu
   Supabase thật chỉ nằm ở 2 nơi, CẢ HAI đều KHÔNG commit lên git:
   (1) biến môi trường `PM_CK_DB_PATH` đặt trên từng máy cục bộ (dùng cho
   `main.py`/`run_full_market.py`/`dashboard` chạy tại chỗ), (2)
   `.streamlit/secrets.toml` thật trên Streamlit Community Cloud (nhập
   qua giao diện web Settings -> Secrets, không phải file trong repo) —
   xem `.streamlit/secrets.toml.example` (file MẪU, an toàn, không có
   giá trị thật) để biết định dạng cần nhập.
4k. **Mục "🧮 Cổ phiếu dài hạn" (bổ sung 25/08/2026) THAY THẾ HOÀN TOÀN**
   mục "🚦 Báo cáo tín hiệu Mua/Bán từng mã" cũ trên dashboard (theo yêu
   cầu người dùng) — dashboard KHÔNG còn hiển thị tín hiệu MUA/BÁN/GIỮ
   ngắn hạn của `stock_signal_engine.py` ở đâu nữa (`main.py`/
   `run_full_market.py` VẪN tính và lưu `signal_summary_report` như cũ,
   chỉ là không còn nơi nào trên dashboard đọc lại category này). Mục mới
   là 1 BỘ LỌC CỔ PHIẾU DÀI HẠN: backtest 8 bộ chỉ số kỹ thuật
   (`core/long_term_indicator_backtest.py`, TÁI SỬ DỤNG
   `backtest/backtest_engine.py::run_backtest()` có sẵn — KHÔNG tự viết
   lại engine) cho TỪNG MÃ trong watchlist, tách theo giai đoạn Uptrend/
   Sideway/Downtrend tại NGÀY VÀO LỆNH của từng giao dịch lịch sử, theo
   CẢ 2 phương pháp phân loại giai đoạn song song: (1) EMA200-đơn giản
   (`market_regime_detector.tinh_chuoi_giai_doan_theo_ngay`, đã có), (2)
   Ensemble 3 phương pháp walk-forward THEO TỪNG NGÀY cho 1 mã — hàm MỚI
   `market_regime_ensemble.tinh_chuoi_ensemble_theo_ngay()` (khác với
   `phan_tich_ensemble_theo_nhom()` đã có, vốn chỉ tính cho NGÀY GẦN NHẤT
   của 1 NHÓM nhiều mã, không phải chuỗi lịch sử của 1 mã đơn lẻ). Kết quả
   lưu vào category MỚI `long_term_screener_report` (key=mã) qua bước
   pipeline MỚI `main.run_long_term_screener_step()` — gọi ở CẢ 2 entry
   point (`main.py` cuối `run_pipeline()`, `run_full_market.py` sau
   `run_market_regime_ensemble_step`), đúng bài học 4c/4c2 bên dưới. ĐÂY
   LÀ BƯỚC RẤT NẶNG khi chạy cho watchlist lớn — Phương pháp Ensemble cần
   fit mô hình Markov cho TỪNG mã (refit mỗi 5 phiên, ước tính ~26
   giây/mã với lịch sử ~750 phiên) ⇒ **~90 phút CHỈ RIÊNG PHẦN NÀY cho
   toàn bộ ~212 mã**, cộng thêm vào runtime hiện tại của
   `run_full_market.py` (~35-40 phút). Vì vậy bước này TỰ CHECKPOINT theo
   từng mã bằng chính storage (mã đã có bản ghi trong
   `long_term_screener_report` sẽ được bỏ qua ở lần chạy sau, trừ khi gọi
   với `force_recompute=True`) — KHÔNG cần thêm category checkpoint riêng
   như bước fetch OHLCV.
4f. **CẢNH GIÁC với việc thêm QUÁ NHIỀU test `AppTest` liên tiếp trong 1
   file test dashboard** — đã gặp hiện tượng "tràn trạng thái form" thực
   tế (27/07/2026): thêm 2 test AppTest MỚI vào cuối `test_dashboard.py`
   (đã có ~13 test AppTest khác dùng `st.form`) khiến chúng FAIL dù chạy
   riêng lẻ PASS 100% — lỗi "st.button() can't be used in an st.form()"
   xuất hiện ở nơi hoàn toàn không liên quan, do Streamlit tích lũy trạng
   thái form giữa nhiều `AppTest.from_file()` liên tiếp trong cùng 1 phiên
   pytest. CÁCH XỬ LÝ: ưu tiên viết test HÀM THUẦN TÚY (không qua
   `AppTest`) khi có thể; nếu cần kiểm tra hành vi UI cụ thể, GHÉP vào một
   test smoke ĐÃ CÓ SẴN VÀ ỔN ĐỊNH thay vì tạo class `AppTest` mới — tránh
   tăng thêm số lượng `AppTest.from_file()` trong file không cần thiết.
5. **Toàn bộ code/comment/UI dùng tiếng Việt** (người dùng không đọc tiếng
   Anh trôi chảy). Giữ nguyên quy ước này cho mọi code mới.
6. **Vietstock/CafeF/dữ liệu tài chính doanh nghiệp CHƯA có nguồn** — các
   module cần EPS/ROE/D-E/CFO/P-E ngành (`stock_signal_engine.py`) hiện chạy
   ở chế độ "chỉ kỹ thuật", lớp cơ bản luôn `None` (trung tính). Đây là
   khoảng trống lớn nhất của dự án — nếu người dùng muốn mở rộng, cần tìm
   nguồn dữ liệu báo cáo tài chính trước.
7. **Dữ liệu vĩ mô Mỹ/VN (Fed Rate, CPI, tỷ giá, lãi suất liên ngân hàng, sự
   kiện địa chính trị) KHÔNG có API tự động** — người dùng nhập tay qua
   dashboard (mục "🌍 Nhập dữ liệu vĩ mô thủ công"). Claude có thể tự tra
   cứu tin tức thời sự qua web search để gợi ý cập nhật "sự kiện địa chính
   trị" khi được yêu cầu (xem `update_geopolitical_event.py` làm ví dụ).

## 3. Kiến trúc — chuỗi 4 module cốt lõi (theo đúng tài liệu kỹ thuật gốc)

```
Điểm Vĩ Mô (macro_score_engine.py)
    -> Trạng thái Thị trường (market_regime_detector.py + market_breadth.py)
    -> Phân bổ Vốn (capital_allocation_engine.py)
    -> Tín hiệu Mua/Bán từng mã (stock_signal_engine.py)
```

Mỗi module đều có **bản định lượng chi tiết (`_quant`/`_v2`) SONG SONG** với
bản định tính đơn giản ban đầu — không thay thế, để đối chiếu. Ví dụ:
`detect_market_regime()` (cũ, đơn giản) và `detect_market_regime_quant()`
(mới, đầy đủ 3 lớp) cùng tồn tại.

### Danh sách file `core/` và vai trò

| File | Vai trò |
|---|---|
| `data_collector.py` | `DataSource` (abstract) + `MockDataSource` + `VnstockDataSource`. `DataCollector` bọc ngoài, có cache + retry + rate-limit handling. |
| `indicators.py` | MA/EMA/RSI/volume MA, `resample_ohlcv()` (ngày/tuần/tháng), `get_indicator_snapshot()`. |
| `pattern_detector.py` | Phát hiện mô hình thu hẹp biên độ (narrowing amplitude). |
| `market_breadth.py` | % Breadth EMA200, ADX, ATR, Bollinger Band Width, Volume Ratio, A/D Line, **Bullish/Bearish Divergence**, MA cross, tổng hợp Lớp 3 theo nhóm (`aggregate_layer3_indicators_for_group`). |
| `market_regime_detector.py` | `detect_market_regime()` (cũ) + `detect_market_regime_quant()` (mô hình 3 lớp đầy đủ, nhận `precomputed_macro_score` tùy chọn). |
| `market_regime_ensemble.py` | Ensemble 3 phương pháp (Breadth/Peak-Trough/Markov) — `phan_tich_ensemble_theo_nhom()` (1 nhóm, ngày gần nhất) + `tinh_chuoi_ensemble_theo_ngay()` (walk-forward THEO TỪNG NGÀY cho 1 mã, dùng cho backtest lịch sử — bổ sung 25/08/2026, xem mục 4k). |
| `long_term_indicator_backtest.py` | Backtest 8 bộ chỉ số kỹ thuật (MA20/EMA Cross/RSI14/Bollinger x2/Volume Breakout/Trend Filter/Buy&Hold) cho 1 mã, bucket theo giai đoạn tại ngày vào lệnh — dùng cho mục "🧮 Cổ phiếu dài hạn" (bổ sung 25/08/2026). Tái sử dụng `backtest/backtest_engine.py::run_backtest()`, không tự viết lại engine. |
| `macro_score_engine.py` | Điểm vĩ mô 6 nhóm (Fed/CPI Mỹ/CPI VN/tỷ giá/liên NH/sự kiện) theo đúng công thức tài liệu, có cơ chế **override** khi sự kiện nghiêm trọng. |
| `manual_macro_data.py` | Quản lý chuỗi thời gian dữ liệu vĩ mô NHẬP TAY + tính các đại lượng dẫn xuất (delta, %YTD, số kỳ tăng liên tiếp, khoảng cách đỉnh) làm input cho `macro_score_engine`. |
| `capital_allocator.py` | Bản phân bổ vốn ĐƠN GIẢN (cũ, theo mã đơn lẻ). |
| `capital_allocation_engine.py` | Bản phân bổ vốn ĐẦY ĐỦ (mới): ATR14, hỗ trợ/kháng cự tự động, khoảng giá vào lệnh theo chiến lược (pullback/breakout/support), nhiều đợt giải ngân, phân bổ theo breadth ngành. |
| `stock_signal_engine.py` | Tín hiệu MUA/GIỮ/BÁN từng mã — nến đảo chiều, mẫu hình kỹ thuật, phủ quyết, bảng quyết định 5 bước, báo cáo tổng hợp `build_signal_summary_report()`. |
| `short_term_signal.py` | Tiêu chí ngắn hạn: quá mua VN-Index/cổ phiếu so với MA20, thống kê xác suất điều chỉnh (event study lịch sử), tín hiệu "bắt cá hồi" sau giảm mạnh (10-15% từ đỉnh 40 phiên, ưu tiên Ngân hàng/Chứng khoán/Thép/VN30). |
| `entry_screener.py` | Rà soát danh sách vào lệnh ngắn hạn — lớp TỔNG HỢP/LỌC trên `pattern_detector.py`+`stock_signal_engine.py`: xếp hạng theo EMA200, cờ lọc tích lũy dài hạn/sắp breakout, bộ lọc chọn tiêu chí kết hợp, phân kỳ tăng 3 điểm (độ tin cậy CAO hơn phân kỳ 2 điểm gốc). |
| `stock_character_classifier.py` | Phân loại "TÍNH CÁCH" giao dịch của TỪNG mã (đặc điểm vận động giá NỘI TẠI — chuẩn hóa theo percentile so với CHÍNH lịch sử của mã đó, không dùng ngưỡng cứng chung). Các thành phần: streak (chuỗi phiên cùng chiều), velocity (%/phiên), Choppiness Index (CHOP), closing strength, tỷ lệ đảo chiều, autocorrelation. Nhãn: DUT_KHOAT_TANG/GIAM, BUNG_NO_NGAN, LINH_XINH, TRUNG_TINH. Cảnh báo SQUAT (breakout yếu)/CHURNING (nghi phân phối ẩn). **28/07/2026: đây là LẦN THỨ 3 mới nhận được đúng bản gốc** — 2 lần trước mình tự đoán lại sai (API khác hẳn: thiếu tham số `ma`, thiếu `he_so_chiet_khau_do_tin_cay()`/`gioi_han_ty_trong_theo_tinh_cach()`). May mắn: `main.py`/`run_full_market.py`/`dashboard/app.py` (viết ở phiên trước khi bị nén tóm tắt) đã LUÔN gọi đúng theo API THẬT này — chỉ 2 file `core/stock_character_classifier.py` + `core/character_integration.py` bị mất/sai, code gọi chúng không hề sai. |
| `character_integration.py` | Tích hợp tính cách vào tín hiệu mua/bán + phân bổ vốn đã có — dùng ĐÚNG `he_so_chiet_khau_do_tin_cay()` và `gioi_han_ty_trong_theo_tinh_cach()` từ `stock_character_classifier.py` (không viết lại công thức). `quet_tinh_cach_watchlist()` quét cả watchlist, không dừng khi 1 mã lỗi. |
| `stock_character_classifier.py` | Phân loại "tính cách giao dịch" nội tại của từng mã (DỨT_KHOÁT_TĂNG/GIẢM, BÙNG_NỔ_NGẮN, LÌNH_XÌNH, TRUNG_TÍNH) — dựa trên percentile NỘI TẠI (so với chính lịch sử mã đó, không dùng ngưỡng cứng chung), gồm streak/velocity/Choppiness Index/closing strength/reversal rate/autocorrelation. Phát hiện thêm cảnh báo SQUAT (giả bứt phá) và CHURNING (nghi phân phối ẩn). **Nhận từ người dùng qua upload, không phải mình xây từ đầu** — đã bổ sung bộ test riêng (28 test) sau khi dò số liệu thật xác nhận từng hàm con. |
| `character_integration.py` | Lớp TÍCH HỢP `stock_character_classifier.py` với `stock_signal_engine.py` + `capital_allocator.py` — CHIẾT KHẤU độ tin cậy tín hiệu Breakout khi mã đang lình xình, GIẢM tỷ trọng phân bổ khi mã bùng nổ ngắn/có cảnh báo CHURNING, và `quet_tinh_cach_watchlist()` quét hàng loạt. **Người dùng đã viết sẵn `tests/test_character_integration.py`** (13 test, tự chạy pass ngay lần đầu khi mình viết module khớp đúng chữ ký). |
| `trade_journal.py` | Nhật ký giao dịch mua/bán mô phỏng (ghi tay qua dashboard) — KHÁC với `paper_portfolio.py`. |
| `paper_portfolio.py` | Theo dõi NAV/vị thế danh mục mô phỏng (bản cũ hơn, đơn giản hơn `trade_journal.py`). |
| `chart_annotations.py` | Chú thích sự kiện lên biểu đồ (vd. "Mỹ đánh Iran"). |
| `notifier.py` | Gửi cảnh báo qua Telegram (tùy chọn, đang tắt mặc định). |
| `storage.py` | SQLite key-value đơn giản, APPEND-ONLY. `get_latest(category,key)` lấy bản mới nhất, `get_history()` lấy toàn bộ lịch sử, `query_all_keys(category)` liệt kê mọi key. Không có UPDATE tại chỗ — muốn "sửa" thì `save()` lại (ghi đè hiệu quả qua get_latest). **Hỗ trợ CẢ 2 backend** (từ 27/07/2026): SQLite (mặc định, `db_path="./data/pm_ck.db"`) HOẶC PostgreSQL/Supabase (dùng chung nhiều máy, `db_path="postgresql://..."`) — tự động nhận diện qua tiền tố connection string, API công khai GIỐNG HỆT giữa 2 backend, không cần sửa code gọi.

### Entry point

- `main.py` — chạy watchlist NHỎ (`config.yaml` → `watchlist.symbols`), đầy đủ
  cả 4 module + Lớp 3 tính riêng cho từng mã (vì watchlist nhỏ, đủ tài
  nguyên tính chi tiết).
- `run_full_market.py` — quét TOÀN THỊ TRƯỜNG (~1.500+ mã qua
  `vnstock.Listing`), có checkpoint (resume được), rate-limit-aware. Lớp 3
  tính THEO NGÀNH (tổng hợp nhiều mã) ở bước riêng sau khi quét xong, không
  tính riêng lẻ như `main.py` (quá tốn thời gian với quy mô lớn).
  **QUAN TRỌNG**: nếu `config.yaml` → `watchlist.symbols` KHÔNG rỗng, dùng
  ĐÚNG danh sách mã + ngành TÙY CHỈNH đó (không gọi
  `fetch_symbol_sector_map()` quét toàn thị trường) — cho phép người dùng
  tự định nghĩa watchlist theo ngành riêng (ví dụ 212 mã theo 16 nhóm
  ngành tự phân loại), vẫn giữ nguyên toàn bộ cơ chế an toàn
  (checkpoint/chống rate-limit) dù quy mô nhỏ hơn toàn thị trường.
- `check_macro_score.py`, `update_geopolitical_event.py` — script tiện ích
  độc lập, không nằm trong pipeline chính.

## 4. Cách chạy

```bash
cd C:\projects\pm_ck
venv\Scripts\activate

pytest tests/ -v                    # LUÔN chạy sau khi sửa code — kỳ vọng: tất cả PASS
python main.py                       # watchlist nhỏ, nhanh (vài giây)
python run_full_market.py            # toàn thị trường, ~35-40 phút (Ctrl+C an toàn, resume được)
python run_full_market.py --reset    # bỏ qua checkpoint, quét lại từ đầu
streamlit run dashboard/app.py       # xem kết quả trên trình duyệt
```

**`config/config.yaml`** — `data_source.adapter` PHẢI LÀ `"vnstock"` để dùng
dữ liệu thật (`"mock"` chỉ dùng khi code/test không có mạng).

## 5. Quy ước viết code trong dự án này

- Mỗi hàm mới **BẮT BUỘC có test đi kèm**, tính tay ít nhất 1 giá trị kỳ vọng
  cụ thể (không chỉ test "không lỗi").
- Khi thêm chỉ báo/mẫu hình kỹ thuật mới, **dò số liệu thật bằng script nhỏ
  trước** để tạo đúng kịch bản test (nhiều lần viết test "lý thuyết" đã sai
  vì chỉ báo có độ trễ xác nhận — vd. swing low cần vài phiên SAU để xác
  nhận, không nhận diện được ở đúng phiên cuối cùng).
- `raise InvalidXxxError(ValueError)` riêng cho từng module khi input sai,
  kèm thông báo lỗi tiếng Việt rõ ràng, gợi ý cách sửa.
- Dashboard: mọi section mới đặt trong hàm `render_xxx_section(storage, ...)`
  riêng, gọi trong `main()` cuối file `dashboard/app.py`.
- Test dashboard dùng `AppTest` (Streamlit testing framework) + fixture
  `isolated_db_path`/`seeded_storage` trong `tests/test_dashboard.py` — LUÔN
  dùng 2 fixture này, không tự tạo đường dẫn DB khác.

## 6. Khoảng trống / hướng mở rộng còn lại

1. **Dữ liệu tài chính doanh nghiệp** (EPS, ROE, D/E, CFO, P/E ngành) — chưa
   có nguồn. `stock_signal_engine.py` và phần lọc cơ bản của
   `capital_allocation_engine`/`stock_signal_engine` đang chạy trung tính.
2. **Dữ liệu vĩ mô tự động** — Fed dot-plot, CPI Mỹ/VN, lãi suất liên ngân
   hàng VN, tỷ giá SBV đều đang NHẬP TAY. FRED API (Mỹ) có thể tự động hóa
   được (miễn phí, có API key) — SBV/GSO (VN) chưa rõ có API chính thức hay
   không, cần khảo sát thêm.
3. **Phát hiện mẫu hình nến chỉ dùng rule tự viết** (`is_bullish_engulfing`,
   `is_pin_bar`...) — đơn giản hóa so với thư viện TA-Lib đầy đủ tài liệu gốc
   đề xuất.
4. **Đếm số lần chạm vùng hỗ trợ lịch sử** (`check_support_bounce_pattern`)
   hiện CHƯA đếm số lần chạm — chỉ kiểm tra giá đang ở gần mức hỗ trợ cho
   trước. Tài liệu gốc yêu cầu "xác nhận ≥ 2 lần trước đó".
5. **Backtest toàn bộ ngưỡng số** (60/40/20/80% breadth, ADX 25/20, các hệ số
   0.3/0.5/1.0 trong `macro_score_engine`...) — tài liệu gốc yêu cầu backtest
   trên dữ liệu lịch sử 3-5 năm để hiệu chỉnh, CHƯA làm.
6. **Position sizing trong `stock_signal_engine`** chưa nối với
   `capital_allocation_engine` để lấy đúng mức cắt lỗ đã đặt khi vào lệnh
   (hiện `position_info["gia_cat_lo"]` luôn `None` khi gọi từ
   `run_stock_signal_step()` trong `main.py`).

## 7. Lịch sử phiên làm việc (tóm tắt, xem chi tiết trong lịch sử chat claude.ai nếu cần)

Dự án được xây dần qua nhiều phiên hội thoại trên claude.ai (không phải
Claude Code) — bắt đầu từ 5 module cốt lõi cơ bản, sau đó mở rộng dần theo
5 tài liệu kỹ thuật chi tiết do người dùng cung cấp (trạng thái thị trường
3 lớp, phân bổ vốn ATR14, điểm vĩ mô 6 nhóm, tín hiệu mua/bán 5 bước), cộng
thêm dashboard Streamlit đầy đủ tính năng (biểu đồ nến + MA/EMA/RSI, nhật ký
giao dịch, chú thích sự kiện, nhập vĩ mô thủ công, báo cáo tổng hợp).

*28/07/2026*: người dùng cho biết đã dùng công cụ khác (khả năng Claude
Code) viết thêm `core/stock_character_classifier.py` +
`core/character_integration.py` (phân loại "tính cách giao dịch" nội tại
từng mã — dứt khoát/bùng nổ/lình xình — điều chỉnh tín hiệu Mua/Bán và
phân bổ vốn) — đã upload lại, tích hợp vào pipeline chính, viết bộ test
riêng cho `stock_character_classifier.py` (28 test, module gốc chưa có
test kèm theo). Từ nay yêu cầu toàn bộ thay đổi dự án đi qua claude.ai,
không qua công cụ khác nữa — cần thêm `scipy>=1.11` vào dependencies.

