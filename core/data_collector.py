"""
data_collector.py
==================
[Giai đoạn 1 — Nền tảng dữ liệu]

Module thu thập dữ liệu cho phần mềm theo dõi chứng khoán Việt Nam.
CHỈ ĐỌC DỮ LIỆU — không đặt lệnh giao dịch dưới bất kỳ hình thức nào.

Thiết kế theo ADAPTER PATTERN: lớp `DataSource` là interface trừu tượng,
mỗi nguồn dữ liệu cụ thể (mock, vnstock, ssi, vndirect...) kế thừa và cài
đặt lại các phương thức. `DataCollector` chỉ làm việc với interface này,
không phụ thuộc trực tiếp vào bất kỳ nguồn dữ liệu cụ thể nào — nhờ vậy có
thể đổi nguồn dữ liệu mà không cần sửa code các module khác (indicators,
pattern_detector, market_regime_detector...).

CẢNH BÁO CHI PHÍ DỮ LIỆU: nhiều nguồn dữ liệu công khai (SSI iBoard,
VNDIRECT DChart, TCBS, Fireant...) có khả năng bị tính phí theo quy định
mới (cần tự xác minh lại, không giả định chắc chắn). Vì vậy:
- Mặc định BẬT "chế độ dữ liệu trễ" (delayed_mode) trong config.
- Adapter tự khai báo `is_paid_source` — nếu True và delayed_mode=False,
  DataCollector sẽ log cảnh báo rõ ràng mỗi khi lấy dữ liệu.

Module này CHỈ trả về DataFrame/dict — không lưu trữ (việc lưu trữ do
core/storage.py đảm nhiệm ở bước khác).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger("pm_ck.data_collector")


def _doc_so_thuc(gia_tri: Any) -> Optional[float]:
    """Đọc `gia_tri` thành float, trả `None` nếu thiếu/NaN/không hợp lệ —
    dùng cho các cột TÙY CHỌN của bảng giá vnstock (không phải mọi mã/thời
    điểm đều có đủ bid/ask/khối ngoại)."""
    try:
        so = float(gia_tri)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(so) else so


# ==============================================================================
# NGOẠI LỆ (Exceptions)
# ==============================================================================

class DataSourceError(Exception):
    """Lỗi phát sinh khi gọi tới nguồn dữ liệu (network, timeout, dữ liệu rỗng...)."""


class RetryExhaustedError(DataSourceError):
    """Đã thử lại đủ số lần cấu hình nhưng vẫn thất bại."""


# ==============================================================================
# CẤU TRÚC DỮ LIỆU (Data classes)
# ==============================================================================

@dataclass
class FundamentalData:
    """Dữ liệu cơ bản doanh nghiệp tại một thời điểm."""

    symbol: str
    eps: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    dividend_yield: Optional[float] = None
    foreign_net_volume: Optional[float] = None   # khối lượng mua/bán ròng khối ngoại
    proprietary_net_volume: Optional[float] = None  # khối lượng mua/bán ròng tự doanh
    as_of: Optional[datetime] = None


@dataclass
class NewsItem:
    """Một tin tức/sự kiện liên quan tới thị trường hoặc một mã cổ phiếu."""

    title: str
    published_at: datetime
    symbol: Optional[str] = None
    category: str = "general"   # ví dụ: "agm", "dividend", "foreign_room", "general"
    content: str = ""
    url: str = ""


@dataclass
class MacroDataPoint:
    """Một điểm dữ liệu vĩ mô — nhóm dữ liệu ƯU TIÊN CAO NHẤT của hệ thống.

    `affected_sectors` gắn nhãn rõ ngành nào bị ảnh hưởng bởi sự kiện/chính
    sách này, để market_regime_detector và capital_allocator dùng lại.
    """

    category: str          # "fx_intervention" | "omo" | "interest_rate" | "sector_policy"
    description: str
    value: Optional[float] = None
    direction: Optional[str] = None   # "tightening" | "easing" | "neutral"
    affected_sectors: list[str] = field(default_factory=list)
    as_of: Optional[datetime] = None
    source_is_paid: bool = False


# ==============================================================================
# INTERFACE NGUỒN DỮ LIỆU (Adapter Pattern)
# ==============================================================================

class DataSource(ABC):
    """Interface trừu tượng cho một nguồn dữ liệu.

    Mỗi adapter cụ thể (mock, vnstock, ssi, vndirect...) PHẢI kế thừa lớp
    này và cài đặt đầy đủ các phương thức bên dưới.
    """

    #: Tên hiển thị của nguồn dữ liệu, dùng để log.
    name: str = "unknown"

    #: Đánh dấu nguồn có khả năng bị tính phí hay không — dùng để cảnh báo.
    is_paid_source: bool = False

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str = "day") -> pd.DataFrame:
        """Trả về DataFrame gồm cột: date, open, high, low, close, volume."""

    @abstractmethod
    def fetch_realtime_price(self, symbol: str) -> dict:
        """Trả về dict: {symbol, price, volume, timestamp}."""

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> FundamentalData:
        """Trả về dữ liệu cơ bản doanh nghiệp."""

    @abstractmethod
    def fetch_news(self, symbol: Optional[str] = None) -> list[NewsItem]:
        """Trả về danh sách tin tức, lọc theo mã nếu có truyền `symbol`."""

    @abstractmethod
    def fetch_macro_data(self) -> list[MacroDataPoint]:
        """Trả về danh sách các điểm dữ liệu vĩ mô hiện có."""

    def fetch_index_ohlcv(self, symbol: str, timeframe: str = "day") -> pd.DataFrame:
        """Trả về OHLCV cho MỘT CHỈ SỐ thị trường (VNINDEX, VN30, VN100...).

        KHÔNG bắt buộc override — mặc định dùng lại `fetch_ohlcv()` thông
        thường (phù hợp cho nguồn không phân biệt equity/index, ví dụ
        `MockDataSource`). Các adapter dữ liệu thật có API riêng cho chỉ
        số (ví dụ `vnstock` dùng namespace `.index()` khác `.equity()`)
        nên override lại phương thức này để gọi đúng endpoint.
        """
        return self.fetch_ohlcv(symbol, timeframe)


class MockDataSource(DataSource):
    """Adapter dữ liệu GIẢ LẬP — dùng cho phát triển/test, KHÔNG gọi API thật.

    Sinh dữ liệu xác định (deterministic) dựa trên mã cổ phiếu, để unit
    test có thể kiểm tra kết quả một cách ổn định, lặp lại được.

    Có thể mô phỏng lỗi mạng bằng `fail_times` — hữu ích để test cơ chế
    retry của DataCollector mà không cần gọi API thật.
    """

    name = "mock"
    is_paid_source = False

    def __init__(self, fail_times: int = 0):
        # Số lần gọi đầu tiên sẽ giả lập lỗi (dùng để test retry).
        self._fail_times = fail_times
        self._call_counts: dict[str, int] = {}

    def _maybe_fail(self, key: str) -> None:
        count = self._call_counts.get(key, 0)
        self._call_counts[key] = count + 1
        if count < self._fail_times:
            raise DataSourceError(f"[mock] Giả lập lỗi mạng lần {count + 1} cho '{key}'")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "day") -> pd.DataFrame:
        self._maybe_fail(f"ohlcv:{symbol}:{timeframe}")

        n_periods = 260 if timeframe == "day" else 52
        freq = "B" if timeframe == "day" else "W"
        seed = sum(ord(c) for c in symbol)
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_periods, freq=freq)

        base_price = 10.0 + (seed % 50)
        prices = []
        price = base_price
        for i in range(n_periods):
            drift = ((seed + i) % 7 - 3) * 0.05
            price = max(1.0, price + drift)
            prices.append(price)

        df = pd.DataFrame({
            "date": dates,
            "open": [round(p * 0.995, 2) for p in prices],
            "high": [round(p * 1.01, 2) for p in prices],
            "low": [round(p * 0.99, 2) for p in prices],
            "close": [round(p, 2) for p in prices],
            "volume": [1_000_000 + (seed * 137 + i * 977) % 500_000 for i in range(n_periods)],
        })
        return df

    def fetch_realtime_price(self, symbol: str) -> dict:
        self._maybe_fail(f"realtime:{symbol}")
        seed = sum(ord(c) for c in symbol)
        return {
            "symbol": symbol,
            "price": round(10.0 + (seed % 50) + (seed % 7) * 0.1, 2),
            "volume": 1_000_000 + (seed * 137) % 500_000,
            "timestamp": datetime.now(),
        }

    def fetch_fundamentals(self, symbol: str) -> FundamentalData:
        self._maybe_fail(f"fundamentals:{symbol}")
        seed = sum(ord(c) for c in symbol)
        return FundamentalData(
            symbol=symbol,
            eps=round(1000 + (seed % 5000), 0),
            pe=round(8 + (seed % 15), 2),
            pb=round(1 + (seed % 4) * 0.5, 2),
            dividend_yield=round((seed % 8) * 0.5, 2),
            foreign_net_volume=float((seed * 31) % 200_000 - 100_000),
            proprietary_net_volume=float((seed * 17) % 100_000 - 50_000),
            as_of=datetime.now(),
        )

    def fetch_news(self, symbol: Optional[str] = None) -> list[NewsItem]:
        self._maybe_fail(f"news:{symbol or 'all'}")
        target = symbol or "VNINDEX"
        return [
            NewsItem(
                title=f"[Giả lập] {target} thông báo họp Đại hội cổ đông thường niên",
                published_at=datetime.now() - timedelta(days=1),
                symbol=symbol,
                category="agm",
            ),
            NewsItem(
                title=f"[Giả lập] {target} công bố chia cổ tức bằng tiền mặt",
                published_at=datetime.now() - timedelta(days=3),
                symbol=symbol,
                category="dividend",
            ),
        ]

    def fetch_macro_data(self) -> list[MacroDataPoint]:
        self._maybe_fail("macro")
        now = datetime.now()
        return [
            MacroDataPoint(
                category="fx_intervention",
                description="[Giả lập] NHNN bán ra USD can thiệp tỷ giá",
                value=None,
                direction="tightening",
                affected_sectors=["banking", "import"],
                as_of=now,
            ),
            MacroDataPoint(
                category="omo",
                description="[Giả lập] NHNN hút ròng qua kênh thị trường mở (OMO)",
                value=-5000.0,
                direction="tightening",
                affected_sectors=[],
                as_of=now,
            ),
            MacroDataPoint(
                category="interest_rate",
                description="[Giả lập] Lãi suất huy động kỳ hạn 12 tháng tăng nhẹ",
                value=5.2,
                direction="tightening",
                affected_sectors=["banking", "real_estate"],
                as_of=now,
            ),
            MacroDataPoint(
                category="sector_policy",
                description="[Giả lập] Siết room tín dụng cho vay bất động sản",
                value=None,
                direction="tightening",
                affected_sectors=["real_estate", "construction"],
                as_of=now,
            ),
        ]


class VnstockDataSource(DataSource):
    """Adapter dữ liệu THẬT — dùng thư viện mã nguồn mở `vnstock`
    (https://github.com/thinh-vu/vnstock, phiên bản v4+ Unified UI).

    CÀI ĐẶT:
        pip install -U vnstock

    Miễn phí ở mức sử dụng cơ bản, có giới hạn số request/phút theo cấp
    độ (Khách/Cộng đồng/Tài trợ) — xem `register_user()` trong `vnstock`
    để đăng ký tài khoản miễn phí tăng giới hạn.

    GIỚI HẠN ĐÃ BIẾT (đã xác nhận qua kiểm tra thực tế cấu trúc dữ liệu):
        - `fetch_ohlcv`: hoạt động — cột trả về gồm
          ['time','open','high','low','close','volume'], được chuẩn hóa
          lại thành ['date','open','high','low','close','volume'].
        - `fetch_realtime_price`: hoạt động — lấy từ bảng giá (quote/price
          board) mà vnstock trả về, gồm `close_price`/`volume_accumulated`
          + các trường bổ sung (giá tham chiếu/trần/sàn, bid/ask 1, khối
          ngoại còn lại). Trước giờ khớp lệnh (close_price=0) tự động dùng
          giá THAM CHIẾU thay thế, đánh dấu qua `da_khop_lenh=False` — xem
          chi tiết ngay trong thân hàm bên dưới. LƯU Ý ĐƠN VỊ: khác với
          `fetch_ohlcv` (giá đã chia 1000, VD "25.4"), giá từ `quote()` là
          FULL VND chưa chia 1000 (VD "20800") — đã xác nhận qua kiểm tra
          cấu trúc thực tế (`market.quote()` trả `close_price=20800` cho
          HPG khi giá đang ở vùng ~20.800đ).
        - `fetch_fundamentals` / `fetch_news`: CHƯA triển khai — cấu trúc
          API tương ứng của vnstock chưa được xác minh đầy đủ. Gọi 2 hàm
          này sẽ raise `NotImplementedError` kèm hướng dẫn rõ ràng.
        - `fetch_macro_data`: vnstock KHÔNG cung cấp dữ liệu vĩ mô định
          tính (động thái OMO, lãi suất chính sách, định hướng ngành...)
          — đây vốn là dữ liệu chính sách, không phải dữ liệu thị trường.
          Hàm này trả về DANH SÁCH RỖNG kèm cảnh báo log, để không làm
          gãy pipeline — cần bổ sung nguồn khác (tin tức NHNN, thông cáo
          chính sách...) hoặc nhập thủ công cho nhóm dữ liệu này.
    """

    name = "vnstock"
    is_paid_source = False  # miễn phí ở mức cơ bản, có giới hạn request/phút

    def __init__(
        self, interval_map: Optional[dict] = None, ohlcv_count: int = 800,
        start_date: Optional[str] = None, backfill_count: int = 2000,
    ):
        self._interval_map = interval_map or {"day": "1D", "week": "1W"}
        self._ohlcv_count = ohlcv_count
        # `start_date` (bổ sung 04/08/2026): lấy dữ liệu THEO NGÀY BẮT ĐẦU
        # (VD "2021-01-01") thay vì chỉ theo `count` (số phiên gần nhất).
        # Đã kiểm tra chữ ký API thật của vnstock trước khi dùng:
        #   ohlcv(start=None, end=None, interval='1D', count=100, source='kbs', **kwargs)
        # ĐÃ KIỂM CHỨNG THỰC TẾ (04/08/2026, script debug_index_ohlcv.py):
        # CHỈ truyền `start=` mà KHÔNG kèm `count=` LỚN thì vnstock ÂM
        # THẦM BỎ QUA `start`, chỉ trả về đúng 100 dòng gần nhất (giá trị
        # count mặc định của chính vnstock) — dù cho cả cổ phiếu THƯỜNG
        # lẫn CHỈ SỐ. Phải truyền CẢ HAI `start=` và `count=` (đủ lớn)
        # CÙNG LÚC thì `start` mới thực sự có tác dụng — xem `_backfill_count`.
        self._start_date = start_date
        self._backfill_count = backfill_count  # dùng CÙNG với start_date, xem ghi chú trên

    def _get_market(self):
        try:
            from vnstock import Market
        except ImportError as exc:
            raise DataSourceError(
                "Chưa cài đặt thư viện 'vnstock'. Chạy: pip install -U vnstock"
            ) from exc
        return Market()

    def fetch_ohlcv(self, symbol: str, timeframe: str = "day") -> pd.DataFrame:
        market = self._get_market()
        interval = self._interval_map.get(timeframe, "1D")

        try:
            if self._start_date:
                # Lấy theo NGÀY BẮT ĐẦU (dùng khi cần lịch sử dài, VD
                # backfill từ 2021). QUAN TRỌNG — đã kiểm chứng thực tế
                # (04/08/2026): CHỈ truyền `start=` mà KHÔNG kèm `count=`
                # lớn thì vnstock ÂM THẦM BỎ QUA `start`, chỉ trả về đúng
                # 100 dòng gần nhất (giá trị count mặc định của vnstock)
                # — phải truyền CẢ HAI cùng lúc thì `start` mới thực sự có
                # hiệu lực. `end=None` để vnstock tự lấy tới ngày gần nhất.
                raw_df = market.equity(symbol).ohlcv(
                    start=self._start_date, interval=interval, count=self._backfill_count,
                )
            else:
                raw_df = market.equity(symbol).ohlcv(interval=interval, count=self._ohlcv_count)
        except Exception as exc:  # noqa: BLE001 — bọc mọi lỗi thành DataSourceError thống nhất
            raise DataSourceError(
                f"Lỗi khi lấy OHLCV từ vnstock cho '{symbol}': {exc}"
            ) from exc

        required = {"time", "open", "high", "low", "close", "volume"}
        missing = required - set(raw_df.columns)
        if missing:
            raise DataSourceError(
                f"Cấu trúc dữ liệu OHLCV từ vnstock đã thay đổi, thiếu cột "
                f"{sorted(missing)}. Cần cập nhật lại VnstockDataSource.fetch_ohlcv()."
            )

        df = raw_df.rename(columns={"time": "date"})[
            ["date", "open", "high", "low", "close", "volume"]
        ].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)

    def fetch_realtime_price(self, symbol: str) -> dict:
        market = self._get_market()

        try:
            quote_df = market.quote(symbol)
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(
                f"Lỗi khi lấy giá hiện tại từ vnstock cho '{symbol}': {exc}"
            ) from exc

        if quote_df is None or len(quote_df) == 0:
            raise DataSourceError(f"vnstock không trả về dữ liệu giá cho '{symbol}'.")

        # ĐÃ XÁC NHẬN THỰC TẾ (27/08/2026, gọi lặp lại nhiều lần liên tiếp):
        # `market.quote()` KHÔNG ổn định về số cột trả về — có lúc trả ĐẦY
        # ĐỦ bảng giá (close_price/volume_accumulated/bid/ask/time...), có
        # lúc CHỈ trả bộ cột tối giản (symbol/exchange/ceiling_price/
        # floor_price/reference_price/foreign_room, THIẾU HẲN close_price/
        # volume_accumulated/time) — không phải do đổi phiên bản thư viện,
        # mà dao động NGAY GIỮA 2 lần gọi cách nhau vài giây. Vì vậy KHÔNG
        # được coi close_price/volume_accumulated/time là bắt buộc (khác
        # với fetch_ohlcv ở trên) — chỉ bắt buộc có ÍT NHẤT 1 cột giá dùng
        # được (close_price HOẶC reference_price), mọi cột khác đọc qua
        # `.get()`, thiếu thì trả None thay vì raise lỗi.
        if not ({"close_price", "reference_price"} & set(quote_df.columns)):
            raise DataSourceError(
                f"vnstock không trả về cột giá nào (close_price/reference_price) "
                f"cho '{symbol}' — cấu trúc bảng giá: {sorted(quote_df.columns)}."
            )

        row = quote_df.iloc[0]
        try:
            timestamp = datetime.fromtimestamp(float(row.get("time")) / 1000.0)
        except (TypeError, ValueError):
            # Không có cột "time" (bảng giá tối giản) hoặc giá trị không
            # đọc được -> dùng thời điểm gọi API làm mốc thay thế.
            timestamp = datetime.now()

        close_price = _doc_so_thuc(row.get("close_price")) or 0.0
        gia_tham_chieu = _doc_so_thuc(row.get("reference_price"))
        # Trước giờ khớp lệnh (chưa qua ATO), mã bị đình chỉ/không có giao
        # dịch trong phiên, HOẶC bảng giá trả về ở dạng tối giản (không có
        # cột close_price) — dùng giá THAM CHIẾU thay thế để không hiển
        # thị "giá 0đ" gây hiểu lầm. `da_khop_lenh=False` đánh dấu rõ đây
        # KHÔNG phải giá đã khớp thật trong phiên.
        da_khop_lenh = close_price > 0
        gia_hien_thi = close_price if da_khop_lenh else (
            gia_tham_chieu if gia_tham_chieu is not None else None
        )
        if gia_hien_thi is None:
            raise DataSourceError(
                f"vnstock không trả về giá nào dùng được cho '{symbol}' "
                f"(close_price=0 và reference_price cũng thiếu)."
            )

        # Tự tính % thay đổi từ price/reference_price (cùng đơn vị thô từ
        # CÙNG 1 dòng dữ liệu) thay vì tin thẳng cột `percent_change` của
        # vnstock — tránh rủi ro sai lệch đơn vị (đã có tiền lệ cấu trúc dữ
        # liệu vnstock không ổn định — xem ghi chú trên).
        if da_khop_lenh and gia_tham_chieu:
            phan_tram_thay_doi = (close_price - gia_tham_chieu) / gia_tham_chieu * 100
        else:
            phan_tram_thay_doi = _doc_so_thuc(row.get("percent_change"))

        return {
            "symbol": symbol,
            "price": gia_hien_thi,
            "volume": _doc_so_thuc(row.get("volume_accumulated")) or 0.0,
            "timestamp": timestamp,
            "da_khop_lenh": da_khop_lenh,
            # False khi vnstock chỉ trả bảng giá tối giản (thiếu close_price/
            # volume_accumulated/bid/ask) — dashboard dùng để cảnh báo dữ
            # liệu đang bị giới hạn, không phải lỗi thao tác của người dùng.
            "du_lieu_day_du": "close_price" in quote_df.columns and "volume_accumulated" in quote_df.columns,
            "gia_tham_chieu": gia_tham_chieu,
            "gia_tran": _doc_so_thuc(row.get("ceiling_price")),
            "gia_san": _doc_so_thuc(row.get("floor_price")),
            "phan_tram_thay_doi": phan_tram_thay_doi,
            "gia_mua_1": _doc_so_thuc(row.get("bid_price_1")),
            "khoi_luong_mua_1": _doc_so_thuc(row.get("bid_vol_1")),
            "gia_ban_1": _doc_so_thuc(row.get("ask_price_1")),
            "khoi_luong_ban_1": _doc_so_thuc(row.get("ask_vol_1")),
            "khoi_ngoai_con_lai": _doc_so_thuc(row.get("foreign_room")),
        }

    def fetch_fundamentals(self, symbol: str) -> FundamentalData:
        raise NotImplementedError(
            "VnstockDataSource.fetch_fundamentals() chưa triển khai — cần xác "
            "minh thêm cấu trúc API 'Fundamental'/'Finance' của vnstock trước "
            "khi hoàn thiện hàm này."
        )

    def fetch_news(self, symbol: Optional[str] = None) -> list[NewsItem]:
        raise NotImplementedError(
            "VnstockDataSource.fetch_news() chưa triển khai — cần xác minh "
            "thêm API tin tức/sự kiện tương ứng của vnstock trước khi hoàn "
            "thiện hàm này."
        )

    def fetch_macro_data(self) -> list[MacroDataPoint]:
        logger.warning(
            "vnstock không cung cấp dữ liệu vĩ mô định tính (OMO, lãi suất "
            "chính sách, định hướng ngành...) — trả về danh sách rỗng. Cần "
            "bổ sung nguồn khác (tin tức NHNN, thông cáo chính sách...) hoặc "
            "nhập thủ công cho nhóm dữ liệu này."
        )
        return []

    def fetch_index_ohlcv(self, symbol: str, timeframe: str = "day") -> pd.DataFrame:
        """Lấy OHLCV cho MỘT CHỈ SỐ thị trường (VNINDEX, VN30, VN100...)
        qua namespace `.index()` riêng của vnstock — KHÁC với `.equity()`
        dùng cho cổ phiếu thường (`fetch_ohlcv()` ở trên).
        """
        market = self._get_market()
        interval = self._interval_map.get(timeframe, "1D")

        try:
            if self._start_date:
                raw_df = market.index(symbol).ohlcv(
                    start=self._start_date, interval=interval, count=self._backfill_count,
                )
            else:
                raw_df = market.index(symbol).ohlcv(interval=interval, count=self._ohlcv_count)
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(
                f"Lỗi khi lấy dữ liệu chỉ số từ vnstock cho '{symbol}': {exc}"
            ) from exc

        required = {"time", "open", "high", "low", "close", "volume"}
        missing = required - set(raw_df.columns)
        if missing:
            raise DataSourceError(
                f"Cấu trúc dữ liệu chỉ số từ vnstock đã thay đổi, thiếu cột "
                f"{sorted(missing)}. Cần cập nhật lại VnstockDataSource.fetch_index_ohlcv()."
            )

        df = raw_df.rename(columns={"time": "date"})[
            ["date", "open", "high", "low", "close", "volume"]
        ].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)

    def fetch_symbol_sector_map(self) -> dict[str, str]:
        """Lấy TOÀN BỘ danh sách mã cổ phiếu kèm ngành (phân loại ICB) từ
        vnstock — dùng để tự động xây watchlist thay vì gõ tay từng mã
        trong config.yaml. KHÔNG phải phương thức bắt buộc của interface
        `DataSource` (chỉ có ở adapter vnstock), vì đây là tính năng đặc
        thù của nguồn dữ liệu này.

        Trả về dict {symbol: industry_name}, ví dụ {"HPG": "Tài nguyên Cơ bản"}.
        """
        try:
            from vnstock import Listing
        except ImportError as exc:
            raise DataSourceError(
                "Chưa cài đặt thư viện 'vnstock'. Chạy: pip install -U vnstock"
            ) from exc

        try:
            listing = Listing(source="KBS")
            df = listing.symbols_by_industries()
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(
                f"Lỗi khi lấy danh sách mã theo ngành từ vnstock: {exc}"
            ) from exc

        required = {"symbol", "industry_name"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceError(
                f"Cấu trúc dữ liệu ngành từ vnstock đã thay đổi, thiếu cột "
                f"{sorted(missing)}. Cần cập nhật lại fetch_symbol_sector_map()."
            )

        return dict(zip(df["symbol"], df["industry_name"]))


class BinanceDataSource(DataSource):
    """Adapter dữ liệu THẬT cho tài sản quốc tế qua Binance Public API
    (https://api.binance.com/api/v3/klines) — dùng cho module rà soát mô
    hình co hẹp biên độ (`core/volatility_contraction_scanner.py`), áp
    dụng cho XAUUSD và BTC/USD.

    ÁNH XẠ TÊN GỌI THÂN THIỆN -> CẶP GIAO DỊCH BINANCE THẬT (`SYMBOL_MAP`):
        - "XAUUSD" -> "PAXGUSDT" (PAX Gold — token bảo chứng 1:1 bằng vàng
          vật chất, giá bám rất sát XAUUSD giao ngay quốc tế — xem lý do
          chọn hướng này trong docstring module `volatility_contraction_scanner`).
        - "BTCUSD" -> "BTCUSDT" (chênh lệch USDT/USD thực tế thường <0,1%).

    LƯU Ý QUAN TRỌNG VỀ `timeframe`: khác với `VnstockDataSource` ở trên
    (dùng từ vựng chuẩn hóa "day"/"week" của dự án), adapter này nhận
    THẲNG mã khung thời gian gốc của Binance (chữ thường: "1h", "4h",
    "1d", "1w"...) — vì module gọi adapter này (volatility_contraction_scanner)
    cần TỰ THỬ NHIỀU khung thời gian khác nhau theo đúng từ vựng Binance,
    không đi qua tầng chuẩn hóa "day"/"week" dùng chung cho cổ phiếu VN.

    KHÔNG cần API key cho endpoint `klines` (dữ liệu thị trường công khai).
    Giới hạn rate limit: 1200 request/phút/IP (weight-based) — với tần
    suất rà soát định kỳ (vài lần/ngày) hoàn toàn không đáng lo, nhưng vẫn
    nên giãn cách nhỏ nếu quét nhiều symbol/khung thời gian liên tiếp.

    GIỚI HẠN ĐÃ BIẾT:
        - `fetch_fundamentals` / `fetch_news` / `fetch_macro_data`: KHÔNG
          áp dụng được cho vàng/Bitcoin theo đúng nghĩa "dữ liệu cơ bản
          doanh nghiệp" — trả về giá trị RỖNG/mặc định để không làm gãy
          pipeline (nếu lỡ được gọi từ nơi khác), không phải lỗi.
        - PAXG là tài sản giao dịch 24/7 trên Binance, trong khi XAUUSD
          giao ngay có giờ nghỉ cuối tuần theo thị trường liên ngân hàng
          London/New York — có thể có chênh lệch nhỏ (basis) vào các thời
          điểm thị trường vàng truyền thống đóng cửa. Chấp nhận được cho
          mục đích rà soát mẫu hình kỹ thuật, không dùng làm giá giao dịch
          thực tế chính xác tuyệt đối.
    """

    name = "binance"
    is_paid_source = False

    BASE_URL = "https://api.binance.com/api/v3/klines"
    SYMBOL_MAP = {
        "XAUUSD": "PAXGUSDT",
        "BTCUSD": "BTCUSDT",
    }

    def __init__(self, max_attempts: int = 3, backoff_seconds: float = 2.0, timeout_seconds: float = 10.0):
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._timeout_seconds = timeout_seconds

    def _resolve_symbol(self, symbol: str) -> str:
        return self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        import requests

        binance_symbol = self._resolve_symbol(symbol)

        def _goi_api() -> pd.DataFrame:
            try:
                resp = requests.get(
                    self.BASE_URL,
                    params={"symbol": binance_symbol, "interval": timeframe, "limit": 1000},
                    timeout=self._timeout_seconds,
                )
                resp.raise_for_status()
                raw = resp.json()
            except Exception as exc:  # noqa: BLE001 — bọc mọi lỗi thành DataSourceError thống nhất
                raise DataSourceError(
                    f"Lỗi khi lấy OHLCV từ Binance cho '{symbol}' ({binance_symbol}): {exc}"
                ) from exc

            if not isinstance(raw, list) or not raw:
                raise DataSourceError(
                    f"Binance trả về dữ liệu rỗng/không hợp lệ cho '{binance_symbol}' "
                    f"(interval={timeframe}) — kiểm tra lại tên symbol/khung thời gian."
                )

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "n_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ])
            df["date"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

        return _call_with_retry(_goi_api, self._max_attempts, self._backoff_seconds)

    def fetch_realtime_price(self, symbol: str) -> dict:
        import requests

        binance_symbol = self._resolve_symbol(symbol)

        def _goi_api() -> dict:
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": binance_symbol},
                    timeout=self._timeout_seconds,
                )
                resp.raise_for_status()
                raw = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise DataSourceError(
                    f"Lỗi khi lấy giá thời gian thực từ Binance cho '{symbol}': {exc}"
                ) from exc

            return {
                "symbol": symbol,
                "price": float(raw["price"]),
                "volume": None,
                "timestamp": datetime.now(),
            }

        return _call_with_retry(_goi_api, self._max_attempts, self._backoff_seconds)

    def fetch_fundamentals(self, symbol: str) -> FundamentalData:
        # KHÔNG áp dụng cho vàng/Bitcoin — trả về cấu trúc rỗng hợp lệ,
        # không phải lỗi, để không làm gãy pipeline nếu bị gọi nhầm.
        return FundamentalData(symbol=symbol)

    def fetch_news(self, symbol: Optional[str] = None) -> list[NewsItem]:
        return []

    def fetch_macro_data(self) -> list[MacroDataPoint]:
        return []


# ==============================================================================
# TIỆN ÍCH: RETRY
# ==============================================================================

def _call_with_retry(
    func: Callable[[], Any],
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Gọi `func()`, thử lại tối đa `max_attempts` lần nếu gặp DataSourceError.

    Dùng backoff tuyến tính đơn giản (backoff_seconds * lần thử) — đủ dùng
    cho quy mô polling của dự án này, không cần backoff mũ phức tạp.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except DataSourceError as exc:
            last_error = exc
            logger.warning(
                "Lần thử %d/%d thất bại: %s", attempt, max_attempts, exc
            )
            if attempt < max_attempts:
                sleep_fn(backoff_seconds * attempt)
    raise RetryExhaustedError(
        f"Đã thử {max_attempts} lần nhưng vẫn thất bại. Lỗi cuối: {last_error}"
    ) from last_error


# ==============================================================================
# CACHE ĐƠN GIẢN TRONG BỘ NHỚ (in-memory TTL cache)
# ==============================================================================

class _OHLCVCache:
    """Cache OHLCV trong bộ nhớ, có TTL (thời gian sống) tính bằng giây.

    Thiết kế tối giản cho giai đoạn đầu — đủ để tránh gọi lại API không
    cần thiết trong cùng một phiên chạy chương trình. Việc cache bền vững
    (ghi ra đĩa) có thể bổ sung sau qua core/storage.py.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}::{timeframe}"

    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        key = self._key(symbol, timeframe)
        entry = self._store.get(key)
        if entry is None:
            return None
        cached_at, df = entry
        if time.time() - cached_at > self.ttl_seconds:
            del self._store[key]
            return None
        return df.copy()

    def set(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self._store[self._key(symbol, timeframe)] = (time.time(), df.copy())

    def clear(self) -> None:
        self._store.clear()


# ==============================================================================
# DATA COLLECTOR — điều phối chính
# ==============================================================================

DEFAULT_CONFIG: dict = {
    "delayed_mode": True,
    "warn_if_paid_source": True,
    "polling_interval_minutes": 5,
    "trading_hours": {
        "morning": ["09:00", "11:30"],
        "afternoon": ["13:00", "15:00"],
    },
    "cache": {"enabled": True},
    "retry": {"max_attempts": 3, "backoff_seconds": 2},
    "stale_data_alert_minutes": 15,
}


class DataCollector:
    """Điều phối việc thu thập dữ liệu thông qua một `DataSource` cụ thể.

    Đây là điểm truy cập DUY NHẤT mà các module khác (indicators,
    pattern_detector, market_regime_detector...) nên dùng để lấy dữ liệu —
    không gọi thẳng vào `DataSource` để giữ đúng nguyên tắc adapter pattern.
    """

    def __init__(self, source: DataSource, config: Optional[dict] = None):
        self.source = source
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        retry_cfg = self.config.get("retry", {})
        self._max_attempts = retry_cfg.get("max_attempts", 3)
        self._backoff_seconds = retry_cfg.get("backoff_seconds", 2)

        cache_cfg = self.config.get("cache", {})
        self._cache_enabled = cache_cfg.get("enabled", True)
        self._cache = _OHLCVCache(ttl_seconds=cache_cfg.get("ttl_seconds", 300))

        self._last_update_at: dict[str, datetime] = {}

        self._warn_if_paid_source()

    # --------------------------------------------------------------------
    # Cảnh báo nguồn dữ liệu trả phí
    # --------------------------------------------------------------------
    def _warn_if_paid_source(self) -> None:
        if not self.config.get("warn_if_paid_source", True):
            return
        delayed_mode = self.config.get("delayed_mode", True)
        if self.source.is_paid_source and not delayed_mode:
            logger.warning(
                "⚠️  Nguồn dữ liệu '%s' có khả năng bị TÍNH PHÍ và đang chạy ở "
                "chế độ REAL-TIME (delayed_mode=False). Hãy xác nhận lại chi "
                "phí trước khi tiếp tục sử dụng.",
                self.source.name,
            )
        elif self.source.is_paid_source and delayed_mode:
            logger.info(
                "Nguồn dữ liệu '%s' có khả năng bị tính phí, nhưng đang chạy ở "
                "chế độ dữ liệu trễ (delayed_mode=True) nên an toàn hơn về chi phí.",
                self.source.name,
            )

    # --------------------------------------------------------------------
    # Kiểm tra dữ liệu bị gián đoạn (stale data)
    # --------------------------------------------------------------------
    def _mark_updated(self, key: str) -> None:
        self._last_update_at[key] = datetime.now()

    def check_stale(self, key: str) -> bool:
        """Trả về True nếu dữ liệu ứng với `key` chưa cập nhật quá ngưỡng
        cấu hình `stale_data_alert_minutes`. Ghi log cảnh báo nếu có.
        """
        last = self._last_update_at.get(key)
        if last is None:
            return False
        threshold = timedelta(minutes=self.config.get("stale_data_alert_minutes", 15))
        is_stale = datetime.now() - last > threshold
        if is_stale:
            logger.warning(
                "Dữ liệu '%s' không được cập nhật trong hơn %s phút (lần cập nhật "
                "gần nhất: %s).",
                key, self.config.get("stale_data_alert_minutes", 15), last,
            )
        return is_stale

    # --------------------------------------------------------------------
    # Giờ giao dịch
    # --------------------------------------------------------------------
    def is_trading_hours(self, now: Optional[datetime] = None) -> bool:
        """Kiểm tra thời điểm `now` (mặc định là hiện tại) có nằm trong giờ
        giao dịch (9h00-11h30 và 13h00-15h00, các ngày làm việc) hay không.
        """
        now = now or datetime.now()
        if now.weekday() >= 5:  # Thứ 7, Chủ nhật
            return False

        trading_hours = self.config.get("trading_hours", {})
        current_time = now.time()

        for _, (start_str, end_str) in trading_hours.items():
            start = dtime.fromisoformat(start_str)
            end = dtime.fromisoformat(end_str)
            if start <= current_time <= end:
                return True
        return False

    # --------------------------------------------------------------------
    # OHLCV (có cache + retry)
    # --------------------------------------------------------------------
    def get_ohlcv(
        self, symbol: str, timeframe: str = "day", use_cache: bool = True
    ) -> pd.DataFrame:
        """Lấy dữ liệu giá lịch sử OHLCV cho một mã + khung thời gian.

        Ưu tiên đọc từ cache trong bộ nhớ nếu còn hiệu lực (TTL), tránh
        gọi lại API không cần thiết. Tự động retry theo cấu hình khi gặp
        lỗi từ nguồn dữ liệu.
        """
        if use_cache and self._cache_enabled:
            cached = self._cache.get(symbol, timeframe)
            if cached is not None:
                logger.debug("Lấy OHLCV cho %s (%s) từ cache.", symbol, timeframe)
                return cached

        df = _call_with_retry(
            lambda: self.source.fetch_ohlcv(symbol, timeframe),
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff_seconds,
        )

        if self._cache_enabled:
            self._cache.set(symbol, timeframe, df)

        self._mark_updated(f"ohlcv:{symbol}:{timeframe}")
        return df

    def get_index_ohlcv(
        self, symbol: str, timeframe: str = "day", use_cache: bool = True
    ) -> pd.DataFrame:
        """Lấy dữ liệu lịch sử OHLCV cho MỘT CHỈ SỐ thị trường (VNINDEX,
        VN30, VN100...) — tương tự `get_ohlcv()` nhưng dùng đúng endpoint
        chỉ số riêng của nguồn dữ liệu (xem `DataSource.fetch_index_ohlcv`).
        """
        cache_key = f"IDX:{symbol}"
        if use_cache and self._cache_enabled:
            cached = self._cache.get(cache_key, timeframe)
            if cached is not None:
                logger.debug("Lấy OHLCV chỉ số %s (%s) từ cache.", symbol, timeframe)
                return cached

        df = _call_with_retry(
            lambda: self.source.fetch_index_ohlcv(symbol, timeframe),
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff_seconds,
        )

        if self._cache_enabled:
            self._cache.set(cache_key, timeframe, df)

        self._mark_updated(f"index_ohlcv:{symbol}:{timeframe}")
        return df

    # --------------------------------------------------------------------
    # Giá thời gian thực / theo chu kỳ
    # --------------------------------------------------------------------
    def get_realtime_price(self, symbol: str) -> dict:
        """Lấy giá hiện tại của một mã. Ghi log nếu được gọi ngoài giờ
        giao dịch (không chặn, chỉ cảnh báo — dữ liệu có thể không đổi).
        """
        if not self.is_trading_hours():
            logger.info(
                "get_realtime_price('%s') được gọi ngoài giờ giao dịch.", symbol
            )

        result = _call_with_retry(
            lambda: self.source.fetch_realtime_price(symbol),
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff_seconds,
        )
        self._mark_updated(f"realtime:{symbol}")
        return result

    # --------------------------------------------------------------------
    # Dữ liệu cơ bản doanh nghiệp
    # --------------------------------------------------------------------
    def get_fundamentals(self, symbol: str) -> FundamentalData:
        result = _call_with_retry(
            lambda: self.source.fetch_fundamentals(symbol),
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff_seconds,
        )
        self._mark_updated(f"fundamentals:{symbol}")
        return result

    # --------------------------------------------------------------------
    # Tin tức / sự kiện
    # --------------------------------------------------------------------
    def get_news(self, symbol: Optional[str] = None) -> list[NewsItem]:
        result = _call_with_retry(
            lambda: self.source.fetch_news(symbol),
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff_seconds,
        )
        self._mark_updated(f"news:{symbol or 'all'}")
        return result

    # --------------------------------------------------------------------
    # Dữ liệu vĩ mô — ƯU TIÊN CAO NHẤT của toàn hệ thống
    # --------------------------------------------------------------------
    def get_macro_data(self) -> list[MacroDataPoint]:
        """Lấy toàn bộ dữ liệu vĩ mô hiện có (tỷ giá, OMO, lãi suất, chính
        sách ngành). Đây là nhóm dữ liệu quan trọng nhất, được
        `market_regime_detector` dùng làm lớp lọc đầu tiên trước khi xét
        đến chỉ báo kỹ thuật.
        """
        result = _call_with_retry(
            lambda: self.source.fetch_macro_data(),
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff_seconds,
        )
        self._mark_updated("macro")
        return result

    def get_macro_data_by_sector(self, sector: str) -> list[MacroDataPoint]:
        """Tiện ích: lọc dữ liệu vĩ mô theo ngành bị ảnh hưởng — dùng trực
        tiếp bởi capital_allocator khi cần kiểm tra một ngành cụ thể có
        đang bị gắn cờ thận trọng hay không.
        """
        all_macro = self.get_macro_data()
        return [m for m in all_macro if sector in m.affected_sectors]
