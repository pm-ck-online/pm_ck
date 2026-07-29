@echo off
REM ==============================================================================
REM run_full_market_update.bat
REM Chạy run_full_market.py để cập nhật dữ liệu TOÀN BỘ thị trường (~1.500+ mã).
REM LƯU Ý: mất khoảng 30-45 phút — không nên lên lịch chạy quá thường xuyên
REM (khuyến nghị: 1 lần/ngày sau giờ đóng cửa, hoặc 1 lần/tuần).
REM ==============================================================================

cd /d C:\projects\pm_ck
call venv\Scripts\activate.bat
if not exist logs mkdir logs

echo ===============================================
echo Bat dau cap nhat du lieu toan thi truong - %date% %time%
echo ===============================================

python run_full_market.py --reset >> logs\full_market_update.log 2>&1

echo ===============================================
echo Hoan tat - %date% %time%
echo ===============================================
