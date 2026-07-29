@echo off
REM ==============================================================================
REM run_main_update.bat
REM Chạy main.py để cập nhật dữ liệu cho watchlist (HPG, VNM, FPT...)
REM Dùng file này để lên lịch chạy tự động qua Windows Task Scheduler.
REM ==============================================================================

cd /d C:\projects\pm_ck
call venv\Scripts\activate.bat
if not exist logs mkdir logs

echo ===============================================
echo Bat dau cap nhat du lieu watchlist - %date% %time%
echo ===============================================

python main.py >> logs\main_update.log 2>&1

echo ===============================================
echo Hoan tat - %date% %time%
echo ===============================================
