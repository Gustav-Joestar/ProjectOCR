@echo off
chcp 65001 > nul

echo =====================================
echo ProjectOCR start
echo =====================================

cd /d "%~dp0"


if not exist "venv" (

    echo.
    echo ERROR: venv не найден
    echo Сначала запустите install.bat
    pause
    exit /b 1

)


echo.
echo Активация окружения...

call venv\Scripts\activate


echo.
echo Запуск OCR pipeline...
echo.


python main.py


echo.
echo =====================================
echo Работа завершена
echo =====================================

pause