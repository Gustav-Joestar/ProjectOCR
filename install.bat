@echo off
chcp 65001 > nul

echo =====================================
echo ProjectOCR installation
echo =====================================

cd /d "%~dp0"


echo.
echo Проверка Python...

python --version

if errorlevel 1 (
    echo.
    echo ERROR: Python не найден
    pause
    exit /b 1
)


if not exist "venv" (

    echo.
    echo Создание виртуального окружения...

    python -m venv venv

    if errorlevel 1 (
        echo ERROR: Не удалось создать venv
        pause
        exit /b 1
    )

) else (

    echo.
    echo Виртуальное окружение уже существует

)


echo.
echo Активация venv...

call venv\Scripts\activate


echo.
echo Обновление pip...

python -m pip install --upgrade pip


if not exist "requirements.txt" (

    echo.
    echo ERROR: requirements.txt не найден
    pause
    exit /b 1

)


echo.
echo Установка зависимостей...

pip install -r requirements.txt


if errorlevel 1 (

    echo.
    echo ERROR: Ошибка установки зависимостей
    pause
    exit /b 1

)


echo.
echo =====================================
echo Установка завершена успешно
echo =====================================

pause