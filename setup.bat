@echo off
echo =========================================
echo  NI cDAQ Python Environment Setup
echo =========================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.10+ and enable "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo Python detected:
python --version
echo.

echo Upgrading pip tools...
python -m pip install --upgrade pip setuptools wheel
IF ERRORLEVEL 1 goto error

echo Installing required Python packages...
python -m pip install nidaqmx
python -m pip install pyserial
IF ERRORLEVEL 1 goto error

echo.
echo =========================================
echo Setup complete.
echo You can now run the NI GUI script.
echo =========================================
pause
exit /b 0

:error
echo.
echo =========================================
echo ERROR during setup.
echo Check internet connection and Python install.
echo =========================================
pause
exit /b 1
