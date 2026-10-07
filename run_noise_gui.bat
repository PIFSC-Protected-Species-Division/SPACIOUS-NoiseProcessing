@echo off
setlocal

set "REPO=%~dp0"
set "REPO=%REPO:~0,-1%"
set "PYTHON=C:\Users\kaity\anaconda3\envs\PropagationPython3_12_11\python.exe"
set "NOISE_PROCESSING_REPO=%REPO%"

if not exist "%PYTHON%" (
    echo Could not find Python environment at:
    echo   %PYTHON%
    echo.
    echo Update the PYTHON path in this launcher or install the environment.
    pause
    exit /b 1
)

rem Ensure the main dependency used by noiseProcessGoogleCloud.py is available.
"%PYTHON%" -m pip install seaborn --quiet

cd /d "%REPO%"
"%PYTHON%" "%REPO%\NoiseProcessing\ExampleApplications\NoiseProcessingGUI.py"

if errorlevel 1 (
    echo.
    echo The GUI exited with an error.
    pause
)
