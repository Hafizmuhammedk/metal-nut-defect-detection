@echo off
setlocal
cd /d "%~dp0.."
where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)
echo Installing downloader dependencies...
%PYTHON% -m pip install huggingface_hub Pillow
if errorlevel 1 goto failed
echo.
echo Downloading MVTec AD metal_nut into data\metal_nut...
%PYTHON% tools\download_metal_nut.py
if errorlevel 1 goto failed
echo.
echo Complete. Open notebooks\metal_nut_detection.ipynb in VS Code.
pause
exit /b 0
:failed
echo.
echo Download failed. Review the error and retry.
pause
exit /b 1
