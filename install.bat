@echo off
REM ============================================================
REM  DataTaxonomy Windows Installer -- current recommended path.
REM  Run this once to set up and launch the app.
REM ============================================================
setlocal enabledelayedexpansion
title DataTaxonomy -- Windows Installer

echo.
echo  ================================================
echo    DataTaxonomy  ^|  Windows Installer
echo  ================================================
echo.
echo  This will set up everything automatically.
echo  Please keep this window open until finished.
echo  This package contains no project files or previous run results.
echo.

:: Root directory
set "ROOT=%~dp0"
if "!ROOT:~-1!"=="\" set "ROOT=!ROOT:~0,-1!"
cd /d "!ROOT!"

:: ------------------------------------------------
:: STEP 1: Find or install Python
:: ------------------------------------------------
echo  [1/6]  Checking for Python 3...
set "PYTHON_CMD="

:: Try system python in PATH
python --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
    echo         Found: !PYVER!
    set "PYTHON_CMD=python"
    goto :venv
)

:: Try common user-install locations
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%d\python.exe" (
        set "PYTHON_CMD=%%d\python.exe"
        set "PATH=%%d;%%d\Scripts;!PATH!"
        for /f "tokens=*" %%v in ('"%%d\python.exe" --version 2^>^&1') do set "PYVER=%%v"
        echo         Found: !PYVER!
        goto :venv
    )
)

:: Try installing via Windows Package Manager (winget)
echo         Python not found. Trying automatic install...
echo         ^(This may take a minute - please wait^)
echo.
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements >nul 2>&1

if !errorlevel! equ 0 (
    :: Refresh PATH with newly installed location
    for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%d\python.exe" (
            set "PYTHON_CMD=%%d\python.exe"
            set "PATH=%%d;%%d\Scripts;!PATH!"
            echo         Python installed successfully.
            goto :venv
        )
    )
)

:: Give up -- guide user to manual install
echo.
echo  -----------------------------------------------
echo   Python could not be installed automatically.
echo.
echo   Please download it from:
echo   https://www.python.org/downloads/
echo.
echo   IMPORTANT: During install, tick the checkbox:
echo   "Add Python 3.x to PATH"
echo.
echo   Then double-click install.bat again.
echo  -----------------------------------------------
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:: ------------------------------------------------
:: STEP 2: Virtual environment
:: ------------------------------------------------
:venv
echo.
echo  [2/6]  Setting up isolated environment...

if not exist ".venv" (
    "!PYTHON_CMD!" -m venv .venv
    if !errorlevel! neq 0 (
        echo.
        echo  [!] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo         Created: .venv\
) else (
    echo         Already exists -- skipping
)

:: ------------------------------------------------
:: STEP 3: Install packages
:: ------------------------------------------------
echo.
echo  [3/6]  Installing required packages...
echo         ^(First-time install may take 3-5 minutes^)
echo.

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if !errorlevel! neq 0 (
    echo.
    echo  [!] Package installation failed.
    echo      Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo         All packages installed.

:: ------------------------------------------------
:: STEP 4: System dependencies (Tesseract OCR + Poppler)
:: ------------------------------------------------
echo.
echo  [4/6]  Checking system dependencies (OCR + PDF rendering)...

set "SYS_WARN=0"

:: Check Tesseract -- try winget auto-install if missing
where tesseract >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%v in ('tesseract --version 2^>^&1 ^| findstr /i "tesseract"') do echo         Tesseract OCR: found (%%v)
) else (
    echo         Tesseract OCR not found -- attempting auto-install via winget...
    winget install -e --id UB-Mannheim.TesseractOCR --silent --accept-package-agreements --accept-source-agreements >nul 2>&1
    if !errorlevel! equ 0 (
        echo         Tesseract OCR installed.
        :: Refresh PATH so tesseract is findable in this session
        for /d %%d in ("%ProgramFiles%\Tesseract-OCR" "%LOCALAPPDATA%\Programs\Tesseract-OCR") do (
            if exist "%%d\tesseract.exe" set "PATH=%%d;!PATH!"
        )
    ) else (
        set "SYS_WARN=1"
        echo         [!] Could not auto-install Tesseract.
        echo             Scanned PDFs will be processed without OCR.
        echo             To enable OCR later: https://github.com/UB-Mannheim/tesseract/wiki
    )
)

:: Check Poppler -- pymupdf handles PDF rendering as fallback, so this is optional
where pdftoppm >nul 2>&1
if !errorlevel! equ 0 (
    echo         Poppler: found
) else (
    echo         Poppler: not found ^(OK -- PyMuPDF will handle PDF rendering^)
)

if "!SYS_WARN!"=="1" (
    echo.
    echo         Note: missing tools only affect scanned image-only PDFs.
    echo         Text PDFs, Office docs, and CAD files work without them.
)

:: ------------------------------------------------
:: STEP 5: API key
:: ------------------------------------------------
echo.
echo  [5/6]  Setting up API key...

:: Skip if .env already has a real key
set "SKIP_KEY=0"
if exist ".env" (
    findstr /r /c:"OPENROUTER_API_KEY=sk-" ".env" >nul 2>&1 && set "SKIP_KEY=1"
    findstr /r /c:"DEEPSEEK_API_KEY=sk-"   ".env" >nul 2>&1 && set "SKIP_KEY=1"
)

if "!SKIP_KEY!"=="1" (
    echo         .env already configured -- skipping
    goto :done
)

:: Pop a simple input dialog via PowerShell
for /f "delims=" %%k in ('powershell -NoProfile -Command "Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.Interaction]::InputBox('Paste your OpenRouter API key below.' + [char]10 + [char]10 + 'Get a free key at: https://openrouter.ai/keys', 'DataTaxonomy -- API Key Setup', '')"') do (
    set "API_KEY=%%k"
)

if "!API_KEY!"=="" (
    echo         No key entered. Edit the .env file later to add it.
    if not exist ".env" echo OPENROUTER_API_KEY=> ".env"
) else (
    echo OPENROUTER_API_KEY=!API_KEY!> ".env"
    echo         Key saved to .env
)

:: ------------------------------------------------
:: STEP 6: Create config.yaml
:: ------------------------------------------------
:done
echo.
echo  [6/6]  Setting up project config...

if exist "config.yaml" (
    echo         config.yaml already exists -- skipping
    goto :finish
)

copy config.example.yaml config.yaml >nul
echo         Created config.yaml from template.
echo         ^(You can configure your project folder inside the app.^)

:: ------------------------------------------------
:: Done
:: ------------------------------------------------
:finish
echo.
echo  ================================================
echo    Installation complete!
echo  ================================================
echo.
echo    To start DataTaxonomy, open a terminal in
echo    this folder and run:
echo.
echo        python main.py --dashboard-only
echo.
pause
endlocal
