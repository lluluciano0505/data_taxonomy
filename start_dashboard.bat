@echo off
setlocal EnableExtensions EnableDelayedExpansion
title DataTaxonomy Dashboard

REM Always run from this script's folder, including desktop shortcuts.
set "ROOT=%~dp0"
if "!ROOT:~-1!"=="\" set "ROOT=!ROOT:~0,-1!"
set "DT_ROOT=!ROOT!"
cd /d "!ROOT!"

if not exist ".venv\Scripts\python.exe" (
    echo DataTaxonomy is not installed yet.
    echo Running the one-time installer. This may take several minutes...
    echo.
    set "INSTALLER_FROM_LAUNCHER=1"
    call "!ROOT!\install.bat"
    if errorlevel 1 (
        echo.
        echo Installation failed. Please read the message above and try again.
        pause
        exit /b 1
    )
    if not exist ".venv\Scripts\python.exe" (
        echo Installation did not complete. Please run install.bat again.
        pause
        exit /b 1
    )
)

if not exist "config.yaml" (
    echo Creating the local configuration file...
    copy /y "config.example.yaml" "config.yaml" >nul
    if errorlevel 1 (
        echo Could not create config.yaml.
        pause
        exit /b 1
    )
)
if not exist "logs" mkdir "logs"

REM Read configured ports. Defaults keep the package usable if the file is incomplete.
set "CONFIG_PORT=5173"
set "DASHBOARD_PORT=5174"
for /f "usebackq delims=" %%p in (`"!ROOT!\.venv\Scripts\python.exe" -c "import yaml; c=yaml.safe_load(open('config.yaml', encoding='utf-8')) or {}; print((c.get('configuration') or {}).get('port', 5173))" 2^>nul`) do set "CONFIG_PORT=%%p"
for /f "usebackq delims=" %%p in (`"!ROOT!\.venv\Scripts\python.exe" -c "import yaml; c=yaml.safe_load(open('config.yaml', encoding='utf-8')) or {}; print((c.get('dashboard') or {}).get('port', 5174))" 2^>nul`) do set "DASHBOARD_PORT=%%p"

REM Do not let child servers open duplicate browser tabs.
set "NO_AUTO_BROWSER=1"
set "DASHBOARD_PORT=!DASHBOARD_PORT!"

REM Start only services that are not already running.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=New-Object Net.Sockets.TcpClient; try {$c.Connect('127.0.0.1',!CONFIG_PORT!); exit 0} catch {exit 1} finally {$c.Dispose()}" >nul 2>&1
if errorlevel 1 (
    echo Starting configuration service on port !CONFIG_PORT!...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=$env:DT_ROOT; $p=Start-Process -FilePath ($root + '\\.venv\\Scripts\\python.exe') -ArgumentList 'server\\config_server.py' -WorkingDirectory $root -WindowStyle Minimized -RedirectStandardOutput ($root + '\\logs\\config_server.log') -RedirectStandardError ($root + '\\logs\\config_server.error.log') -PassThru; $p.Id | Set-Content -Encoding ascii ($root + '\\.config_server.pid')"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=New-Object Net.Sockets.TcpClient; try {$c.Connect('127.0.0.1',!DASHBOARD_PORT!); exit 0} catch {exit 1} finally {$c.Dispose()}" >nul 2>&1
if errorlevel 1 (
    echo Starting dashboard service on port !DASHBOARD_PORT!...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=$env:DT_ROOT; $p=Start-Process -FilePath ($root + '\\.venv\\Scripts\\python.exe') -ArgumentList 'server\\dashboard_server.py' -WorkingDirectory $root -WindowStyle Minimized -RedirectStandardOutput ($root + '\\logs\\dashboard_server.log') -RedirectStandardError ($root + '\\logs\\dashboard_server.error.log') -PassThru; $p.Id | Set-Content -Encoding ascii ($root + '\\.dashboard_server.pid')"
)

echo Waiting for DataTaxonomy to start...
for /l %%i in (1,1,30) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=New-Object Net.Sockets.TcpClient; try {$c.Connect('127.0.0.1',!CONFIG_PORT!); exit 0} catch {exit 1} finally {$c.Dispose()}" >nul 2>&1
    if not errorlevel 1 goto :wait_dashboard
    timeout /t 1 /nobreak >nul
)
echo.
echo The Configuration page did not respond on port !CONFIG_PORT!.
echo Check logs\config_server.error.log for details.
pause
exit /b 1

:wait_dashboard
for /l %%i in (1,1,30) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=New-Object Net.Sockets.TcpClient; try {$c.Connect('127.0.0.1',!DASHBOARD_PORT!); exit 0} catch {exit 1} finally {$c.Dispose()}" >nul 2>&1
    if not errorlevel 1 goto :open_page
    timeout /t 1 /nobreak >nul
)

echo.
echo The Dashboard did not respond on port !DASHBOARD_PORT!.
echo Check logs\dashboard_server.error.log for details.
pause
exit /b 1

:open_page
REM New installations need the project setup page; later launches go to results.
set "OPEN_URL=http://localhost:!DASHBOARD_PORT!"
if not exist "saved_configs.json" set "OPEN_URL=http://localhost:!CONFIG_PORT!"
findstr /c:"Your Project Name" "config.yaml" >nul 2>&1 && set "OPEN_URL=http://localhost:!CONFIG_PORT!"
start "" "!OPEN_URL!"
echo.
echo DataTaxonomy is ready: !OPEN_URL!
endlocal
