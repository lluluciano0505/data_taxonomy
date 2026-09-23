@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Stop DataTaxonomy
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "STOPPED=0"
for %%f in (".config_server.pid" ".dashboard_server.pid") do (
    if exist "%%~f" (
        set /p PID=<"%%~f"
        if defined PID (
            taskkill /PID !PID! /T /F >nul 2>&1
            if not errorlevel 1 set "STOPPED=1"
        )
        del /q "%%~f" >nul 2>&1
    )
)

REM Clean up services started without a PID file only when their command line points here.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=[IO.Path]::GetFullPath($env:ROOT).TrimEnd('\'); Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | ? { $_.CommandLine -and $_.CommandLine.Contains($root) -and ($_.CommandLine.Contains('config_server.py') -or $_.CommandLine.Contains('dashboard_server.py')) } | %% { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

if "!STOPPED!"=="1" (
    echo DataTaxonomy services stopped.
) else (
    echo No DataTaxonomy services were running.
)
endlocal
