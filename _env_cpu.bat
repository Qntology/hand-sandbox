@echo off
rem CPU-only build environment (no GPU feature; every model runs on the CPU).
call "%~dp0_env_msvc.bat"
if errorlevel 1 exit /b 1
set "HAND_DEVICE=cpu"
exit /b 0
