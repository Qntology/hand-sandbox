@echo off
rem =====================================================================
rem  AMD ROCm / HIP build environment (dev_rocm / build_rocm / check_rocm / test_rocm / serve_rocm).
rem  Same rules as the logis-center commerce scripts. Two ROCm tracks side by side:
rem    legacy : AMD HIP SDK 6.x / 7.x     C:\Program Files\AMD\ROCm\7.2
rem    core   : ROCm Core SDK 10.x        tarball extracted to C:\Program Files\AMD\ROCm\10.1
rem                                       (or anywhere listed in ROCM_SEARCH_PATHS), or rocm-sdk (pip)
rem  Choose with (first match wins):
rem    ROCM_VER=7.2 / 10.1          a folder under C:\Program Files\AMD\ROCm (core-10.1 is also tried)
rem    ROCM_PATH=<root>             any installation
rem    CANDLE_ROCM_TRACK=auto       (default) core when every GPU of this PC is supported by ROCm 10.1,
rem                                 otherwise legacy. legacy / core force a track.
rem  MSVC toolset: VCVARS_VER (default 14.44, which the HIP SDK 7.x headers build with).
rem =====================================================================
if not defined VCVARS_VER set "VCVARS_VER=14.44"
call "%~dp0_env_msvc.bat"
if errorlevel 1 (
    rem The pinned toolset may be missing; try the newest one before giving up.
    set "VCVARS_VER="
    call "%~dp0_env_msvc.bat"
    if errorlevel 1 exit /b 1
)

rem Values set by a previous run in this console are not user choices.
if defined ROCM_ENV_AUTO_VER if "%ROCM_VER%"=="%ROCM_ENV_AUTO_VER%" set "ROCM_VER="
if defined ROCM_ENV_AUTO_PATH if "%ROCM_PATH%"=="%ROCM_ENV_AUTO_PATH%" set "ROCM_PATH="
set "ROCM_ENV_AUTO_VER="
set "ROCM_ENV_AUTO_PATH="
set "ROCM_TRACK="
set "ROCM_HIP_VER="
set "ROCM_SEL_NOTE="

if defined ROCM_VER goto :rocm_from_ver
if defined ROCM_PATH goto :rocm_from_path
goto :rocm_auto

:rocm_from_ver
set "ROCM_PATH=%ProgramFiles%\AMD\ROCm\%ROCM_VER%"
if not exist "%ROCM_PATH%\bin" if exist "%ProgramFiles%\AMD\ROCm\core-%ROCM_VER%\bin" set "ROCM_PATH=%ProgramFiles%\AMD\ROCm\core-%ROCM_VER%"
set "ROCM_ENV_AUTO_PATH=%ROCM_PATH%"
goto :rocm_check

:rocm_from_path
if "%ROCM_PATH:~-1%"=="\" set "ROCM_PATH=%ROCM_PATH:~0,-1%"
for %%I in ("%ROCM_PATH%") do set "ROCM_VER=%%~nxI"
set "ROCM_ENV_AUTO_VER=%ROCM_VER%"
goto :rocm_check

:rocm_auto
set "ROCM_SEL_ROOT="
set "ROCM_SEL_VER="
set "ROCM_SEL_TRACK="
set "ROCM_SEL_HIP="
set "ROCM_SEL_ARCHS="
for /f "usebackq tokens=1,* delims==" %%A in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_rocm_select.ps1"`) do set "ROCM_SEL_%%A=%%B"
if not defined ROCM_SEL_ROOT goto :rocm_not_found
set "ROCM_PATH=%ROCM_SEL_ROOT%"
set "ROCM_VER=%ROCM_SEL_VER%"
set "ROCM_TRACK=%ROCM_SEL_TRACK%"
set "ROCM_HIP_VER=HIP %ROCM_SEL_HIP%"
set "ROCM_ENV_AUTO_VER=%ROCM_VER%"
set "ROCM_ENV_AUTO_PATH=%ROCM_PATH%"
if defined ROCM_SEL_NOTE echo [ROCm] %ROCM_SEL_NOTE%
goto :rocm_check

:rocm_not_found
echo [ROCm] No usable ROCm / HIP SDK installation found (a compiler plus amdhip64.lib and rocblas.lib).
if defined ROCM_SEL_NOTE echo [ROCm] %ROCM_SEL_NOTE%
echo        Installed under "%ProgramFiles%\AMD\ROCm":
dir /b /ad "%ProgramFiles%\AMD\ROCm" 2>nul
echo        Set ROCM_VER, ROCM_PATH or ROCM_SEARCH_PATHS.
exit /b 1

:rocm_check
if not exist "%ROCM_PATH%\bin\hipcc.exe" if not exist "%ROCM_PATH%\lib\llvm\bin\clang++.exe" (
    echo [ROCm] HIP SDK %ROCM_VER% not found: %ROCM_PATH%
    echo        Installed versions:
    dir /b /ad "%ProgramFiles%\AMD\ROCm" 2>nul
    exit /b 1
)
if not defined ROCM_TRACK (
    set "ROCM_TRACK=legacy"
    findstr /r /c:"define HIP_VERSION_MINOR  *[1-9][0-9]" "%ROCM_PATH%\include\hip\hip_version.h" >nul 2>&1 && set "ROCM_TRACK=core"
)
set "HIP_PATH=%ROCM_PATH%"
set "HIPCC="
if exist "%ROCM_PATH%\bin\hipcc.exe" set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%ROCM_PATH%\lib\llvm\bin" set "PATH=%PATH%;%ROCM_PATH%\lib\llvm\bin"
echo [ROCm] %ROCM_TRACK% track: %ROCM_PATH% %ROCM_HIP_VER%
if not defined HAND_DEVICE set "HAND_DEVICE=rocm"
exit /b 0
