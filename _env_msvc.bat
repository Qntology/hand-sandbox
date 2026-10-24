@echo off
rem =====================================================================
rem  Shared by the _env_*.bat files: finds the Visual Studio C++ tools and
rem  loads vcvars64 into this console (logis-center / endmill-sandbox probe).
rem    VCVARS_VER=14.44   pick an MSVC toolset (the HIP SDK 7.x wants 14.44)
rem  Sets TAURI_CLI and PYTHONIOENCODING. Exit code 1 when no compiler is found.
rem =====================================================================
set "VS_SCRIPT="
set "VS_ARGS="
set "VS2022=%ProgramFiles%\Microsoft Visual Studio\2022"
for %%E in (Community Professional Enterprise BuildTools) do call :probe_vs "%VS2022%\%%E"
if not defined VS_SCRIPT call :probe_vs "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools"
if not defined VS_SCRIPT call :probe_vs "%ProgramFiles%\Microsoft Visual Studio\18\Community"
if not defined VS_SCRIPT call :probe_vs "%ProgramFiles%\Microsoft Visual Studio\18\BuildTools"
if not defined VS_SCRIPT call :probe_vswhere
if not defined VS_SCRIPT (
    echo [MSVC] Visual Studio C++ Build Tools not found. Install the "Desktop development with C++" workload.
    exit /b 1
)
set "VCVARS_ARG="
if defined VCVARS_VER set "VCVARS_ARG=-vcvars_ver=%VCVARS_VER%"
rem Per-call log folder: two consoles running this at once must not share one file.
call :unique_tmp VCVARS_DIR
set "VCVARS_LOG=%VCVARS_DIR%\vcvars_out.txt"
call "%VS_SCRIPT%" %VS_ARGS% %VCVARS_ARG% > "%VCVARS_LOG%" 2>&1
echo [MSVC] %VS_SCRIPT%
echo [MSVC] requested=%VCVARS_VER% active=%VCToolsVersion%
where cl.exe > nul 2>&1
if errorlevel 1 (
    type "%VCVARS_LOG%"
    del "%VCVARS_LOG%" > nul 2>&1
    rmdir "%VCVARS_DIR%" > nul 2>&1
    echo [MSVC] cl.exe is not on PATH after vcvars64.
    exit /b 1
)
del "%VCVARS_LOG%" > nul 2>&1
rmdir "%VCVARS_DIR%" > nul 2>&1
set "PYTHONIOENCODING=utf-8"

rem LanceDB's build needs protoc. Use the one on PATH, else a copy under tools\.
if not defined PROTOC (
    where protoc > nul 2>&1
    if errorlevel 1 (
        if not exist "%~dp0tools\protoc\bin\protoc.exe" (
            echo [SETUP] protoc not found - downloading protoc 29.3 for the LanceDB build...
            powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force -Path '%~dp0tools' | Out-Null; Invoke-WebRequest -Uri 'https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-win64.zip' -OutFile '%~dp0tools\protoc.zip' -UseBasicParsing; Expand-Archive -Force '%~dp0tools\protoc.zip' '%~dp0tools\protoc'"
        )
        if exist "%~dp0tools\protoc\bin\protoc.exe" set "PROTOC=%~dp0tools\protoc\bin\protoc.exe"
    )
)

set "TAURI_CLI=cargo tauri"
if exist "%~dp0node_modules\.bin\tauri.cmd" set TAURI_CLI="%~dp0node_modules\.bin\tauri.cmd"
exit /b 0

:probe_vs
if defined VS_SCRIPT exit /b 0
if exist "%~1\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_SCRIPT=%~1\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 0
)
if exist "%~1\Common7\Tools\VsDevCmd.bat" (
    set "VS_SCRIPT=%~1\Common7\Tools\VsDevCmd.bat"
    set "VS_ARGS=-arch=x64 -host_arch=x64"
)
exit /b 0

:probe_vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call :probe_vs "%%I"
exit /b 0

:unique_tmp
rem Creates a fresh folder under %TEMP% and stores its path in the variable named by %1.
rem mkdir is atomic, so consoles that start in the same second (same %RANDOM% seed) cannot share it.
set "UT_DIR=%TEMP%\hand_env_%RANDOM%%RANDOM%"
mkdir "%UT_DIR%" > nul 2>&1 || goto :unique_tmp
set "%~1=%UT_DIR%"
set "UT_DIR="
exit /b 0
