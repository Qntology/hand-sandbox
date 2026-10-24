@echo off
rem =====================================================================
rem  One entry point behind the dev_* / build_* / check_* / test_* / serve_* scripts.
rem    _cargo.bat <cuda|rocm|vulkan|cpu> <dev|build|check|test|serve> [extra cargo arguments]
rem  Sets up the toolchain of the backend (_env_<backend>.bat), the cargo feature flags,
rem  and HAND_DEVICE (which device the app and the tests open; override it before calling).
rem =====================================================================
chcp 65001 > nul
setlocal EnableExtensions
set "BACKEND=%~1"
set "ACTION=%~2"
if "%BACKEND%"=="" goto :usage
if "%ACTION%"=="" goto :usage
if not exist "%~dp0_env_%BACKEND%.bat" goto :usage
set "ARGS="
shift
shift
:collect
if "%~1"=="" goto :collected
set ARGS=%ARGS% %1
shift
goto :collect
:collected

call "%~dp0_env_%BACKEND%.bat"
if errorlevel 1 exit /b 1

rem Feature flags: the app forwards them to hand-runtime, hand-vision and hand-ml.
set "FEAT="
set "WSFEAT="
set "PKGFEAT="
if /i not "%BACKEND%"=="cpu" (
    set "FEAT=--features %BACKEND%"
    set "WSFEAT=--features hand-ml/%BACKEND%,hand-vision/%BACKEND%,hand-runtime/%BACKEND%,hand-sim-app/%BACKEND%"
    rem For one package (-p): every member with models depends on hand-ml and hand-vision.
    set "PKGFEAT=--features hand-ml/%BACKEND%,hand-vision/%BACKEND%"
)
set "RUST_BACKTRACE=1"
cd /d "%~dp0"

if /i "%ACTION%"=="dev" goto :dev
if /i "%ACTION%"=="build" goto :build
if /i "%ACTION%"=="check" goto :check
if /i "%ACTION%"=="test" goto :test
if /i "%ACTION%"=="serve" goto :serve
goto :usage

:dev
cd /d "%~dp0src-tauri"
echo [DEV] Tauri app, backend %BACKEND% (HAND_DEVICE=%HAND_DEVICE%)
call %TAURI_CLI% dev -- %FEAT% %ARGS%
goto :end

:build
cd /d "%~dp0src-tauri"
echo [BUILD] Tauri app bundle, backend %BACKEND%
call %TAURI_CLI% build -- %FEAT% %ARGS%
goto :end

:check
echo [CHECK] cargo check --workspace --all-targets %WSFEAT%
cargo check --workspace --all-targets %WSFEAT% %ARGS%
goto :end

:serve
echo [SERVE] development server http://127.0.0.1:8787/ , backend %BACKEND% (HAND_DEVICE=%HAND_DEVICE%)
cargo run --release -p hand-runtime --example serve %FEAT% %ARGS%
goto :end

:test
if not "%ARGS%"=="" (
    echo [TEST] cargo test %PKGFEAT% %ARGS%
    cargo test %PKGFEAT% %ARGS%
    goto :end
)
call "%~dp0test_runner.bat"
goto :end

:usage
echo usage: _cargo.bat ^<cuda^|rocm^|vulkan^|cpu^> ^<dev^|build^|check^|test^|serve^> [cargo arguments]
endlocal & exit /b 2

:end
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
