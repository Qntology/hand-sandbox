@echo off
rem =====================================================================
rem  One entry point behind the dev_* / build_* / check_* / test_* / serve_* scripts.
rem    _cargo.bat <cuda|rocm|vulkan|cpu> <dev|build|check|test|serve> [extra cargo arguments]
rem  Sets up the toolchain of the backend (_env_<backend>.bat), the cargo feature flags,
rem  and NOTO_DEVICE (which device the app and the tests open; override it before calling).
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

rem Feature flags: the app forwards them to noto-runtime, noto-vision and noto-ml.
set "FEAT="
set "WSFEAT="
set "PKGFEAT="
if /i not "%BACKEND%"=="cpu" (
    set "FEAT=--features %BACKEND%"
    set "WSFEAT=--features noto-ml/%BACKEND%,noto-vision/%BACKEND%,noto-runtime/%BACKEND%,noto-sim-app/%BACKEND%"
    rem For one package (-p): every member with models depends on noto-ml and noto-vision.
    set "PKGFEAT=--features noto-ml/%BACKEND%,noto-vision/%BACKEND%"
)
set "RUST_BACKTRACE=1"
cd /d "%~dp0"

rem AI edits run the SolidCraft engine out of sight; its web package is not in git. Without it the runtime exports designs itself.
if /i not "%ACTION%"=="check" if /i not "%ACTION%"=="test" if not exist "ui\vendor\solidcraft\index.html" call "%~dp0fetch_solidcraft.bat"

if /i "%ACTION%"=="dev" goto :dev
if /i "%ACTION%"=="build" goto :build
if /i "%ACTION%"=="check" goto :check
if /i "%ACTION%"=="test" goto :test
if /i "%ACTION%"=="serve" goto :serve
goto :usage

:dev
cd /d "%~dp0src-tauri"
echo [DEV] Tauri app, backend %BACKEND% (NOTO_DEVICE=%NOTO_DEVICE%)
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
echo [SERVE] development server http://127.0.0.1:8787/ , backend %BACKEND% (NOTO_DEVICE=%NOTO_DEVICE%)
cargo run --release -p noto-runtime --example serve %FEAT% %ARGS%
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
