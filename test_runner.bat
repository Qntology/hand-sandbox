@echo off
rem ============================================================
rem  Sequential test run shared by every backend (endmill-sandbox test_runner.bat).
rem    test_cuda.bat   NVIDIA CUDA        test_rocm.bat  AMD ROCm / HIP
rem    test_vulkan.bat Vulkan             test_cpu.bat   CPU only
rem  _cargo.bat sets the toolchain, BACKEND, WSFEAT (cargo feature flags) and NOTO_DEVICE,
rem  then calls this file. Do not run it directly.
rem
rem  The game logic, the physics (balance, joint dynamics, N:N references) and the
rem  stand-in models run on the CPU for every backend, so steps 02-06 must give the same
rem  results everywhere. The backend changes the models only (SigLIP2, TIPS, SAM, TTM);
rem  step 07 compares them with the CPU. No step measures speed.
rem
rem  Every step writes its own log:  test_logs\<timestamp>_<backend>\NN_name.log
rem  Failures do not stop the run. At the end:
rem    summary.log     exit code and duration of every step, test results, check counts
rem    highlights.log  every check line ([GPU] [NN] [NN-SUM] [PRESET] [DYN])
rem  Options (set before running the wrapper):
rem    NOPAUSE=1          skip the final pause
rem    ONLY=03,07         run only these step numbers (step 01 build always runs)
rem    CARGO_BUILD_JOBS=2 parallel rustc/link jobs
rem ============================================================
setlocal EnableExtensions
if not defined BACKEND set "BACKEND=cpu"
set "RUST_BACKTRACE=1"
rem Two parallel rustc/link jobs by default: with all cores at once a 7 GB machine runs out of
rem virtual memory (os error 1455) while several links map the 160 MB lancedb rlib.
if not defined CARGO_BUILD_JOBS set "CARGO_BUILD_JOBS=2"
set "CT=cargo test %WSFEAT%"
rem One package with models (noto-vision, noto-runtime): only its own dependencies take features.
set "PT=cargo test %PKGFEAT%"
rem Physics and data crates do not depend on the models: no GPU features.
set "CPT=cargo test"
cd /d "%~dp0"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOGDIR=%CD%\test_logs\%TS%_%BACKEND%"
mkdir "%LOGDIR%"
set "SUMMARY=%LOGDIR%\summary.log"
set "FAILED=0"
>"%SUMMARY%" echo ========================================
>>"%SUMMARY%" echo  noto-sandbox test run %TS%  backend=%BACKEND%  features=%WSFEAT%  NOTO_DEVICE=%NOTO_DEVICE%
>>"%SUMMARY%" echo  options: ONLY=%ONLY% CARGO_BUILD_JOBS=%CARGO_BUILD_JOBS%
>>"%SUMMARY%" echo ========================================
echo Logs: %LOGDIR%

rem ---- 00: environment snapshot ----
(
  echo --- date ---
  echo %DATE% %TIME%
  echo --- git ---
  git rev-parse HEAD
  git status --short
  echo --- rustc / cargo ---
  rustc -Vv
  cargo -V
  echo --- backend %BACKEND% ---
  if /i "%BACKEND%"=="cuda" (
    nvcc --version
    nvidia-smi
  )
  if /i "%BACKEND%"=="rocm" (
    if exist "%ROCM_PATH%\bin\hipInfo.exe" "%ROCM_PATH%\bin\hipInfo.exe"
  )
  if /i "%BACKEND%"=="vulkan" (
    if exist "%SystemRoot%\System32\vulkan-1.dll" (echo vulkan-1.dll present) else (echo vulkan-1.dll MISSING - the models fall back to the CPU)
    where vulkaninfo >nul 2>nul && vulkaninfo --summary
  )
  echo --- env ---
  echo BACKEND=%BACKEND%
  echo WSFEAT=%WSFEAT%
  echo NOTO_DEVICE=%NOTO_DEVICE%
  echo CUDA_COMPUTE_CAP=%CUDA_COMPUTE_CAP%
  echo ROCM_PATH=%ROCM_PATH%
) > "%LOGDIR%\00_env.log" 2>&1

rem ---- 01: build every test binary first (separates compile errors from test failures) ----
call :run 01 build_tests "%CT% --workspace --no-run"
rem ---- 02: unit tests of every crate (physics, presets, balance, SDS, models with random weights) ----
call :run 02 unit "%CT% --workspace --lib --no-fail-fast -- --nocapture --test-threads=1"
rem ---- 03: joint dynamics and presets: every preset change must reach the physics ----
call :run 03 presets_physics "%CPT% -p noto-core --test presets_physics --no-fail-fast -- --nocapture --test-threads=1"
rem ---- 04: N:N composite scenarios against the research references (no tuning to them) ----
call :run 04 nn_scenarios "%CPT% -p noto-core --test nn_scenarios --no-fail-fast -- --nocapture --test-threads=1"
rem ---- 05: world scenarios (quests, game logic) ----
call :run 05 scenarios "%PT% -p noto-runtime --test scenarios --no-fail-fast -- --nocapture --test-threads=1"
rem ---- 06: integrated test (game pass, ML perception pass, ML decision pass) ----
call :run 06 integrated "%PT% -p noto-runtime --test integrated --no-fail-fast -- --nocapture --test-threads=1"
rem ---- 07: the models on the GPU backend compared with the CPU (skipped on the cpu backend) ----
if /i "%BACKEND%"=="cpu" (
    >>"%SUMMARY%" echo 07_gpu_parity: skipped - cpu backend
) else (
    call :run 07 gpu_parity "%PT% -p noto-vision --test gpu_backend --no-fail-fast -- --nocapture --test-threads=1"
)
rem ---- 08: Lance tables ----
call :run 08 lance "%CPT% -p noto-data --no-fail-fast -- --test-threads=1"

>>"%SUMMARY%" echo ========================================
if "%FAILED%"=="0" (
    >>"%SUMMARY%" echo ALL STEPS PASSED
) else (
    >>"%SUMMARY%" echo STEPS FAILED: %FAILED%
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test_summary.ps1" -LogDir "%LOGDIR%" -Backend "%BACKEND%"
echo.
type "%SUMMARY%"
echo.
echo Logs saved in: %LOGDIR%
if not defined NOPAUSE pause
endlocal & exit /b %FAILED%

rem ------------------------------------------------------------
rem  :run <NN> <name> "<command>"
rem ------------------------------------------------------------
:run
set "ID=%~1"
set "STEP=%~1_%~2"
set "CMDLINE=%~3"
set "STEPLOG=%LOGDIR%\%STEP%.log"
if not defined ONLY goto :run_go
if "%ID%"=="01" goto :run_go
echo ,%ONLY%, | findstr /C:",%ID%," >nul
if errorlevel 1 (
    echo [%STEP%] skipped - not in ONLY=%ONLY%
    >>"%SUMMARY%" echo %STEP%: skipped - ONLY
    exit /b 0
)
:run_go
echo [%STEP%] %CMDLINE%
for /f %%i in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "T0=%%i"
>"%STEPLOG%" echo [%TIME%] start %STEP%  backend=%BACKEND%
>>"%STEPLOG%" echo $ %CMDLINE%
>>"%STEPLOG%" echo.
%CMDLINE% >> "%STEPLOG%" 2>&1
set "RC=%ERRORLEVEL%"
for /f %%i in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "T1=%%i"
set /a DUR=T1-T0
>>"%STEPLOG%" echo.
>>"%STEPLOG%" echo [%TIME%] end %STEP% exit=%RC% (%DUR% s)
if not "%RC%"=="0" set /a FAILED+=1
>>"%SUMMARY%" echo %STEP%: exit=%RC%  %DUR% s
findstr /B /C:"test result:" "%STEPLOG%" >> "%SUMMARY%" 2>nul
findstr /C:"FAILED" /C:"panicked at" "%STEPLOG%" >> "%SUMMARY%" 2>nul
echo   -^> exit=%RC%  %DUR% s  (%STEPLOG%)
exit /b 0
