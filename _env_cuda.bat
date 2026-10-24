@echo off
rem =====================================================================
rem  NVIDIA CUDA build environment (dev_cuda / build_cuda / check_cuda / test_cuda / serve_cuda).
rem    CUDA_PATH            toolkit root (set by the CUDA installer)
rem    CUDA_COMPUTE_CAP=86  compile the kernels for one GPU only (read from nvidia-smi when unset;
rem                         RTX 3050 Laptop = 8.6). Set it to build for another GPU than this one.
rem  CUDA 13's CCCL headers need MSVC's standard preprocessor, passed to nvcc as
rem  NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor (logis-center build_cargo.bat).
rem =====================================================================
call "%~dp0_env_msvc.bat"
if errorlevel 1 exit /b 1

where nvcc > nul 2>&1
if errorlevel 1 (
    if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "PATH=%CUDA_PATH%\bin;%PATH%"
)
where nvcc > nul 2>&1
if errorlevel 1 (
    echo [CUDA] nvcc not found. Install the CUDA toolkit or set CUDA_PATH.
    exit /b 1
)
for /f "delims=" %%I in ('where cl.exe') do (
    if not defined NVCC_CCBIN set "NVCC_CCBIN=%%I"
)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
if not defined CUDA_COMPUTE_CAP (
    for /f "usebackq tokens=1,2 delims=." %%A in (`nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul`) do (
        if not defined CUDA_COMPUTE_CAP set "CUDA_COMPUTE_CAP=%%A%%B"
    )
)
set "CUDA_COMPUTE_CAP=%CUDA_COMPUTE_CAP: =%"
for /f "delims=" %%V in ('nvcc --version ^| findstr /c:"release"') do echo [CUDA] %%V
echo [CUDA] NVCC_CCBIN=%NVCC_CCBIN%
echo [CUDA] CUDA_COMPUTE_CAP=%CUDA_COMPUTE_CAP%
if not defined HAND_DEVICE set "HAND_DEVICE=cuda"
exit /b 0
