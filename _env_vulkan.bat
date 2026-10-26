@echo off
rem =====================================================================
rem  Vulkan build environment (dev_vulkan / build_vulkan / check_vulkan / test_vulkan / serve_vulkan).
rem  Only the Vulkan loader is needed at run time (vulkan-1.dll ships with the GPU driver).
rem  Without a GPU, CANDLE_VULKAN_ALLOW_CPU=1 lets the tests use a CPU Vulkan device
rem  (llvmpipe / lavapipe / WARP).
rem =====================================================================
call "%~dp0_env_msvc.bat"
if errorlevel 1 exit /b 1

if not exist "%SystemRoot%\System32\vulkan-1.dll" (
    echo [Vulkan] vulkan-1.dll not found. It ships with the GPU driver; without it the models run on the CPU.
)
if not defined NOTO_DEVICE set "NOTO_DEVICE=vulkan"
exit /b 0
